# -*- coding: utf-8 -*-
"""
HuBERT CTC Probe Task（LightningCLI 兼容）
- 负责：加载/配置 HuggingFace HubertForCTC、优化器/调度器、日志指标（loss/WER/blank_ratio）
- 与 DataModule 协作：两者需使用一致的 processor/vocab.json（建议在 YAML 里给同一路径）

YAML 用法示例：
model:
  class_path: marble.tasks.ASR.probe.HuBERTCTCTask
  init_args:
    backbone: facebook/hubert-base-ls960      # 或 facebook/hubert-large-ll60k
    sampling_rate: 16000
    vocab_path: ./hubert_ctc/vocab.json       # 与 data.init_args.vocab_path 一致
    # 训练稳态
    lr: 5.0e-5
    warmup_ratio: 0.1
    use_bitsandbytes_adam8bit: false
    # 模型配置
    ctc_loss_reduction: mean
    attention_dropout: 0.0
    activation_dropout: 0.0
    feat_proj_dropout: 0.0
    hidden_dropout: 0.0
    final_dropout: 0.0
    mask_time_prob: 0.05
    layerdrop: 0.0
    freeze_fe: false
    grad_ckpt: false
"""

from typing import Any, Dict, List, Tuple

import torch
import lightning.pytorch as pl
from transformers import HubertForCTC, get_linear_schedule_with_warmup, Wav2Vec2Processor
from jiwer import wer

# 直接复用 datamodule 里创建 processor 的方法（或自行读取 vocab）
from marble.tasks.ASR.datamodule import create_processor


class HuBERTCTCTask(pl.LightningModule):
    """
    简洁版 HuBERT+CTC 任务：
      - 输入 batch 来自 DataModule.collator（含 input_values / attention_mask / labels / texts）
      - 训练：最小化 CTC loss
      - 验证/测试：计算 WER、blank_ratio，并打印部分样例
    """

    def __init__(
        self,
        backbone: str = "facebook/hubert-base-ls960",
        sampling_rate: int = 16000,
        vocab_path: str | None = None,

        # 优化器/调度器
        lr: float = 5e-5,
        warmup_ratio: float = 0.1,
        use_bitsandbytes_adam8bit: bool = False,

        # 模型/正则化配置
        ctc_loss_reduction: str = "mean",
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        feat_proj_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        final_dropout: float = 0.0,
        mask_time_prob: float = 0.05,
        layerdrop: float = 0.0,
        freeze_fe: bool = False,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert vocab_path is not None, "请为 Task 指定 vocab_path（需与 DataModule 保持一致）"
        self.processor: Wav2Vec2Processor = create_processor(vocab_path, sampling_rate)

        # 1) 模型
        self.model = HubertForCTC.from_pretrained(
            backbone,
            vocab_size=len(self.processor.tokenizer),
            ctc_loss_reduction=ctc_loss_reduction,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        cfg = self.model.config
        cfg.pad_token_id = self.processor.tokenizer.pad_token_id
        cfg.vocab_size = len(self.processor.tokenizer)
        cfg.ctc_zero_infinity = True

        # dropout/specaug
        cfg.attention_dropout = attention_dropout
        cfg.activation_dropout = activation_dropout
        cfg.feat_proj_dropout = feat_proj_dropout
        cfg.hidden_dropout = hidden_dropout
        cfg.final_dropout = final_dropout
        cfg.mask_time_prob = mask_time_prob
        cfg.layerdrop = layerdrop

        # lm_head 尺寸安全对齐
        if self.model.lm_head.out_features != len(self.processor.tokenizer):
            self.model.lm_head = torch.nn.Linear(cfg.hidden_size, len(self.processor.tokenizer))

        # gradient checkpointing / freeze FE
        if grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable()
                cfg.gradient_checkpointing = True
                self.print("Gradient checkpointing enabled.")
            except Exception as e:
                self.print(f"Enable gradient checkpointing failed: {e}")
        if freeze_fe:
            for p in self.model.hubert.feature_extractor.parameters():
                p.requires_grad = False
            self.print("Feature extractor frozen.")

        # 2) 优化器超参
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.use_bnb = use_bitsandbytes_adam8bit

        # 评估缓存
        self.example_printed = False  # 仅打印一次样例

    # --------- Lightning hooks ---------

    def forward(self, input_values, attention_mask=None):
        return self.model(input_values=input_values, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        # collator 可能返回 texts（仅用于调试），这里去掉
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = self.model(**batch)
        loss = out.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["input_values"].size(0))
        return loss

    def _eval_common(self, batch, stage: str):
        texts = batch.get("texts", [])
        batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(input_values=batch["input_values"], attention_mask=batch.get("attention_mask")).logits
        pred_ids = torch.argmax(logits, dim=-1)

        # blank_ratio（按 tokenizer.pad_token_id 统计）
        pad_id = self.processor.tokenizer.pad_token_id
        blank_ratio = (pred_ids == pad_id).float().mean()

        # 解码 & WER
        pred_txt = self.processor.batch_decode(pred_ids)  # 带 '|'
        pred_txt = [s.replace('|', ' ') for s in pred_txt]
        # 将 labels 中 -100 mask 掉再解码
        with torch.no_grad():
            lab = batch["labels"].clone()
            lab[lab < 0] = self.processor.tokenizer.pad_token_id
        ref_txt = self.processor.batch_decode(lab, group_tokens=False)
        ref_txt = [s.replace('|', ' ') for s in ref_txt]

        # jiwer 逐 batch 计算（小批量 bias 可忽略）
        try:
            score = wer(ref_txt, pred_txt)
        except Exception:
            score = 1.0

        # 打印一个样例看 decoding
        if (not self.example_printed) and len(ref_txt) > 0:
            self.print(f"[{stage}] REF: {ref_txt[0]}")
            self.print(f"[{stage}] HYP: {pred_txt[0]}")
            self.example_printed = True

        # 日志
        bsz = batch["input_values"].size(0)
        self.log(f"{stage}/wer", score, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bsz)
        self.log(f"{stage}/blank_ratio", blank_ratio, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)

        return score

    def validation_step(self, batch, batch_idx):
        self._eval_common(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_common(batch, "test")

    def configure_optimizers(self):
        # Optimizer
        if self.use_bnb:
            try:
                import bitsandbytes as bnb
                optim = bnb.optim.Adam8bit(self.parameters(), lr=self.lr)
                self.print("Using bitsandbytes Adam8bit optimizer.")
            except Exception as e:
                self.print(f"bitsandbytes not available, fallback to AdamW: {e}")
                optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Scheduler（按 Lightning 的估计步数来设 total steps）
        # 注意：需要在 Trainer 已知 dataloader 后才能拿到 estimated_stepping_batches
        if self.trainer is not None and getattr(self.trainer, "estimated_stepping_batches", None):
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            # 回退：若拿不到，设 None 不使用 warmup（或你也可以读取 max_epochs * steps_per_epoch）
            total_steps = None

        if total_steps is not None:
            warmup_steps = int(self.warmup_ratio * total_steps)
            scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optim}
