# marble/tasks/LibriSpeechASR/finetune.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

import torch
import lightning.pytorch as pl
from transformers import HubertForCTC, get_linear_schedule_with_warmup, Wav2Vec2Processor
from torchmetrics.text import WordErrorRate

from marble.tasks.LibriSpeechASR.datamodule import create_processor


class HuBERTCTCTask(pl.LightningModule):
    """
    简洁稳定的 HuBERT+CTC 任务模块。
    期望 batch 来自 DataModule.collator，包含：
      - input_values: (B, T)
      - attention_mask: (B, T) or None
      - labels: (B, L)  其中 pad→-100
      - texts: List[str]（仅用于调试/日志）
    """

    def __init__(
        self,
        backbone: str = "facebook/hubert-base-ls960",
        sampling_rate: int = 16000,
        vocab_path: Optional[str] = None,

        # 优化器/调度器
        lr: float = 5e-5,
        warmup_steps: float = 2000,
        use_bitsandbytes_adam8bit: bool = False,

        # 模型/正则化配置（与原脚本一致）
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

        assert vocab_path is not None, "请为 Task 指定 vocab_path（需与 DataModule 使用的 vocab.json 一致）。"
        # 与 DataModule 共享相同 processor 设定
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
        self.tm_wer_val = WordErrorRate()
        self.tm_wer_test = WordErrorRate()

        # lm_head 尺寸安全对齐
        if self.model.lm_head.out_features != len(self.processor.tokenizer):
            self.model.lm_head = torch.nn.Linear(cfg.hidden_size, len(self.processor.tokenizer))

        # gradient checkpointing / freeze FE
        if grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable()
                cfg.gradient_checkpointing = True
                print("Gradient checkpointing enabled.")
            except Exception as e:
                print(f"Enable gradient checkpointing failed: {e}")
        if freeze_fe:
            for p in self.model.hubert.feature_extractor.parameters():
                p.requires_grad = False
            print("Feature extractor frozen.")

        # 2) 优化器与调度器参数
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.use_bnb = use_bitsandbytes_adam8bit

        # 打印样例控制
        self._printed_example = {"train": False, "val": False, "test": False}

    # --------- Lightning hooks ---------

    def forward(self, input_values, attention_mask=None):
        return self.model(input_values=input_values, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        # collator 可能返回 texts（仅用于调试），这里去掉
        batch_tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = self.model(**batch_tensors)
        loss = out.loss
        bsz = batch_tensors["input_values"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)
        # 可选：打印一个样例
        if not self._printed_example["train"]:
            self._maybe_print_example(batch, stage="train", logits=out.logits.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        self._eval_common(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._eval_common(batch, stage="test")
        
    def on_validation_epoch_start(self):
        self.tm_wer_val.reset()

    def on_test_epoch_start(self):
        self.tm_wer_test.reset()

    # --------- Eval helpers ---------

    @torch.no_grad()
    def _eval_common(self, batch, stage: str):
        batch_tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(
            input_values=batch_tensors["input_values"],
            attention_mask=batch_tensors.get("attention_mask")
        ).logits

        # 预测 ID 与 blank 比例（使用 tokenizer.pad_token_id 作为“空白”计数近似）
        pred_ids = torch.argmax(logits, dim=-1)
        pad_id = self.processor.tokenizer.pad_token_id
        blank_ratio = (pred_ids == pad_id).float().mean()

        # 预测/参考解码成文本
        pred_txt = self.processor.batch_decode(pred_ids, group_tokens=True)
        pred_txt = [s.replace('|', ' ') for s in pred_txt]

        # 参考文本：优先使用 collator 提供的“原始文本”（与 asr.py 一致，避免 PAD/blank 污染）
        if isinstance(batch.get("texts"), list) and len(batch["texts"]) > 0:
            ref_txt = [t.replace('|', ' ') for t in batch["texts"]]
        else:
            # 兜底：若没有 texts，再从 labels 反解，但要把 blank 去掉
            labels = batch_tensors["labels"].clone()
            labels[labels < 0] = pad_id
            # 注意：group_tokens=True 会移除 CTC blank（pad）并合并重复，得到干净序列
            ref_txt = self.processor.batch_decode(labels, group_tokens=True)
            ref_txt = [s.replace('|', ' ') for s in ref_txt]

        bsz = batch_tensors["input_values"].size(0)
        self.log(f"{stage}/blank_ratio", blank_ratio, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        
        if stage == "val":
            self.tm_wer_val.update(pred_txt, ref_txt)
            self.log("val/wer", self.tm_wer_val, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        else:
            self.tm_wer_test.update(pred_txt, ref_txt)
            self.log("test/wer", self.tm_wer_test, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        # 打印一个样例
        if not self._printed_example[stage]:
            self._maybe_print_example(batch, stage=stage, logits=logits.detach())

    def _maybe_print_example(self, batch: Dict[str, Any], stage: str, logits: Optional[torch.Tensor] = None):
        try:
            texts = batch.get("texts", [])
            if isinstance(texts, list) and len(texts) > 0:
                # 将 '|' 改回空格展示
                ref = texts[0].replace('|', ' ')
            else:
                ref = "(no text)"
            hyp = "(no pred)"
            if logits is not None and logits.ndim == 3 and logits.size(0) > 0:
                pred_ids = torch.argmax(logits[0], dim=-1, keepdim=True).T  # (1, T)
                hyp = self.processor.batch_decode(pred_ids, group_tokens=True)[0].replace('|', ' ')
            self.print(f"[{stage}] REF: {ref}")
            self.print(f"[{stage}] HYP: {hyp}")
            self._printed_example[stage] = True
        except Exception as e:
            self.print(f"Print example failed: {e}")

    # --------- Optim / Sched ---------

    def configure_optimizers(self):
        # Optimizer
        if self.use_bnb:
            try:
                import bitsandbytes as bnb
                optim = bnb.optim.Adam8bit(self.parameters(), lr=self.lr)
                self.print("Using bitsandbytes Adam8bit optimizer.")
            except Exception as e:
                self.print(f"bitsandbytes 不可用，回退 AdamW：{e}")
                optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # Scheduler（按 Lightning 的估计步数设置 warmup）
        total_steps = None
        if self.trainer is not None and getattr(self.trainer, "estimated_stepping_batches", None):
            total_steps = int(self.trainer.estimated_stepping_batches)

        if total_steps and total_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
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
