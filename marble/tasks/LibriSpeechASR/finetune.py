# marble/tasks/LibriSpeechASR/finetune.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

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
        mask_time_prob: float = 0.0,
        layerdrop: float = 0.0,
        freeze_fe: bool = False,
        grad_ckpt: bool = False,

        # 对齐 asr.py：用于估算最小步数
        total_stride: int = 320,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert vocab_path is not None, "请为 Task 指定 vocab_path（需与 DataModule 使用的 vocab.json 一致）。"
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

        # 指标：与 asr.py 对齐
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
        
        # ---- 对齐 asr.py：epoch 聚合容器（val/test 各自用）----
        self._epoch_bucket = {
            "val": None,
            "test": None,
        }
        # bucket 结构：{
        #   "blank_sum": int, "total_tokens": int,
        #   "min_input_len_samples": Optional[int], "max_label_len": int,
        #   "raw_pred_sample": Optional[str], "pairs": List[Tuple[str, str]]
        # }
        
        print(f"[init] vocab_size={len(self.processor.tokenizer)}, "
               f"tokenizer.pad_id={self.processor.tokenizer.pad_token_id}, "
               f"model.pad_id(blank)={self.model.config.pad_token_id}, "
               f"ctc_zero_infinity={self.model.config.ctc_zero_infinity}")

    # --------- Lightning hooks ---------

    def forward(self, input_values, attention_mask=None):
        return self.model(input_values=input_values, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        # collator 可能返回 texts（仅用于调试），这里去掉
        batch_tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = self.model(**batch_tensors)
        loss = out.loss
        bsz = batch_tensors["input_values"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz, sync_dist=True)

        # 可选：打印一个样例
        if not self._printed_example["train"]:
            self._maybe_print_example(batch, stage="train", logits=out.logits.detach())
        return loss

    def on_validation_epoch_start(self):
        self.tm_wer_val.reset()
        self._reset_bucket("val")

    def on_test_epoch_start(self):
        self.tm_wer_test.reset()
        self._reset_bucket("test")

    def validation_step(self, batch, batch_idx):
        self._eval_common(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._eval_common(batch, stage="test")

    def on_validation_epoch_end(self):
        self._epoch_end_print_and_log("val")
        wer_val = float(self.tm_wer_val.compute().item())
        self.log("val/wer", wer_val, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        self._epoch_end_print_and_log("test")
        wer_test = float(self.tm_wer_test.compute().item())
        self.log("test/wer", wer_test, prog_bar=True, sync_dist=True)

    # --------- Eval helpers ---------

    @torch.no_grad()
    def _eval_common(self, batch, stage: str):
        bucket = self._epoch_bucket[stage]
        assert bucket is not None, f"{stage} bucket not initialized."

        batch_tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(
            input_values=batch_tensors["input_values"],
            attention_mask=batch_tensors.get("attention_mask")
        ).logits

        # 预测 ID 与 blank 统计
        pred_ids = torch.argmax(logits, dim=-1)
        pad_id = int(self.model.config.pad_token_id)
        blank_sum = (pred_ids == pad_id).sum().item()
        total_tokens = pred_ids.numel()
        bucket["blank_sum"] += int(blank_sum)
        bucket["total_tokens"] += int(total_tokens)

        # 长度统计（与 asr.py 一致）
        if batch_tensors.get("attention_mask") is not None:
            inp_len_samples = batch_tensors["attention_mask"].sum(-1)  # 每条样本的非 pad 采样点数
            b_min_inp = int(inp_len_samples.min().item())
            bucket["min_input_len_samples"] = (
                b_min_inp if bucket["min_input_len_samples"] is None
                else min(bucket["min_input_len_samples"], b_min_inp)
            )
        if batch_tensors.get("labels") is not None:
            lab_len = (batch_tensors["labels"] != -100).sum(-1)
            b_max_lab = int(lab_len.max().item())
            bucket["max_label_len"] = max(bucket["max_label_len"], b_max_lab)

        # 原始解码样例（仅保存一次）
        if bucket["raw_pred_sample"] is None:
            raw_pred = self.processor.batch_decode(pred_ids, group_tokens=False)
            if len(raw_pred) > 0:
                bucket["raw_pred_sample"] = raw_pred[0][:200]

        # 预测/参考解码成文本（用于 WER 与样例打印）
        pred_txt = self.processor.batch_decode(pred_ids)  # group_tokens=True（默认）
        pred_txt = [s.replace('|', ' ') for s in pred_txt]

        # 参考文本：首选 collator 提供的原始文本（含 '|'）
        if isinstance(batch.get("texts"), list) and len(batch["texts"]) > 0:
            ref_txt = [t.replace('|', ' ') for t in batch["texts"]]
        else:
            labels = batch_tensors["labels"].clone()
            labels[labels < 0] = pad_id
            ref_txt = self.processor.batch_decode(labels, group_tokens=False) 
            ref_txt = [s.replace('|', ' ') for s in ref_txt]

        # WER 更新
        if stage == "val":
            self.tm_wer_val.update(pred_txt, ref_txt)
        else:
            self.tm_wer_test.update(pred_txt, ref_txt)

        # 收集前 5 对样例
        if len(bucket["pairs"]) < 5:
            take = min(5 - len(bucket["pairs"]), len(ref_txt))
            for i in range(take):
                bucket["pairs"].append((ref_txt[i], pred_txt[i]))

        # step 级 blank_ratio（与旧实现保持；epoch 汇总会再打一次）
        bsz = batch_tensors["input_values"].size(0)
        batch_blank_ratio = (pred_ids == pad_id).float().mean()
        self.log(f"{stage}/blank_ratio", batch_blank_ratio, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)

        # 打印一个简短样例
        if not self._printed_example[stage]:
            self._maybe_print_example(batch, stage=stage, logits=logits.detach())

    # --------- Epoch end aggregate & print ---------

    def _epoch_end_print_and_log(self, stage: str):
        bucket = self._epoch_bucket[stage]
        if bucket is None:
            return

        # 分布式聚合（blank_sum/total_tokens 用求和；min/max 用 min/max）
        device = self.device if hasattr(self, "device") else torch.device("cpu")

        # 保护 None：用大数表示未见最短输入
        sentinel = torch.tensor([2**31 - 1], device=device, dtype=torch.long)
        min_inp = sentinel if bucket["min_input_len_samples"] is None else torch.tensor([bucket["min_input_len_samples"]], device=device, dtype=torch.long)
        max_lab = torch.tensor([bucket["max_label_len"]], device=device, dtype=torch.long)
        sums = torch.tensor([bucket["blank_sum"], bucket["total_tokens"]], device=device, dtype=torch.long)

        try:
            # all_gather 并在 rank0 汇总
            all_min = self.all_gather(min_inp)
            all_max = self.all_gather(max_lab)
            all_sums = self.all_gather(sums)
            # 形状处理（Lightning 在单卡时返回同形）
            g_min = int(all_min.view(-1).min().item())
            g_max = int(all_max.view(-1).max().item())
            g_blank = int(all_sums.view(-1, 2)[:, 0].sum().item())
            g_tokens = int(all_sums.view(-1, 2)[:, 1].sum().item())
        except Exception:
            # 无分布式或失败时，退化为本 rank 值
            g_min = int(min_inp.item()) if bucket["min_input_len_samples"] is not None else (2**31 - 1)
            g_max = int(max_lab.item())
            g_blank = int(sums[0].item())
            g_tokens = int(sums[1].item())

        blank_ratio_epoch = float(g_blank) / max(1, float(g_tokens))
        est_min_steps = (g_min // max(1, int(self.hparams.total_stride))) if g_min != (2**31 - 1) else 0

        # 计算 WER（torchmetrics）
        if stage == "val":
            wer_val = float(self.tm_wer_val.compute().item())
            # log 额外 epoch 级指标
            self.log("val/blank_ratio_epoch", blank_ratio_epoch, sync_dist=True, prog_bar=True)
            self.log("val/min_input_len_samples", float(0 if g_min == (2**31 - 1) else g_min), sync_dist=True)
            self.log("val/max_label_len", float(g_max), sync_dist=True)
            self.log("val/est_min_steps", float(est_min_steps), sync_dist=True)
            # 已在 validation_step 中 self.log("val/wer", self.tm_wer_val, ...)
        else:
            wer_val = float(self.tm_wer_test.compute().item())
            self.log("test/blank_ratio_epoch", blank_ratio_epoch, sync_dist=True, prog_bar=True)
            self.log("test/min_input_len_samples", float(0 if g_min == (2**31 - 1) else g_min), sync_dist=True)
            self.log("test/max_label_len", float(g_max), sync_dist=True)
            self.log("test/est_min_steps", float(est_min_steps), sync_dist=True)
            # 已在 test_step 中 self.log("test/wer", self.tm_wer_test, ...)

        # rank0 打印对齐 asr.py 的信息
        if getattr(self, "global_rank", 0) == 0:
            ep = int(self.current_epoch)
            if bucket["raw_pred_sample"] is not None:
                self.print(f"raw_pred_sample: {bucket['raw_pred_sample']}")
            for (ref, pred) in bucket["pairs"]:
                self.print(f"ref:  {ref}")
                self.print(f"pred: {pred}\n")
            self.print(f"Epoch {ep} blank_ratio: {blank_ratio_epoch:.4f}")
            self.print(f"Epoch {ep} min_input_len(samples): {None if g_min == (2**31 - 1) else g_min}, "
                       f"est_min_steps(~/ {int(self.hparams.total_stride)}): {est_min_steps}, "
                       f"max_label_len: {g_max}")
            self.print(f"Epoch {ep} WER: {wer_val:.4f}")

    def _reset_bucket(self, stage: str):
        self._epoch_bucket[stage] = {
            "blank_sum": 0,
            "total_tokens": 0,
            "min_input_len_samples": None,
            "max_label_len": 0,
            "raw_pred_sample": None,
            "pairs": [],
        }

    def _maybe_print_example(self, batch: Dict[str, Any], stage: str, logits: Optional[torch.Tensor] = None):
        try:
            texts = batch.get("texts", [])
            if isinstance(texts, list) and len(texts) > 0:
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
