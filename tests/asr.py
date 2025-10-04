#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HuBERT + CTC 训练脚本（融合官方 tricks & 默参，含防塌缩 v3）
- 适配 OpenSLR 手动下载并解压的 LibriSpeech 目录：./LibriSpeech/train-clean-100
- 使用 torchaudio.datasets.LIBRISPEECH 读取（不再下载）

新增/融合要点：
  * 文本清洗：支持 chars_to_ignore（参考官方脚本），统一空格→'|', 大写
  * 词表策略：默认手写 vocab；可选 --auto_vocab 从数据集中抽字符表生成 vocab.json
  * Collator：支持 pad_to_multiple_of（提升 Tensor Cores 利用率），可选过滤 U>T 的样本
  * 模型 config 接口：attention/activation/feat_proj/hidden/final dropout, mask_time_prob, layerdrop, ctc_loss_reduction
  * 训练稳态：warmup、梯度裁剪、可选冻结特征提取器、可选 gradient checkpointing、可选 Adam8bit
  * 评估调试：blank_ratio、raw_pred_sample、输入/标签长度检查、WER
  * 时长过滤：--min_duration_sec/--max_duration_sec（仅过滤，不截断，避免 U>T）

用法示例：
    pip install -U torch torchaudio transformers jiwer soundfile
    # 如需8bit优化器：pip install bitsandbytes
    python asr4.py \
      --output_dir ./hubert_ctc --backbone facebook/hubert-base-ls960 \
      --epochs 20 --batch_size 32 --lr 5e-5 --warmup_ratio 0.1 \
      --freeze_fe --filter_long_labels --pad_to_multiple_of 8 \
      --mask_time_prob 0.05 --layerdrop 0.0 --hidden_dropout 0.0 \
      --adam8bit --grad_ckpt --auto_vocab --use_dev_clean

如果没有 dev-clean，本脚本会从 train-clean-100 随机切 5% 做验证。
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Set

import torch
from torch.utils.data import DataLoader, random_split
import torchaudio

from transformers import (
    HubertForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup,
)
from jiwer import wer


# -----------------------
# Argparse
# -----------------------

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_root', type=str, default='/aifs4su/mmdata/rawdata/codeclm/speech/librispeech', help='包含 ./LibriSpeech/ 的根目录')
    p.add_argument('--use_dev_clean', action='store_true', help='已解压 dev-clean 则加此项')
    p.add_argument('--min_duration_sec', type=float, default=0.0, help='过滤短于该秒数的样本（0 关闭）')
    p.add_argument('--max_duration_sec', type=float, default=0.0, help='过滤长于该秒数的样本（0 关闭）')

    # I/O
    p.add_argument('--output_dir', type=str, default='./hubert_ctc_from_openslr')
    p.add_argument('--overwrite_output_dir', action='store_true', help='若目录非空，是否覆盖')

    # Loader
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)

    # Train
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=5e-5, help='建议 5e-5~1e-4')
    p.add_argument('--warmup_ratio', type=float, default=0.1, help='线性 warmup 比例（0~1）')
    p.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值（范数）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true', help='开启 gradient checkpointing（省显存）')
    p.add_argument('--adam8bit', action='store_true', help='使用 bitsandbytes Adam8bit 优化器（需安装 bnb）')

    # Model
    p.add_argument('--backbone', type=str, default='facebook/hubert-large-ll60k')  # 可改为 facebook/hubert-base-ls960
    p.add_argument('--sampling_rate', type=int, default=16000)
    p.add_argument('--freeze_fe', action='store_true', help='冻结特征提取器（可选）')

    # Config drops/specaug（对齐官方可控项）
    p.add_argument('--ctc_loss_reduction', type=str, default='mean', choices=['mean', 'sum'])
    p.add_argument('--attention_dropout', type=float, default=0.0)
    p.add_argument('--activation_dropout', type=float, default=0.0)
    p.add_argument('--feat_proj_dropout', type=float, default=0.0)
    p.add_argument('--hidden_dropout', type=float, default=0.0)
    p.add_argument('--final_dropout', type=float, default=0.0)
    p.add_argument('--mask_time_prob', type=float, default=0.05)
    p.add_argument('--layerdrop', type=float, default=0.0)

    # Tokenizer/Vocab & text cleaning
    p.add_argument('--auto_vocab', action='store_true', help='从数据集自动抽字符集生成 vocab.json')
    p.add_argument('--chars_to_ignore', type=str, default=",.?;!:\"“”‘’'()[]{}-—–…`~@#$%^&*_+=|/\\<>", help='需要剔除的字符集合（字符串形式）')

    # Collator
    p.add_argument('--filter_long_labels', action='store_true', help='过滤标签长度超过下采样后输入步数的样本')
    p.add_argument('--total_stride', type=int, default=320, help='HuBERT/W2V2 卷积下采样步长乘积，默认 320')
    p.add_argument('--pad_to_multiple_of', type=int, default=8, help='输入 padding 对齐到该倍数（0/1 关闭）')

    return p.parse_args()


# -----------------------
# Utils
# -----------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_chars_regex(chars: str) -> Optional[re.Pattern]:
    if not chars:
        return None
    # 转义并编译为字符类
    escaped = ''.join(re.escape(c) for c in chars)
    return re.compile(f"[{escaped}]")


def clean_text(s: str, chars_re: Optional[re.Pattern]) -> str:
    # 去除不需要字符、收缩空格、转大写
    if chars_re is not None:
        s = chars_re.sub(' ', s)
    s = ' '.join(s.strip().split()).upper()
    return s


def build_vocab(save_dir: str, charset: Optional[Set[str]] = None) -> str:
    """构建 vocab.json；若给定 charset（已清洗后、含 '|' 或空格），则按其生成；否则用默认 A-Z + ' + '|'"""
    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, 'vocab.json')

    if charset is None:
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ["'"]
        vocab_tokens = letters + ['|']
    else:
        # 将空格统一映射为 '|'
        processed = set()
        for ch in charset:
            if ch == ' ':
                processed.add('|')
            elif ch == '\\t' or ch == '\\n':
                continue
            else:
                processed.add(ch)
        # 只保留 A-Z、'、|（超集可以扩展，这里与 tokenizer 对齐）
        keep = [c for c in sorted(processed) if (c == '|' or c == "'" or ('A' <= c <= 'Z'))]
        if '|' not in keep:
            keep.append('|')
        if "'" not in keep:
            keep.append("'")
        vocab_tokens = keep

    vocab = {c: i for i, c in enumerate(vocab_tokens)}
    vocab['[UNK]'] = len(vocab)
    vocab['[PAD]'] = len(vocab)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab_path


def create_processor(vocab_path: str, sampling_rate: int) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token='[UNK]',
        pad_token='[PAD]',
        word_delimiter_token='|',
        do_lower_case=False,
    )
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)


def get_librispeech_splits(root: str, use_dev_clean: bool, seed: int):
    train_ds = torchaudio.datasets.LIBRISPEECH(root=root, url='train-clean-100', download=False)
    if use_dev_clean:
        eval_ds = torchaudio.datasets.LIBRISPEECH(root=root, url='dev-clean', download=False)
        return train_ds, eval_ds
    size = len(train_ds)
    tr_len = int(size * 0.95)
    ev_len = size - tr_len
    tr, ev = random_split(train_ds, [tr_len, ev_len], generator=torch.Generator().manual_seed(seed))
    return tr, ev


def maybe_gather_charset(train_ds, eval_ds, chars_re: Optional[re.Pattern], max_items: Optional[int] = None) -> Set[str]:
    """遍历数据集抽取清洗后的字符集合（用于 --auto_vocab）"""
    charset: Set[str] = set()
    def add_from_ds(ds):
        n = len(ds)
        limit = n if max_items is None else min(n, max_items)
        for i in range(limit):
            _w, _sr, txt, *_ = ds[i]
            txt = clean_text(txt, chars_re)
            for ch in txt:
                charset.add(ch)
    add_from_ds(train_ds)
    add_from_ds(eval_ds)
    return charset


# -----------------------
# Collator
# -----------------------

@dataclass
class DataCollatorCTC:
    processor: Wav2Vec2Processor
    sampling_rate: int
    chars_re: Optional[re.Pattern]
    filter_long_labels: bool = False
    total_stride: int = 320
    pad_to_multiple_of: int = 8
    min_duration_sec: float = 0.0
    max_duration_sec: float = 0.0

    def __call__(self, batch: List):
        wave_list, txt_list = [], []
        for waveform, sr, transcript, *_ in batch:
            # 时长过滤（单位：秒）
            dur_sec = float(waveform.shape[-1]) / float(sr)
            if (self.min_duration_sec and dur_sec < self.min_duration_sec) or \
               (self.max_duration_sec and dur_sec > self.max_duration_sec):
                continue
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            wave_list.append(waveform.squeeze(0).numpy())
            txt_list.append(clean_text(transcript, self.chars_re))

        if len(wave_list) == 0:
            # 退化保护：若本 batch 全被过滤，至少放回一个空白样本（让外层处理）
            return {
                'input_values': torch.zeros((1, 1), dtype=torch.float32),
                'attention_mask': torch.ones((1, 1), dtype=torch.long),
                'labels': torch.full((1, 1), -100, dtype=torch.long),
                'texts': [],
            }

        # pad_to_multiple_of：0/1 视为关闭
        p2m = self.pad_to_multiple_of if self.pad_to_multiple_of and self.pad_to_multiple_of > 1 else None

        feats = self.processor(
            wave_list,
            sampling_rate=self.sampling_rate,
            return_tensors='pt',
            padding=True,
            pad_to_multiple_of=p2m,
        )

        # 将空格→'|'，并用 [PAD] 位置置 -100
        txt_list_bar = [t.replace(' ', '|') for t in txt_list]
        lab_batch = self.processor.tokenizer(text=txt_list_bar, return_tensors='pt', padding=True, add_special_tokens=False)
        labels = lab_batch['input_ids']
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        # （可选）过滤 U>T：label_len <= input_len//total_stride
        if self.filter_long_labels:
            inp_len_samples = feats['attention_mask'].sum(-1)  # 样本级长度（单位：采样点）
            down_len = torch.div(inp_len_samples, self.total_stride, rounding_mode='floor')
            lab_len = (labels != -100).sum(-1)
            valid = (lab_len <= down_len)
            if not valid.all():
                if valid.sum() == 0:
                    valid_idx = torch.tensor([int(torch.argmax(inp_len_samples).item())])  # 取该 batch 最长语音
                else:
                    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
                feats['input_values'] = feats['input_values'][valid_idx]
                if 'attention_mask' in feats and feats['attention_mask'] is not None:
                    feats['attention_mask'] = feats['attention_mask'][valid_idx]
                labels = labels[valid_idx]
                txt_list_bar = [txt_list_bar[i] for i in valid_idx.tolist()]

        return {
            'input_values': feats['input_values'],
            'attention_mask': feats.get('attention_mask'),
            'labels': labels,
            'texts': txt_list_bar,
        }


# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # 目录安全检查
    os.makedirs(args.output_dir, exist_ok=True)
    if (not args.overwrite_output_dir) and os.listdir(args.output_dir):
        raise ValueError(f"输出目录 {args.output_dir} 非空，且未指定 --overwrite_output_dir。为避免覆盖，请更换输出目录或加该参数。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    train_ds, eval_ds = get_librispeech_splits(args.data_root, args.use_dev_clean, args.seed)

    # 文本清洗 regex
    chars_re = make_chars_regex(args.chars_to_ignore)

    # 词表：可选从数据抽取
    if args.auto_vocab:
        charset = maybe_gather_charset(train_ds, eval_ds, chars_re)
        vocab_path = build_vocab(args.output_dir, charset)
    else:
        vocab_path = build_vocab(args.output_dir)

    processor = create_processor(vocab_path, args.sampling_rate)

    # 模型
    model = HubertForCTC.from_pretrained(
        args.backbone,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction=args.ctc_loss_reduction,
        pad_token_id=processor.tokenizer.pad_token_id,
    ).to(device)

    # Config 对齐与正则化超参（参考官方）
    cfg = model.config
    cfg.pad_token_id = processor.tokenizer.pad_token_id
    cfg.vocab_size = len(processor.tokenizer)
    cfg.ctc_zero_infinity = True
    # dropout/specaug
    cfg.attention_dropout = args.attention_dropout
    cfg.activation_dropout = args.activation_dropout
    cfg.feat_proj_dropout = args.feat_proj_dropout
    cfg.hidden_dropout = args.hidden_dropout
    cfg.final_dropout = args.final_dropout
    cfg.mask_time_prob = args.mask_time_prob
    cfg.layerdrop = args.layerdrop

    # lm_head 尺寸对齐
    if model.lm_head.out_features != len(processor.tokenizer):
        model.lm_head = torch.nn.Linear(cfg.hidden_size, len(processor.tokenizer)).to(device)
        cfg.vocab_size = len(processor.tokenizer)

    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
            cfg.gradient_checkpointing = True
            print('Gradient checkpointing enabled.')
        except Exception as e:
            print('Enable gradient checkpointing failed:', e)

    if args.freeze_fe:
        for p in model.hubert.feature_extractor.parameters():
            p.requires_grad = False
        print('Feature extractor frozen.')

    # Dataloaders
    collate_train = DataCollatorCTC(
        processor=processor,
        sampling_rate=args.sampling_rate,
        chars_re=chars_re,
        filter_long_labels=args.filter_long_labels,   # 训练可开
        total_stride=args.total_stride,
        pad_to_multiple_of=args.pad_to_multiple_of,
        min_duration_sec=args.min_duration_sec,       # 训练可按需开
        max_duration_sec=args.max_duration_sec,
    )

    collate_eval = DataCollatorCTC(
        processor=processor,
        sampling_rate=args.sampling_rate,
        chars_re=chars_re,
        filter_long_labels=False,                     # 评测统一关闭
        total_stride=args.total_stride,
        pad_to_multiple_of=args.pad_to_multiple_of,   # 这个开没问题，不改样本集合
        min_duration_sec=0.0,                         # 评测不做时长过滤
        max_duration_sec=0.0,
    )
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem, collate_fn=collate_train)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_mem, collate_fn=collate_eval)

    # Optimizer
    if args.adam8bit:
        try:
            import bitsandbytes as bnb
            optim = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
            print('Using bitsandbytes Adam8bit optimizer.')
        except Exception as e:
            print('bitsandbytes 不可用，回退到 AdamW：', e)
            optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler
    t_total = len(train_loader) * max(1, args.epochs)
    warmup_steps = int(args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # AMP
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16 and device.type == 'cuda')

    # 训练
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            # 若 batch 因过滤为空，跳过
            if batch['input_values'].numel() <= 1 and len(batch['texts']) == 0:
                continue

            batch.pop('texts', None)
            batch = {k: v.to(device) for k, v in batch.items()}

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
                out = model(**batch)
                loss = out.loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

            scheduler.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        # --- 简单评估 + 调试信息 ---
        model.eval()
        preds, refs = [], []
        blank_sum, total_tokens = 0, 0
        global_min_inp_len_samples = None
        global_max_lab_len = 0

        with torch.no_grad():
            for b_idx, batch in enumerate(eval_loader):
                if len(batch['texts']) == 0:
                    continue
                original_texts_bar = batch['texts']  # 带 '|'

                inp_len_samples = batch['attention_mask'].sum(-1)
                lab_len = (batch['labels'] != -100).sum(-1)
                b_min_inp = int(inp_len_samples.min().item())
                b_max_lab = int(lab_len.max().item())
                global_min_inp_len_samples = b_min_inp if global_min_inp_len_samples is None else min(global_min_inp_len_samples, b_min_inp)
                global_max_lab_len = max(global_max_lab_len, b_max_lab)

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.amp.autocast('cuda', enabled=args.fp16 and device.type == 'cuda'):
                    logits = model(input_values=batch['input_values'], attention_mask=batch['attention_mask']).logits

                pred_ids = torch.argmax(logits, dim=-1)

                pad_id = processor.tokenizer.pad_token_id
                blank_sum += (pred_ids == pad_id).sum().item()
                total_tokens += pred_ids.numel()

                raw_pred = processor.batch_decode(pred_ids, group_tokens=False)
                if b_idx == 0:
                    print('raw_pred_sample:', raw_pred[0][:200])

                pred_txt = processor.batch_decode(pred_ids)
                pred_txt = [s.replace('|', ' ') for s in pred_txt]

                preds.extend(pred_txt)
                refs.extend([s.replace('|', ' ') for s in original_texts_bar])

            n_show = min(5, len(refs))
            for i in range(n_show):
                print('ref:', refs[i])
                print('pred:', preds[i])
                print()

        blank_ratio = blank_sum / max(1, total_tokens)
        est_min_steps = (global_min_inp_len_samples // args.total_stride) if global_min_inp_len_samples is not None else 0
        print(f"Epoch {epoch} blank_ratio: {blank_ratio:.4f}")
        print(f"Epoch {epoch} min_input_len(samples): {global_min_inp_len_samples}, est_min_steps(~/ {args.total_stride}): {est_min_steps}, max_label_len: {global_max_lab_len}")
        print(f"Epoch {epoch} WER: {wer(refs, preds):.4f}")
        model.train()

    # 保存
    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    print('Saved to:', args.output_dir)


if __name__ == '__main__':
    main()
