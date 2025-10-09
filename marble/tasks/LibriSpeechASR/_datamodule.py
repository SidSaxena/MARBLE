# -*- coding: utf-8 -*-
"""
LibriSpeech CTC DataModule（LightningCLI 兼容）
- 复用了你脚本里的文本清洗、pad_to_multiple_of、U>T 过滤等逻辑
- Dataset 直接用 torchaudio.datasets.LIBRISPEECH
- collate 在 DataModule 中管理（train/val/test 各一份）
- 仍支持 BaseDataModule 的 transform 包装（如需做波形增强）

YAML 用法示例（与 LightningCLI 配合）：
data:
  class_path: marble.tasks.ASR.datamodule.LibriSpeechCTCDataModule
  init_args:
    batch_size: 16
    num_workers: 8
    sampling_rate: 16000
    vocab_path: ./hubert_ctc/vocab.json     # 建议显式给出
    chars_to_ignore: ",.?;!:\"“”‘’'()[]{}-—–…`~@#$%^&*_+=|/\\<>"
    pad_to_multiple_of: 8
    filter_long_labels: true
    total_stride: 320
    min_duration_sec: 0.0
    max_duration_sec: 0.0
    # 这三段 dataset 配置会由 BaseDataModule 用 instantiate_from_config 实例化
    train:
      class_path: marble.tasks.ASR.datamodule.LibriSpeechRaw
      init_args: { root: /path/to/LibriSpeech, url: train-clean-100, download: false }
    val:
      class_path: marble.tasks.ASR.datamodule.LibriSpeechRaw
      init_args: { root: /path/to/LibriSpeech, url: dev-clean, download: false }
    test:
      class_path: marble.tasks.ASR.datamodule.LibriSpeechRaw
      init_args: { root: /path/to/LibriSpeech, url: dev-clean, download: false }
"""

from dataclasses import dataclass
import os
import re
import json
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule


# -----------------------
# 文本清洗与 vocab
# -----------------------

def make_chars_regex(chars: str) -> Optional[re.Pattern]:
    if not chars:
        return None
    esc = ''.join(re.escape(c) for c in chars)
    return re.compile(f"[{esc}]")

def clean_text(s: str, chars_re: Optional[re.Pattern]) -> str:
    if chars_re is not None:
        s = chars_re.sub(' ', s)
    s = ' '.join(s.strip().split()).upper()
    return s

def build_vocab(save_dir: str, charset: Optional[set] = None) -> str:
    """
    若给定 charset（清洗后字符集，含 '|' 或空格），据此生成；否则用 A-Z + ' + '|'
    """
    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, 'vocab.json')
    if charset is None:
        tokens = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ["'", "|"]
    else:
        processed = set()
        for ch in charset:
            processed.add('|' if ch == ' ' else ch)
        keep = [c for c in sorted(processed) if (c == '|' or c == "'" or ('A' <= c <= 'Z'))]
        if '|' not in keep:
            keep.append('|')
        if "'" not in keep:
            keep.append("'")
        tokens = keep
    vocab = {c: i for i, c in enumerate(tokens)}
    vocab['[UNK]'] = len(vocab)
    vocab['[PAD]'] = len(vocab)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab_path

def create_processor(vocab_path: str, sampling_rate: int) -> Wav2Vec2Processor:
    tok = Wav2Vec2CTCTokenizer(
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
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)


# -----------------------
# Dataset：薄包装 torchaudio LibriSpeech
# -----------------------

class LibriSpeechRaw(Dataset):
    """
    直接透传 torchaudio.datasets.LIBRISPEECH，保持 (waveform, sr, transcript, spk, chap, utt)
    - BaseDataModule 的 transform 包装（如需要）将包在本 Dataset 外层。
    """
    def __init__(self, root: str, url: str, download: bool = False, backend: str | None = None):
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download, backend=backend)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return self.ds[idx]  # waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id


# -----------------------
# Collator：对齐/标注/过滤
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
            dur_sec = float(waveform.shape[-1]) / float(sr)
            if (self.min_duration_sec and dur_sec < self.min_duration_sec) or \
               (self.max_duration_sec and dur_sec > self.max_duration_sec):
                continue
            if sr != self.sampling_rate:
                # 每个 worker 进程内会复用 transforms.Resample 实例（由 torchaudio 内部缓存）
                waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            wave_list.append(waveform.squeeze(0).numpy())
            txt_list.append(clean_text(transcript, self.chars_re))

        if len(wave_list) == 0:
            # 退化保护：返回一个空样本，避免上游崩溃
            return {
                'input_values': torch.zeros((1, 1), dtype=torch.float32),
                'attention_mask': torch.ones((1, 1), dtype=torch.long),
                'labels': torch.full((1, 1), -100, dtype=torch.long),
                'texts': [],
            }

        p2m = self.pad_to_multiple_of if self.pad_to_multiple_of and self.pad_to_multiple_of > 1 else None

        feats = self.processor(
            wave_list,
            sampling_rate=self.sampling_rate,
            return_tensors='pt',
            padding=True,
            pad_to_multiple_of=p2m,
        )

        txt_list_bar = [t.replace(' ', '|') for t in txt_list]
        lab_batch = self.processor.tokenizer(text=txt_list_bar, return_tensors='pt', padding=True, add_special_tokens=False)
        labels = lab_batch['input_ids']
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        if self.filter_long_labels:
            inp_len_samples = feats['attention_mask'].sum(-1)
            down_len = torch.div(inp_len_samples, self.total_stride, rounding_mode='floor')
            lab_len = (labels != -100).sum(-1)
            valid = (lab_len <= down_len)
            if not valid.all():
                if valid.sum() == 0:
                    valid_idx = torch.tensor([int(torch.argmax(inp_len_samples).item())])
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
            'texts': txt_list_bar,  # 便于验证阶段打印/对齐参考
        }


# -----------------------
# DataModule
# -----------------------

class LibriSpeechCTCDataModule(BaseDataModule):
    """
    继承 BaseDataModule 以复用 transform 包装，但覆盖 *dataloader* 以注入 collate_fn。
    你可以在 YAML 中用 instantiate_from_config 传入 train/val/test 的 Dataset 配置（见顶部示例）。
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train: dict,
        val: dict,
        test: dict,
        audio_transforms: dict | None = None,

        # ASR 相关
        sampling_rate: int = 16000,
        vocab_path: str | None = None,
        auto_vocab: bool = False,
        auto_vocab_dir: str = "./",  # auto_vocab=True 时，将 vocab.json 写入该目录
        chars_to_ignore: str = ",.?;!:\"“”‘’'()[]{}-—–…`~@#$%^&*_+=|/\\<>",

        # Collator 超参
        filter_long_labels: bool = False,
        total_stride: int = 320,
        pad_to_multiple_of: int = 8,
        min_duration_sec: float = 0.0,
        max_duration_sec: float = 0.0,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            train=train,
            val=val,
            test=test,
            audio_transforms=audio_transforms,
        )
        self.sampling_rate = sampling_rate
        self._vocab_path = vocab_path
        self.auto_vocab = auto_vocab
        self.auto_vocab_dir = auto_vocab_dir
        self.chars_to_ignore = chars_to_ignore

        self.filter_long_labels = filter_long_labels
        self.total_stride = total_stride
        self.pad_to_multiple_of = pad_to_multiple_of
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

        # 运行时对象
        self.processor: Wav2Vec2Processor | None = None
        self._collate_train = None
        self._collate_eval = None

    def _maybe_gather_charset(self, max_items: int | None = None) -> set:
        """
        遍历（可部分）数据集抽取字符集，仅用于 auto_vocab=True。
        注意：这一步可能略慢，建议只在首次构建时运行，然后固化 vocab_path。
        """
        assert hasattr(self, "train_dataset") and hasattr(self, "val_dataset")
        chars_re = make_chars_regex(self.chars_to_ignore)
        charset: set = set()

        def add_from(ds: Dataset, limit: Optional[int]):
            n = len(ds)
            k = n if limit is None else min(n, limit)
            for i in range(k):
                # 兼容 transform 包装：AudioTransformDataset 可能改变返回项顺序，但通常 (wave, sr, txt, ..)
                item = ds[i]
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    txt = clean_text(item[2], chars_re)
                    for ch in txt:
                        charset.add(ch)

        add_from(self.train_dataset, max_items)
        add_from(self.val_dataset, max_items)
        return charset

    def setup(self, stage: str | None = None):
        # 1) 由 BaseDataModule 构建 train/val/test 原始数据集，并在 _wrap 中按 stage 套上 transforms
        super().setup(stage)

        # 2) vocab / processor
        if self.auto_vocab:
            charset = self._maybe_gather_charset(max_items=None)
            self._vocab_path = build_vocab(self.auto_vocab_dir, charset)
        if self._vocab_path is None:
            # 回退到默认（A-Z、'、|）
            self._vocab_path = build_vocab(self.auto_vocab_dir, charset=None)
        self.processor = create_processor(self._vocab_path, self.sampling_rate)

        # 3) 构造 collator（train 与 eval 对 U>T / 时长过滤策略不同）
        chars_re = make_chars_regex(self.chars_to_ignore)
        self._collate_train = DataCollatorCTC(
            processor=self.processor,
            sampling_rate=self.sampling_rate,
            chars_re=chars_re,
            filter_long_labels=self.filter_long_labels,
            total_stride=self.total_stride,
            pad_to_multiple_of=self.pad_to_multiple_of,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
        )
        self._collate_eval = DataCollatorCTC(
            processor=self.processor,
            sampling_rate=self.sampling_rate,
            chars_re=chars_re,
            filter_long_labels=False,  # 验证/测试固定关闭 U>T 过滤，保证集合一致
            total_stride=self.total_stride,
            pad_to_multiple_of=self.pad_to_multiple_of,
            min_duration_sec=0.0,
            max_duration_sec=0.0,
        )

    # 覆盖 BaseDataModule 的 3 个 dataloader 以注入 collate_fn
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self._collate_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self._collate_eval,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self._collate_eval,
        )

    # 便于模型侧取用（保持解码一致）
    @property
    def processor_path(self) -> str:
        return self._vocab_path

    @property
    def tokenizer_vocab_size(self) -> int:
        return len(self.processor.tokenizer) if self.processor is not None else 0
