# marble/tasks/LibriSpeechASR/datamodule.py
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import lightning.pytorch as pl

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from marble.core.base_datamodule import BaseDataModule


# -----------------------
# 文本 & vocab 工具
# -----------------------

def make_chars_regex(chars: str) -> Optional[re.Pattern]:
    if not chars:
        return None
    esc = ''.join(re.escape(c) for c in chars)
    return re.compile(f"[{esc}]")

def clean_text(s: str, chars_re: Optional[re.Pattern]) -> str:
    if chars_re is not None:
        s = chars_re.sub(' ', s)
    return ' '.join(s.strip().split()).upper()

def build_vocab(save_dir: str, charset: Optional[set] = None) -> str:
    """若给定 charset（清洗后字符集，含 '|' 或空格），据此生成；否则用 A-Z + ' + '|'。"""
    os.makedirs(save_dir, exist_ok=True)
    vocab_path = os.path.join(save_dir, 'vocab.json')
    if charset is None:
        tokens = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ["'", "|"]
    else:
        processed = set('|' if ch == ' ' else ch for ch in charset)
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
# Dataset：前移大部分样本级处理
# -----------------------

class LibriSpeechASRDataset(Dataset):
    """
    样本级预处理：
      - 加载 waveform / sr / transcript（torchaudio.datasets.LIBRISPEECH）
      - 重采样到 target_sr
      - 文本清洗 + 空格→'|'
      - [可选] 直接样本级 tokenize（提供 vocab_path 时）
    返回三元组：
      (wave_1d: FloatTensor[T], label_or_txt, text_bar: str)
      - 若 vocab_path 非空：label_or_txt = LongTensor[L]（未 pad）
      - 若 vocab_path 为空：label_or_txt = text_bar（同第三项）
    """
    def __init__(
        self,
        root: str,
        url: str,
        target_sr: int,
        vocab_path: Optional[str] = None,
        chars_to_ignore: str = ",.?;!:\"“”‘’'()[]{}-—–…`~@#$%^&*_+=|/\\<>",
        download: bool = False,
    ):
        self.ds = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)
        self.target_sr = int(target_sr)
        self.chars_re = make_chars_regex(chars_to_ignore)

        # 懒缓存各原始 SR 的 resampler
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        # 可选 tokenizer（若不给 vocab_path，则只返回文本，由 Collator 统一 tokenize）
        self.tokenizer: Optional[Wav2Vec2CTCTokenizer] = None
        self.pad_id: Optional[int] = None
        if vocab_path:
            self.tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token='[UNK]',
                pad_token='[PAD]',
                word_delimiter_token='|',
                do_lower_case=False,
            )
            self.pad_id = self.tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.ds)

    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sr:
            return wav
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
        return self._resamplers[sr](wav)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any, str]:
        waveform, sr, transcript, *_ = self.ds[idx]  # (1, T), sr, str
        waveform = self._resample(waveform, sr)
        wave_1d = waveform.squeeze(0).contiguous()   # (T,)

        # 文本清洗 + 空格→'|'
        txt_bar = clean_text(transcript, self.chars_re).replace(' ', '|')

        if self.tokenizer is None:
            # 只返回文本，由 Collator 统一 tokenize
            return wave_1d, txt_bar, txt_bar

        # 样本级 tokenizer（未 pad）
        ids = self.tokenizer(text=txt_bar, add_special_tokens=False)["input_ids"]
        label_ids = torch.tensor(ids, dtype=torch.long)
        return wave_1d, label_ids, txt_bar


# -----------------------
# Collator：精简仅做打包/过滤
# -----------------------

@dataclass
class LibriSpeechASRCollator:
    """
    - 对音频做动态 padding/归一化（Wav2Vec2FeatureExtractor）
    - 对 labels 动态 pad，并转 pad→-100
    - 可选 U>T 过滤（估算下采样长度）
    - 可选时长过滤
    * 兼容两种样本：
      1) (wave_1d, label_ids, txt)      # 数据集已 tokenize
      2) (wave_1d, txt, txt)            # 数据集未 tokenize（由本 collator 统一 tokenize）
    """
    processor: Wav2Vec2Processor
    pad_to_multiple_of: int = 8
    total_stride: int = 320
    filter_ut: bool = True
    min_duration_sec: float = 0.0
    max_duration_sec: float = 0.0

    def __call__(self, batch: List[Tuple[Any, Any, str]]):
        sr = self.processor.feature_extractor.sampling_rate

        waves_np, label_ids_list, texts = [], [], []
        need_tokenize = False

        for a, b, txt in batch:
            # 时长过滤（已是 target_sr）
            dur = a.numel() / float(sr)
            if (self.min_duration_sec and dur < self.min_duration_sec) or \
               (self.max_duration_sec and dur > self.max_duration_sec):
                continue

            waves_np.append(a.numpy())
            texts.append(txt)

            if isinstance(b, torch.Tensor):
                # 已经是 label_ids
                label_ids_list.append(b)
            else:
                # b 是字符串（与 txt 相同），需在 collator 里 tokenize
                need_tokenize = True

        if len(waves_np) == 0:
            return {
                "input_values": torch.zeros((1, 1), dtype=torch.float32),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
                "texts": [],
            }

        # 1）音频动态 padding
        p2m = self.pad_to_multiple_of if self.pad_to_multiple_of and self.pad_to_multiple_of > 1 else None
        feats = self.processor(
            waves_np,
            sampling_rate=sr,
            return_tensors='pt',
            padding=True,
            pad_to_multiple_of=p2m,
        )  # input_values, attention_mask

        # 2）labels：若需要则先 tokenize，再 pad；pad → -100
        pad_id = self.processor.tokenizer.pad_token_id
        if need_tokenize or len(label_ids_list) == 0:
            enc = self.processor.tokenizer(text=texts, return_tensors=None, padding=False, add_special_tokens=False)
            for ids in enc["input_ids"]:
                label_ids_list.append(torch.tensor(ids, dtype=torch.long))

        labels = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=pad_id)
        labels[labels == pad_id] = -100

        # 3）可选 U>T 过滤
        if self.filter_ut:
            inp_len_samples = feats['attention_mask'].sum(-1)  # 每条输入的样本点数
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
                texts = [texts[i] for i in valid_idx.tolist()]

        return {
            "input_values": feats["input_values"],
            "attention_mask": feats.get("attention_mask"),
            "labels": labels,
            "texts": texts,
        }


# -----------------------
# DataModule：覆盖 dataloader 以注入精简 collator
# -----------------------

class LibriSpeechASRDataModule(BaseDataModule):
    """
    继承 BaseDataModule（保留 transform 包装），覆盖 *dataloader* 注入 LibriSpeechASRCollator。
    支持 auto_vocab（建议数据集 vocab_path 设为 null，由 Collator 统一 tokenize）。
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train: dict,
        val: dict,
        test: dict,
        audio_transforms: Optional[dict] = None,

        # ASR 相关
        sampling_rate: int = 16000,
        vocab_path: Optional[str] = None,  # 若 auto_vocab=False，需提供
        auto_vocab: bool = False,
        auto_vocab_dir: str = "./",        # auto_vocab=True 时写出 vocab.json 的目录
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
        self.sampling_rate = int(sampling_rate)
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
        self.processor: Optional[Wav2Vec2Processor] = None
        self._collate_train: Optional[LibriSpeechASRCollator] = None
        self._collate_eval: Optional[LibriSpeechASRCollator] = None

    # —— charset 抽取（auto_vocab 用） ——
    def _maybe_gather_charset(self, max_items: Optional[int] = None) -> set:
        """
        遍历（可部分）train/val 数据集抽取字符集。
        - 兼容数据集是否已 tokenize：
          * 若返回 (wave, label_ids, txt) → 用 txt
          * 若返回 (wave, txt, txt)     → 用 txt
        """
        charset: set = set()
        chars_re = make_chars_regex(self.chars_to_ignore)

        def add_from(ds: Dataset, limit: Optional[int]):
            n = len(ds)
            k = n if limit is None else min(n, limit)
            for i in range(k):
                item = ds[i]
                txt = None
                if isinstance(item, (list, tuple)):
                    # 优先第三项（我们保证 Dataset 第三项是 txt）
                    if len(item) >= 3 and isinstance(item[2], str):
                        txt = item[2]
                    elif len(item) >= 2 and isinstance(item[1], str):
                        txt = item[1]
                if txt is None:
                    continue
                # 为稳妥，重新清洗一次（防止上游 Dataset 自定义）
                txt = clean_text(txt.replace('|', ' '), chars_re).replace(' ', '|')
                for ch in txt:
                    charset.add(ch)

        add_from(self.train_dataset, max_items)
        add_from(self.val_dataset, max_items)
        return charset

    def setup(self, stage: Optional[str] = None):
        """
        1) 让 BaseDataModule 实例化 & 包装 train/val/test 数据集（可带 transforms）
        2) 若 auto_vocab：遍历数据抽字符表并写出 vocab.json；否则使用传入 vocab_path 或默认
        3) 基于 vocab 构建 processor，并创建精简 collator（train / eval）
        """
        super().setup(stage)

        # 2) vocab / processor
        if self.auto_vocab:
            # 建议：此时各 Dataset 的 vocab_path 设为 null（仅返回文本）
            charset = self._maybe_gather_charset(max_items=None)
            self._vocab_path = build_vocab(self.auto_vocab_dir, charset)
        if self._vocab_path is None:
            # 回退默认（A-Z、'、|）
            self._vocab_path = build_vocab(self.auto_vocab_dir, charset=None)

        self.processor = create_processor(self._vocab_path, self.sampling_rate)

        # 3) collator
        self._collate_train = LibriSpeechASRCollator(
            processor=self.processor,
            pad_to_multiple_of=self.pad_to_multiple_of,
            total_stride=self.total_stride,
            filter_ut=self.filter_long_labels,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
        )
        self._collate_eval = LibriSpeechASRCollator(
            processor=self.processor,
            pad_to_multiple_of=self.pad_to_multiple_of,
            total_stride=self.total_stride,
            filter_ut=False,  # 验证/测试固定关闭 U>T 过滤，保证集合一致
            min_duration_sec=0.0,
            max_duration_sec=0.0,
        )

    # 覆盖 dataloader 以注入 collate_fn
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
