# File: marble/tasks/HookTheoryMelody/datamodule.py

import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule


class _HookTheoryMelodyDataset(Dataset):
    """
    Dataset for HookTheory melody extraction task.
    - Splits each audio file into clips of length `clip_seconds`.
    - For each clip, returns:
        * waveform: Tensor of shape (channels, clip_len_samples)
        * melody_labels: 1D Tensor of length label_len (pitch per frame, -1 if no note)
        * audio_path: str
    """
    MELODY_OCTAVE = 5

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        label_freq: int,
        audio_dir: str,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.label_freq = label_freq
        self.audio_dir = audio_dir
        self.min_clip_ratio = min_clip_ratio

        if self.channels == 1 and self.channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        # Load metadata
        with open(jsonl, 'r') as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Prepare alignment & melody annotations
        self.alignments = []
        self.melodies = []
        self.yt_ids = []
        for info in self.meta:
            ytid = info['youtube']['id']
            self.yt_ids.append(ytid)
            beats = np.array(info['alignment']['refined']['beats'], dtype=np.float32)
            times = np.array(info['alignment']['refined']['times'], dtype=np.float32)
            self.alignments.append((beats, times))
            self.melodies.append(info['annotations']['melody'])

        # Pre-create resamplers and build index_map
        self.resamplers = {}
        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, ytid in enumerate(self.yt_ids):
            audio_path = os.path.join(self.audio_dir, f"{ytid}.mp3")
            info = torchaudio.info(audio_path)
            orig_sr = info.sample_rate
            num_frames = info.num_frames
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue
            n_full = num_frames // orig_clip_frames
            rem = num_frames - n_full * orig_clip_frames
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full
            for slice_idx in range(n_slices):
                self.index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))
            if orig_sr != self.sample_rate:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns:
            waveform: Tensor (channels, clip_len_target)
            melody_labels: Tensor(shape=(label_len,), dtype=torch.long)
            audio_path: str
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        ytid = self.yt_ids[file_idx]
        audio_path = os.path.join(self.audio_dir, f"{ytid}.mp3")

        # 1. Load waveform
        waveform = self._load_and_preprocess(
            path=audio_path,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )

        # 2. Build beat→time interpolation
        beats, times = self.alignments[file_idx]
        beat_to_time_fn = interp1d(beats, times, kind='linear', fill_value='extrapolate')
        num_beats = int(self.meta[file_idx]['annotations']['num_beats'])
        track_end_time = float(beat_to_time_fn(num_beats))

        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        clip_end_time = clip_start_time + self.clip_seconds

        # 3. Generate melody labels
        label_len = int(self.label_freq * self.clip_seconds)
        melody_labels = -1 * np.ones(label_len, dtype=np.int64)

        # For each note in melody annotation
        for note in self.melodies[file_idx]:
            onset_beat = float(note['onset'])
            offset_beat = float(note['offset'])
            onset_sec = float(beat_to_time_fn(onset_beat))
            offset_sec = float(beat_to_time_fn(offset_beat))
            midi_pitch = int(note['pitch_class'] + (self.MELODY_OCTAVE + int(note['octave'])) * 12)
            rel_onset = onset_sec - clip_start_time
            rel_offset = offset_sec - clip_start_time
            start_idx = int(np.floor(rel_onset * self.label_freq))
            end_idx = int(np.ceil(rel_offset * self.label_freq))
            start_idx = max(0, start_idx)
            end_idx = min(label_len, end_idx)
            if start_idx >= label_len or end_idx <= 0:
                continue
            melody_labels[start_idx:end_idx] = midi_pitch

        melody_labels = torch.from_numpy(melody_labels)
        return waveform, melody_labels, audio_path

    def _load_and_preprocess(
        self,
        path: str,
        slice_idx: int,
        orig_sr: int,
        orig_clip_frames: int
    ) -> torch.Tensor:
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip_frames
        )
        orig_ch = waveform.size(0)
        # Channel handling
        if orig_ch >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1, :]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    if torch.rand(()) < 0.5:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        idx = torch.randint(0, orig_ch, ())
                        waveform = waveform[idx:idx+1, :]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                waveform = waveform[: self.channels, :]
        else:
            deficit = self.channels - orig_ch
            tail = waveform[-1:, :].repeat(deficit, 1)
            waveform = torch.cat([waveform, tail], dim=0)

        # Resample
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # Pad or truncate
        cur_len = waveform.size(1)
        if cur_len < self.clip_len_target:
            pad_amt = self.clip_len_target - cur_len
            waveform = F.pad(waveform, (0, pad_amt), mode='constant', value=0.0)
        elif cur_len > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform


class HookTheoryMelodyTrain(_HookTheoryMelodyDataset):
    """Training split: shuffle in DataLoader."""
    pass


class HookTheoryMelodyVal(_HookTheoryMelodyDataset):
    """Validation split: no shuffling."""
    pass


class HookTheoryMelodyTest(HookTheoryMelodyVal):
    """Test split: same behavior as validation."""
    pass


class HookTheoryMelodyDataModule(BaseDataModule):
    """
    DataModule for HookTheory Melody task.
    Configuration example:
        datamodule:
            _target_: marble.tasks.HookTheoryMelody.datamodule.HookTheoryMelodyDataModule
            sample_rate: 22050
            channels: 1
            clip_seconds: 15.0
            jsonl: path/to/hooktheory_melody.jsonl
            label_freq: 100
            audio_dir: path/to/audio_files
            batch_size: 16
            num_workers: 4
    """
    pass
