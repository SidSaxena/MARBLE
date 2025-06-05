# marble/tasks/Chords1217/datamodule.py

import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule
from marble.utils.utils import chord_to_majmin, id2chord_str

class _Chords1217AudioBase(Dataset):
    """
    Base dataset for chord recognition tasks.
    - Splits each audio file into clips of length `clip_seconds`.
    - For each clip, returns:
        * waveform tensor of shape (channels, clip_len_target)
        * targets dict with:
            - "chord": 1D LongTensor of length label_len (frame-level chord indices, 0–24)
    Expected JSONL format per line:
    {
        "audio_path": "<path/to/audio.wav>",
        "label": [
            {
                "start_time": <float seconds>,
                "end_time": <float seconds>,
                "chord_str": "<e.g. 'C:maj' or 'N'>"
            },
            ...
        ],
        "duration": <float seconds>,
        "sample_rate": <int>,
        "num_samples": <int>,
        "channels": <int>
    }
    """
    EXAMPLE_JSONL = {
        "audio_path": "data/Chords1217/audio/TRWTQYM149E35B4C40.mp3", 
        "label": 
            [
                {"start_time": 0.0, "end_time": 0.42124700000000004, "chord_str": "N"}, 
                {"start_time": 0.42124700000000004, "end_time": 1.2562470000000001, 
                "chord_str": "N"}, 
                {"start_time": 1.2562470000000001, "end_time": 21.840737000000004, "chord_str": "E:maj"}, 
                {"start_time": 21.840737, "end_time": 23.048174, "chord_str": "F#:maj"}, 
                {"start_time": 23.048174000000003, "end_time": 24.290442000000002, "chord_str": "B:maj"}, 
                {"start_time": 24.290442000000002, "end_time": 25.474660000000004, "chord_str": "E:maj"}, 
                {"start_time": 25.47466, "end_time": 26.682097, "chord_str": "C#:maj"}, 
                {"start_time": 26.682097000000002, "end_time": 27.877925, "chord_str": "F#:maj"}, 
                {"start_time": 27.877925, "end_time": 29.073753, "chord_str": "B:maj"}, 
                {"start_time": 29.073753000000004, "end_time": 29.372436000000004, "chord_str": "E:maj"}, 
                {"start_time": 29.372436, "end_time": 31.546678, "chord_str": "D:maj/2"}, 
                {"start_time": 31.546678000000004, "end_time": 52.293707000000005, "chord_str": "E:maj"}, 
                {"start_time": 52.293707000000005, "end_time": 53.466315, "chord_str": "F#:maj"}, 
                {"start_time": 53.466315, "end_time": 54.720192000000004, "chord_str": "B:maj"}, 
                {"start_time": 54.720192000000004, "end_time": 55.939240000000005, "chord_str": "E:maj"}, 
                {"start_time": 55.939240000000005, "end_time": 57.158288000000006, "chord_str": "C#:maj"}, 
                {"start_time": 57.158288000000006, "end_time": 58.365725000000005, "chord_str": "F#:maj"}, 
                {"start_time": 58.365725000000005, "end_time": 59.584773000000006, "chord_str": "B:maj"}, 
                {"start_time": 59.584773000000006, "end_time": 59.893728, "chord_str": "E:maj"}, 
                {"start_time": 59.893728, "end_time": 62.022868, "chord_str": "D:maj/2"}, 
                {"start_time": 62.022868, "end_time": 76.26493500000001, "chord_str": "E:maj"}, 
                {"start_time": 76.264934999, "end_time": 96.71347999900001, "chord_str": "E:maj"}, 
                {"start_time": 96.71348, "end_time": 97.94413800000001, "chord_str": "F#:maj"}, 
                {"start_time": 97.94413800000001, "end_time": 99.16318600000001, "chord_str": "B:maj"}, 
                {"start_time": 99.16318600000001, "end_time": 100.393843, "chord_str": "E:maj"}, 
                {"start_time": 100.393843, "end_time": 101.578061, "chord_str": "C#:maj"}, 
                {"start_time": 101.578061, "end_time": 102.80871900000001, "chord_str": "F#:maj"}, 
                {"start_time": 102.80871900000001, "end_time": 103.969716, "chord_str": "B:maj"}, 
                {"start_time": 103.969716, "end_time": 104.27789100000001, "chord_str": "E:maj"}, 
                {"start_time": 104.27789100000001, "end_time": 106.45425100000001, "chord_str": "D:maj/2"}, 
                {"start_time": 106.45425100000001, "end_time": 114.15805800000001, "chord_str": "E:maj"}, 
                {"start_time": 114.15805800000001, "end_time": 118.987755, "chord_str": "N"}
            ], 
        "duration": 119.01387755102041, 
        "sample_rate": 44100, 
        "num_samples": 5248512, 
        "bit_depth": 0, 
        "channels": 1
        }

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        label_freq: int,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        super().__init__()
        assert isinstance(sample_rate, (int, float)), "sample_rate must be an integer or float"
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        self.clip_seconds = clip_seconds
        # Target clip length in samples after resampling
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.label_freq = label_freq

        # Validate channel_mode if we expect mono output
        if self.channels == 1 and self.channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        # Load metadata entries from JSONL file
        with open(jsonl, 'r') as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Pre-extract chord annotations per track
        # Each entry in chords_meta is a list of tuples (start_time, chord_idx)
        self.chords_meta: List[List[Tuple[float, int]]] = []
        for info in self.meta:
            chord_list = info.get('label', [])
            annotated: List[Tuple[float, int]] = []
            for segment in chord_list:
                t0 = float(segment['start_time'])
                chord_str = segment['chord_str']
                idx = chord_to_majmin(chord_str)
                annotated.append((t0, idx))
            # Ensure the chord list is sorted by start_time
            annotated = sorted(annotated, key=lambda x: x[0])
            self.chords_meta.append(annotated)

        # Pre-create resamplers for any original sample rate != target sample_rate
        self.resamplers = {}
        for info in self.meta:
            orig_sr = int(info['sample_rate'])
            if orig_sr != self.sample_rate:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

        # Build an index map: for each file, compute how many clips can be extracted
        # and store tuples of (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr = int(info['sample_rate'])
            total_samples = int(info['num_samples'])
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue  # skip invalid
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

            for slice_idx in range(n_slices):
                self.index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns:
            waveform: torch.Tensor of shape (channels, clip_len_target)
            targets: torch.LongTensor(shape=(label_len,), values in [0..24] or -1)
            audio_path: str - path to the audio file
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info = self.meta[file_idx]
        audio_path = info['audio_path']

        # 1. Load and preprocess waveform
        waveform = self._load_and_preprocess(
            path=audio_path,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )
        # waveform.shape == (self.channels, self.clip_len_target)

        # 2. Generate chord label sequence for this clip
        chord_annotations = self.chords_meta[file_idx]  # List of (start_time, chord_idx)
        # Compute clip start/end times in seconds (based on original sampling rate)
        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        clip_end_time = clip_start_time + self.clip_seconds

        # Compute length of label sequence: floor(label_freq * clip_seconds)
        label_len = int(self.label_freq * self.clip_seconds)
        chord_seq = np.zeros(label_len, dtype=np.int64)

        # If there are no annotations, fill with "no chord" class (index 24)
        if len(chord_annotations) == 0:
            chord_seq[:] = 24
        else:
            # Build a list of segment boundaries and labels based on start times
            # Append a sentinel at end with "no chord"
            segments: List[Tuple[float, int]] = []
            for (t0, cidx) in chord_annotations:
                # Only keep annotations that intersect this clip by start time
                if t0 >= clip_end_time:
                    break
                if t0 < clip_start_time:
                    continue
                segments.append((t0, cidx))

            # Handle the case where the first annotation starts after the clip's start time.
            # Instead of just prepending a "no chord", check if the previous chord should still be used.
            if len(segments) == 0 or segments[0][0] > clip_start_time:
                # Check if there's a previous chord that lasts into this clip
                # For the very first segment, use the last chord of the previous clip if its end_time hasn't passed
                prev_chord = None
                for (prev_start_time, prev_chord_label) in reversed(self.chords_meta[file_idx]):
                    if prev_start_time < clip_start_time:
                        prev_chord = prev_chord_label
                        break
                
                if prev_chord is not None:
                    # If the previous chord is still ongoing, carry it over
                    segments.insert(0, (clip_start_time, prev_chord))
                else:
                    # Otherwise, insert 'no chord' if no ongoing chord is found
                    segments.insert(0, (clip_start_time, 24))  # No chord at the beginning

            # Append end sentinel at clip_end_time
            segments.append((clip_end_time, 24))  # No chord at the end

            # Now assign chord label per frame
            seg_ptr = 0
            current_label = segments[0][1]
            next_change_time = segments[1][0] if len(segments) > 1 else clip_end_time
            for frame_idx in range(label_len):
                t = clip_start_time + frame_idx / self.label_freq
                # Advance segment pointer if needed
                while t >= next_change_time and seg_ptr + 1 < len(segments) - 1:
                    seg_ptr += 1
                    current_label = segments[seg_ptr][1]
                    next_change_time = segments[seg_ptr + 1][0] if seg_ptr + 1 < len(segments) else clip_end_time

                # Handle 'no chord' case: use the previous segment's chord label
                if current_label == 24 and seg_ptr > 0:
                    current_label = segments[seg_ptr - 1][1]  # Inherit previous chord if 'no chord' is encountered
                chord_seq[frame_idx] = current_label

        chord_tensor = torch.from_numpy(chord_seq)
        targets = chord_tensor
        return waveform, targets, audio_path

    def _load_and_preprocess(
        self,
        path: str,
        slice_idx: int,
        orig_sr: int,
        orig_clip_frames: int
    ) -> torch.Tensor:
        """
        Load a slice of audio from disk and apply:
        - channel selection / downmixing
        - resampling to self.sample_rate
        - padding or truncation to ensure shape (self.channels, self.clip_len_target)

        Returns:
            Tensor of shape (self.channels, self.clip_len_target)
        """
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip_frames
        )  # waveform.shape == (orig_ch, actual_frames)

        orig_ch = waveform.size(0)

        # 1. Channel handling / downmixing
        if orig_ch >= self.channels:
            # If we only need fewer channels than original
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1, :]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    # Use 50% chance to mix, 50% chance to pick a random channel
                    if torch.rand(()) < 0.5:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        idx = torch.randint(0, orig_ch, ())
                        waveform = waveform[idx:idx+1, :]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                # Keep the first self.channels channels
                waveform = waveform[: self.channels, :]
        else:
            # orig_ch < self.channels: replicate last channel to match desired channel count
            deficit = self.channels - orig_ch
            tail = waveform[-1:, :].repeat(deficit, 1)
            waveform = torch.cat([waveform, tail], dim=0)

        # 2. Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # 3. Pad or truncate to exact length
        cur_len = waveform.size(1)
        if cur_len < self.clip_len_target:
            pad_amt = self.clip_len_target - cur_len
            waveform = F.pad(waveform, (0, pad_amt), mode='constant', value=0.0)
        elif cur_len > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform


class Chords1217AudioTrain(_Chords1217AudioBase):
    """Training split: shuffle in DataLoader."""
    pass


class Chords1217AudioVal(_Chords1217AudioBase):
    """Validation split: no shuffling."""
    pass


class Chords1217AudioTest(Chords1217AudioVal):
    """Test split: same behavior as validation."""
    pass

class Chords1217DataModule(BaseDataModule):
    pass
