# marble/tasks/Chords1217/datamodule.py

import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import lightning.pytorch as pl

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset
from marble.utils.utils import chord_to_majmin, id2chord_str


class _Chords1217AudioBase(BaseAudioDataset):
    """
    Base dataset for chord recognition tasks.
    - Splits each audio file into clips of length `clip_seconds`.
    - For each clip, returns:
        * waveform tensor of shape (channels, clip_len_target)
        * targets dict with:
            - "chord": 1D LongTensor of length label_len (frame-level chord indices, 0–24)
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

    def __init__(self, jsonl: str, sample_rate: int, channels: int,
                 clip_seconds: float, label_freq: int, channel_mode: str="first",
                 min_clip_ratio: float=1.0):
        super().__init__(
            jsonl=jsonl,
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            label_freq=label_freq,
            channel_mode=channel_mode,
            min_clip_ratio=min_clip_ratio
        )
        # Build a per-file list of (start_time, chord_idx) sorted by start_time
        self.chords_meta: List[List[Tuple[float, int]]] = []
        for info in self.meta:
            ann_list = info.get("label", [])
            annotated: List[Tuple[float, int]] = []
            for seg in ann_list:
                t0 = float(seg["start_time"])
                idx = chord_to_majmin(seg["chord_str"])
                annotated.append((t0, idx))
            annotated.sort(key=lambda x: x[0])
            self.chords_meta.append(annotated)

    def get_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        """
        Generate a frame-level chord label sequence for a single clip.

        Args:
            file_idx (int): Index of the audio file in self.meta.
            slice_idx (int): Which slice number (0-based) within that file.
            orig_sr (int): Original sample rate of the audio file.
            orig_clip_frames (int): Number of samples per clip at the original sample rate
                                    (i.e., floor(clip_seconds * orig_sr)).

        Returns:
            torch.LongTensor: A 1D tensor of length `label_len = int(self.label_freq * self.clip_seconds)`,
                              where each element is a chord index (0–23 for chords, 24 for “no chord”).
        """
        # Retrieve the sorted list of (start_time, chord_idx) pairs for this file
        chord_annotations = self.chords_meta[file_idx]

        # Compute the clip's start time in seconds:
        #   slice_idx * (orig_clip_frames / orig_sr) gives seconds offset of this slice
        clip_start = slice_idx * (orig_clip_frames / orig_sr)

        # Total number of label frames for each clip = label_freq * clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)

        # Create an array to hold the chord index for each frame; initialize to zeros
        chord_seq = np.zeros(label_len, dtype=np.int64)

        # Compute the clip's end time in seconds
        clip_end = clip_start + self.clip_seconds

        # If there are no chord annotations at all for this file, fill with “no chord” (index 24)
        if not chord_annotations:
            chord_seq[:] = 24
        else:
            # Build a list `segments` that contains only the annotations whose start_time
            # falls within [clip_start, clip_end). We’ll later insert boundary markers.
            segments: List[Tuple[float, int]] = []
            for (t0, cidx) in chord_annotations:
                if t0 >= clip_end:
                    # Once we reach an annotation that starts after this clip, stop collecting
                    break
                if t0 < clip_start:
                    # Skip any annotation that started before the clip began
                    continue
                segments.append((t0, cidx))

            # If the first collected segment starts after clip_start, we need to decide:
            #   a) Does a previous chord extend into this clip?
            #   b) If not, we must begin this clip with “no chord” (24).
            if not segments or segments[0][0] > clip_start:
                prev_chord = None
                # Scan the original annotations in reverse to find the last chord
                # that started before clip_start. If found, that chord is still “active”
                # at the start of this clip.
                for (ps, pl) in reversed(chord_annotations):
                    if ps < clip_start:
                        prev_chord = pl
                        break

                if prev_chord is not None:
                    # If we found a chord that started earlier and lasted into this clip,
                    # insert it at time=clip_start so we carry it over.
                    segments.insert(0, (clip_start, prev_chord))
                else:
                    # Otherwise, explicitly mark “no chord” at clip_start
                    segments.insert(0, (clip_start, 24))

            # Append a sentinel at the end of the clip with “no chord” (24). This
            # helps us know when to stop each chord segment.
            segments.append((clip_end, 24))

            # At this point, `segments` is a list of (time, chord_index) sorted by time,
            # covering [clip_start, clip_end]. Example:
            #   [
            #     (clip_start,    some_chord_idx or 24),
            #     (t1,            chord_idx1),
            #     (t2,            chord_idx2),
            #     ...,
            #     (clip_end,      24)
            #   ]

            # We’ll walk through each label frame (0 .. label_len-1), compute its
            # timestamp, and determine which segment applies.
            seg_ptr = 0
            current_label = segments[0][1]
            # The next boundary time when chord changes:
            next_change_time = segments[1][0] if len(segments) > 1 else clip_end

            # Iterate over every label frame
            for f_idx in range(label_len):
                # Compute the exact time (in seconds) of this frame:
                t = clip_start + f_idx / self.label_freq

                # Advance the segment pointer if we have passed the next boundary
                while t >= next_change_time and seg_ptr + 1 < len(segments) - 1:
                    seg_ptr += 1
                    current_label = segments[seg_ptr][1]
                    # Update the next boundary time (or use clip_end if at last real segment)
                    next_change_time = (
                        segments[seg_ptr + 1][0]
                        if (seg_ptr + 1) < len(segments)
                        else clip_end
                    )

                # If current_label is “no chord” (24) but there is a previous segment,
                # inherit that previous chord. This ensures that short “no chord” markers
                # don’t override an ongoing chord unless it truly ends.
                if current_label == 24 and seg_ptr > 0:
                    current_label = segments[seg_ptr - 1][1]

                # Assign the chosen label index to this frame
                chord_seq[f_idx] = current_label

        # Convert the numpy array to a PyTorch LongTensor and return
        return torch.from_numpy(chord_seq)


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
