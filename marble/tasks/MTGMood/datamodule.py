# marble/tasks/MTGMood/datamodule.py
from typing import Optional

import torch

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTGMoodAudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'mood/theme---background': 0,
        'mood/theme---film': 1,
        'mood/theme---melancholic': 2,
        'mood/theme---melodic': 3,
        'mood/theme---children': 4,
        'mood/theme---relaxing': 5,
        'mood/theme---documentary': 6,
        'mood/theme---emotional': 7,
        'mood/theme---space': 8,
        'mood/theme---love': 9,
        'mood/theme---drama': 10,
        'mood/theme---adventure': 11,
        'mood/theme---energetic': 12,
        'mood/theme---heavy': 13,
        'mood/theme---dark': 14,
        'mood/theme---calm': 15,
        'mood/theme---action': 16,
        'mood/theme---dramatic': 17,
        'mood/theme---epic': 18,
        'mood/theme---powerful': 19,
        'mood/theme---upbeat': 20,
        'mood/theme---slow': 21,
        'mood/theme---inspiring': 22,
        'mood/theme---soft': 23,
        'mood/theme---meditative': 24,
        'mood/theme---fun': 25,
        'mood/theme---happy': 26,
        'mood/theme---positive': 27,
        'mood/theme---romantic': 28,
        'mood/theme---sad': 29,
        'mood/theme---hopeful': 30,
        'mood/theme---motivational': 31,
        'mood/theme---deep': 32,
        'mood/theme---uplifting': 33,
        'mood/theme---ballad': 34,
        'mood/theme---soundscape': 35,
        'mood/theme---dream': 36,
        'mood/theme---movie': 37,
        'mood/theme---fast': 38,
        'mood/theme---nature': 39,
        'mood/theme---cool': 40,
        'mood/theme---corporate': 41,
        'mood/theme---travel': 42,
        'mood/theme---funny': 43,
        'mood/theme---sport': 44,
        'mood/theme---commercial': 45,
        'mood/theme---advertising': 46,
        'mood/theme---holiday': 47,
        'mood/theme---christmas': 48,
        'mood/theme---sexy': 49,
        'mood/theme---game': 50,
        'mood/theme---groovy': 51,
        'mood/theme---retro': 52,
        'mood/theme---summer': 53,
        'mood/theme---party': 54,
        'mood/theme---trailer': 55
        }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTG/audio/48/948.low.flac", 
        "label": ["mood/theme---background"], 
        "duration": 212.66666666666666, 
        "sample_rate": 44100, 
        "num_samples": 9378600, 
        "bit_depth": 16, 
        "channels": 1
        }

    def __init__(self, jsonl: str, sample_rate: int, channels: int,
                 clip_seconds: float, channel_mode: str="first",
                 min_clip_ratio: float=1.0, backend: Optional[str] = None):
        super().__init__(
            jsonl=jsonl,
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            channel_mode=channel_mode,
            min_clip_ratio=min_clip_ratio,
            backend=backend
        )

    def get_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        info = self.meta[file_idx]
        label_indices = [self.LABEL2IDX[tag] for tag in info['label']]
        num_labels = len(self.LABEL2IDX)
        label = torch.zeros(num_labels, dtype=torch.int)
        label[label_indices] = 1
        return label


class MTGMoodAudioTrain(_MTGMoodAudioBase):
    pass


class MTGMoodAudioVal(_MTGMoodAudioBase):
    pass


class MTGMoodAudioTest(MTGMoodAudioVal):
    pass


class MTGMoodDataModule(BaseDataModule):
    pass
