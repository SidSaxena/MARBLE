# marble/tasks/MTGTop50/datamodule.py
from typing import Optional

import torch

from marble.core.base_datamodule import BaseDataModule, BaseAudioDataset


class _MTGTop50AudioBase(BaseAudioDataset):
    LABEL2IDX = {
        'genre---rock': 0,
        'genre---pop': 1,
        'genre---classical': 2,
        'instrument---voice': 3,
        'genre---popfolk': 4,
        'genre---funk': 5,
        'genre---ambient': 6,
        'genre---chillout': 7,
        'genre---downtempo': 8,
        'genre---easylistening': 9,
        'genre---electronic': 10,
        'genre---lounge': 11,
        'instrument---synthesizer': 12,
        'genre---triphop': 13,
        'genre---techno': 14,
        'genre---newage': 15,
        'genre---jazz': 16,
        'genre---metal': 17,
        'instrument---piano': 18,
        'genre---alternative': 19,
        'genre---experimental': 20,
        'genre---soundtrack': 21,
        'mood/theme---film': 22,
        'genre---world': 23,
        'instrument---strings': 24,
        'genre---trance': 25,
        'genre---orchestral': 26,
        'instrument---guitar': 27,
        'genre---hiphop': 28,
        'genre---instrumentalpop': 29,
        'mood/theme---relaxing': 30,
        'genre---reggae': 31,
        'mood/theme---emotional': 32,
        'instrument---keyboard': 33,
        'instrument---violin': 34,
        'genre---dance': 35,
        'instrument---bass': 36,
        'instrument---computer': 37,
        'instrument---drummachine': 38,
        'instrument---drums': 39,
        'instrument---electricguitar': 40,
        'genre---folk': 41,
        'instrument---acousticguitar': 42,
        'genre---poprock': 43,
        'genre---indie': 44,
        'mood/theme---energetic': 45,
        'mood/theme---happy': 46,
        'instrument---electricpiano': 47,
        'genre---house': 48,
        'genre---atmospheric': 49
    }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/MTG/audio-low/41/241.low.mp3", 
        "label": ["genre---rock"], 
        "duration": 340.1066666666667, 
        "sample_rate": 44100, 
        "num_samples": 14998704, 
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


class MTGTop50AudioTrain(_MTGTop50AudioBase):
    pass


class MTGTop50AudioVal(_MTGTop50AudioBase):
    pass


class MTGTop50AudioTest(MTGTop50AudioVal):
    pass


class MTGTop50DataModule(BaseDataModule):
    pass
