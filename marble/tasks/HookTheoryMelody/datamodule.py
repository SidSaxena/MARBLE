# File: marble/tasks/HookTheoryMelody/datamodule.py

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule
from marble.utils.emb_cache import make_clip_id


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
        audio_ext: str = ".mp3",
        precompute_labels: bool = False,
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
        # Filename extension for audio lookup. Default ".mp3" matches the raw
        # m-a-p/HookTheory dump; set to ".wav" after running
        # scripts/data/convert_audio_format.py (--to wav) to skip MP3 decode entirely.
        self.audio_ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
        self.precompute_labels = precompute_labels

        if self.channels == 1 and self.channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {self.channel_mode}")

        # Frame-grid sanity: label_freq × clip_seconds must be integer, otherwise
        # per-frame label/embedding alignment drifts silently over time
        label_len_float = self.label_freq * self.clip_seconds
        if abs(label_len_float - round(label_len_float)) > 1e-6:
            raise ValueError(
                f"label_freq ({self.label_freq}) × clip_seconds ({self.clip_seconds}) "
                f"= {label_len_float} must be integer; otherwise per-frame label "
                f"indices drift relative to encoder-token indices. Pick clip_seconds "
                f"that divides 1/label_freq cleanly (e.g. 15.0 @ 25 Hz = 375 frames)."
            )

        # Embedding-cache audio-I/O bypass. Stays None unless the host task
        # (with cache_embeddings=true, cache_pool_time=false) injects
        # `cache_check_fn = cache.has` during setup. On a hit we return a
        # dummy waveform — the encoder pass is short-circuited by the
        # mixin and the cached (L, T, H) is used instead.
        self.cache_check_fn = None

        # Load metadata. Some upstream records have ``alignment: null`` or
        # ``alignment.refined: null`` (m-a-p's pipeline failed to align them
        # to YouTube audio for whatever reason). The build script ideally
        # filters these out, but be defensive here too in case a stale
        # JSONL is passed in.
        # Cross-OS JSONL load (Windows backslash audio_paths → POSIX).
        # See marble/utils/path_compat.py.
        from marble.utils.path_compat import load_jsonl

        raw_meta: list[dict] = load_jsonl(jsonl)

        self.alignments = []
        self.melodies = []
        self.yt_ids = []
        self.meta: list[dict] = []
        # Pre-vectorised note arrays per file (parallel to self.melodies). On
        # the hot __getitem__ path we look up (onsets_sec, offsets_sec,
        # midi_pitches) instead of looping the JSON-shaped melody list and
        # calling beat_to_time_fn() per note. interp1d() is also cached per
        # file rather than rebuilt every call. Combined: ~10× speedup on
        # label compute, frees DataLoader workers to spend time on audio
        # decode instead.
        self._beat_to_time_fns: list = []
        self._note_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        skipped_no_alignment = 0
        for info in raw_meta:
            align = info.get("alignment") or {}
            refined = align.get("refined") or {}
            beats_raw = refined.get("beats")
            times_raw = refined.get("times")
            if not beats_raw or not times_raw:
                skipped_no_alignment += 1
                continue
            ytid = info.get("youtube", {}).get("id")
            if not ytid:
                continue
            self.meta.append(info)
            self.yt_ids.append(ytid)
            beats_arr = np.array(beats_raw, dtype=np.float32)
            times_arr = np.array(times_raw, dtype=np.float32)
            self.alignments.append((beats_arr, times_arr))
            beat_to_time_fn = interp1d(
                beats_arr, times_arr, kind="linear", fill_value="extrapolate"
            )
            self._beat_to_time_fns.append(beat_to_time_fn)
            melody = info["annotations"]["melody"]
            self.melodies.append(melody)
            if melody:
                onset_beats = np.array([float(n["onset"]) for n in melody], dtype=np.float32)
                offset_beats = np.array([float(n["offset"]) for n in melody], dtype=np.float32)
                # Compute onset/offset times once for the whole file. Each
                # note's index into the per-slice label grid is a cheap
                # subtract+floor — no interp1d() call needed in __getitem__.
                onsets_sec = beat_to_time_fn(onset_beats).astype(np.float32)
                offsets_sec = beat_to_time_fn(offset_beats).astype(np.float32)
                midi_pitches = np.array(
                    [
                        max(
                            0,
                            min(
                                127,
                                int(n["pitch_class"])
                                + (self.MELODY_OCTAVE + int(n["octave"])) * 12,
                            ),
                        )
                        for n in melody
                    ],
                    dtype=np.int64,
                )
            else:
                onsets_sec = np.zeros(0, dtype=np.float32)
                offsets_sec = np.zeros(0, dtype=np.float32)
                midi_pitches = np.zeros(0, dtype=np.int64)
            self._note_arrays.append((onsets_sec, offsets_sec, midi_pitches))
        if skipped_no_alignment:
            print(
                f"  HookTheoryMelody: {skipped_no_alignment:,} records skipped "
                f"(no refined alignment in metadata); kept {len(self.meta):,}",
                file=__import__("sys").stderr,
            )

        # Pre-create resamplers and build index_map.
        #
        # FAST PATH: if the JSONL record carries `num_samples` and
        # `sample_rate` (populated by
        # scripts/data/cache_audio_info_in_jsonl.py), use those directly.
        # SLOW PATH: fall back to torchaudio.info() per file. This is
        # tolerable on local SSD (~10ms/file) but catastrophic on Modal
        # volumes (~50-100ms/file × ~10k files = 10-15 min before training
        # even starts). The cache script eliminates the slow path entirely.
        self.resamplers = {}
        self.index_map: list[tuple[int, int, int, int]] = []
        for file_idx, ytid in enumerate(self.yt_ids):
            meta = self.meta[file_idx]
            cached_sr = meta.get("sample_rate")
            cached_ns = meta.get("num_samples")
            if isinstance(cached_sr, int) and isinstance(cached_ns, int):
                orig_sr = cached_sr
                num_frames = cached_ns
            else:
                audio_path = os.path.join(self.audio_dir, f"{ytid}{self.audio_ext}")
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

        # Optional: precompute every (file_idx, slice_idx) → label row up
        # front so __getitem__ does zero Python work on labels.
        #
        # Storage layout matters here. An earlier version of this code used
        # ``list[torch.Tensor]`` — 300 k separate Python objects. DataLoader
        # workers fork at training start, and every list lookup +
        # refcount-touch on a child Tensor broke the COW share, ballooning
        # per-worker RSS until the page cache evicted the audio corpus we'd
        # just warmed. Throughput collapsed ~3× vs no precompute.
        #
        # Single contiguous tensor avoids that: index returns a row view
        # without touching individual element refcounts, and the buffer
        # stays COW-shared across all forked workers. Memory cost is
        # len(index_map) × label_len × 8 B (~900 MB for HookTheory train),
        # held once and shared.
        self.label_cache: torch.Tensor | None = None
        if self.precompute_labels:
            label_len = int(self.label_freq * self.clip_seconds)
            self.label_cache = torch.empty((len(self.index_map), label_len), dtype=torch.int64)
            for i, (file_idx, slice_idx, orig_sr, orig_clip_frames) in enumerate(self.index_map):
                self.label_cache[i] = self._compute_labels(
                    file_idx, slice_idx, orig_sr, orig_clip_frames
                )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns a 4-tuple:
            waveform: Tensor (channels, clip_len_target)
            melody_labels: Tensor(shape=(label_len,), dtype=torch.long)
            audio_path: str
            clip_id: str  # for embedding-cache lookup; safe to ignore if
                          # cache_embeddings is False
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        ytid = self.yt_ids[file_idx]
        audio_path = os.path.join(self.audio_dir, f"{ytid}{self.audio_ext}")
        clip_id = make_clip_id(audio_path, slice_idx)

        # Cache hit: skip audio decode AND label compute. The probe's
        # forward path will load the (L, T, H) embedding from disk via
        # the cache mixin. The waveform is unused on hits but must have
        # the right shape so the DataLoader's default collate doesn't
        # complain. Labels still need to be real — they go through
        # the loss + metrics, not through the encoder.
        melody_labels = self._labels_for(idx, file_idx, slice_idx, orig_sr, orig_clip_frames)

        cache_check = getattr(self, "cache_check_fn", None)
        if cache_check is not None and cache_check(clip_id):
            waveform = torch.zeros(self.channels, self.clip_len_target)
            return waveform, melody_labels, audio_path, clip_id

        # 1. Load waveform
        waveform = self._load_and_preprocess(
            path=audio_path, slice_idx=slice_idx, orig_sr=orig_sr, orig_clip_frames=orig_clip_frames
        )
        return waveform, melody_labels, audio_path, clip_id

    def _labels_for(
        self, idx: int, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int
    ) -> torch.Tensor:
        """Return labels for index ``idx``, hitting the precomputed cache when present."""
        if self.label_cache is not None:
            return self.label_cache[idx]
        return self._compute_labels(file_idx, slice_idx, orig_sr, orig_clip_frames)

    def _compute_labels(
        self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int
    ) -> torch.Tensor:
        """Build the frame-level pitch label vector for one slice.

        Uses cached (onset_sec, offset_sec, midi_pitch) arrays built once at
        __init__ time — no per-call interp1d() or per-note float() conversion.
        """
        onsets_sec, offsets_sec, midi_pitches = self._note_arrays[file_idx]
        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        label_len = int(self.label_freq * self.clip_seconds)
        labels = -1 * np.ones(label_len, dtype=np.int64)
        if midi_pitches.size == 0:
            return torch.from_numpy(labels)
        # Vectorised index math — np ops over the full note array beat a
        # per-note Python loop by ~5× even for typical 30-50 note files.
        rel_starts = (onsets_sec - clip_start_time) * self.label_freq
        rel_ends = (offsets_sec - clip_start_time) * self.label_freq
        start_idxs = np.floor(rel_starts).astype(np.int64)
        end_idxs = np.ceil(rel_ends).astype(np.int64)
        np.clip(start_idxs, 0, label_len, out=start_idxs)
        np.clip(end_idxs, 0, label_len, out=end_idxs)
        # Only the small subset of notes that actually overlap this slice.
        # The bulk is filtered out by a single boolean mask, avoiding the
        # branchy "if start >= label_len or end <= 0: continue" Python check.
        active = start_idxs < end_idxs
        if not active.any():
            return torch.from_numpy(labels)
        for s, e, p in zip(start_idxs[active], end_idxs[active], midi_pitches[active], strict=True):
            labels[s:e] = p
        return torch.from_numpy(labels)

    def _load_and_preprocess(
        self, path: str, slice_idx: int, orig_sr: int, orig_clip_frames: int
    ) -> torch.Tensor:
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(path, frame_offset=offset, num_frames=orig_clip_frames)
        # Defensive: corrupt / truncated mp3 can yield a zero-length tensor
        # which then explodes downstream in resample / pad with cryptic
        # reshape errors. Synthesize silence at the expected pre-resample
        # length and let the rest of the pipeline handle it normally.
        if waveform.size(1) == 0:
            waveform = torch.zeros(self.channels, orig_clip_frames, dtype=waveform.dtype)
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
                        waveform = waveform[idx : idx + 1, :]
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
            waveform = F.pad(waveform, (0, pad_amt), mode="constant", value=0.0)
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
