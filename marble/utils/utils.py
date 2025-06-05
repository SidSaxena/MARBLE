# marble/utils/io_utils.py

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal

import torch
import torchaudio
import numpy as np
from mir_eval.chord import encode, QUALITIES
from scipy.ndimage import maximum_filter1d


def list_audio_files(
    root: Union[str, Path],
    extensions: Tuple[str, ...] = ('.wav', '.flac', '.mp3', '.webm', '.mp4'),
    recursive: bool = True,
) -> List[Path]:
    """
    List all audio files in a directory, optionally searching recursively.
    Parameters
    ----------
    root : str or Path
        The root directory to search for audio files.
    extensions : tuple of str
        A tuple of file extensions to consider as audio files. Default includes common audio formats.
    recursive : bool
        If True, search recursively in subdirectories. If False, only search the root directory.
    Returns
    -------
    List[Path]
        A sorted list of Paths to audio files found in the directory.
    """
    root = Path(root)
    if recursive:
        files = [p for p in root.rglob('*') if p.suffix.lower() in extensions]
    else:
        files = [p for p in root.iterdir() if p.suffix.lower() in extensions]
    return sorted(files)


def widen_temporal_events(events, num_neighbors):
    """
    Widen binary “event” indicators along the time axis by spreading each detected event
    to its neighbors. This is often used in signal-processing or time-series contexts
    where single-frame detections are too sparse, and we want to broaden each event
    to include adjacent frames (e.g., to capture uncertainty or context around a detection).

    Parameters
    ----------
    events : 1D numpy array or list-like of integers/floats
        A one-dimensional array of length T (number of time frames) where each entry
        represents whether an “event” occurs at that frame. Expected values:
        - Exactly 1 (or >0) indicates a positive detection at that time index.
        - Exactly 0 indicates no detection.
        (If events is not already a numpy array, it will be treated as such.)
        
    num_neighbors : int
        The number of “layers” of neighboring frames on each side that should be
        affected by each original event. Each iteration spreads the event by one
        frame in both directions (using a sliding window of size 3). After each spread,
        newly spanned frames are down-weighted by 0.5 unless they were original events.

    Returns
    -------
    widen_events : 1D numpy array of floats
        A one-dimensional array of the same length as `events`.  
        - Original event frames (where events == 1) remain valued 1.  
        - Neighboring frames (within num_neighbors steps of an original event) are set
          to progressively lower values (0.5, 0.25, …) depending on how many times
          they were “inherited” from the original event.  
        - All other frames remain 0.

    How it Works (Principle)
    ------------------------
    1. We begin with a binary (0/1) array called `events`.
    2. We make a working copy called `widen_events`. In the first iteration (i = 0),
       we apply a 1D max‐filter with a window size of 3. The max‐filter essentially
       looks at each index t and replaces the value at t with the maximum value among
       [t-1, t, t+1]. This “spreads” any 1-valued event to its immediate neighbors.
    3. After the filter, any index that was originally 0 in `events` but became > 0 in
       `widen_events` (i.e., those indices are neighbors of original events) is down‐
       weighted by multiplying by 0.5. This ensures that the newly added neighbor
       frames have value = 0.5 (because the maximum filter produces 1 at those locations).
    4. On the next iteration (i = 1), we again run the 3-length max‐filter on the current
       `widen_events`. Now some neighbors might already hold 0.5 from the previous pass.
       Wherever the max‐filter produces a positive output, we again identify indices
       that were not original events (`events != 1`) but now exceed 0 in `widen_events`,
       and multiply those by another factor of 0.5. Thus, a neighbor that was 0.5 can
       become 0.25 (0.5 * 0.5) if it is one step further away from an original event.
    5. Repeat for `num_neighbors` iterations total. Each iteration effectively pushes the
       influence of an original event outward by one more frame, but with an extra
       0.5 attenuation each time.

    In effect, after `k` iterations:
      - Frames exactly k steps away from an original event will have value 1 * (0.5)**k.
      - Frames closer to the event may have larger values (e.g., 0.5 ** 1 = 0.5, 0.5 ** 2 = 0.25, …).
      - Original frames remain at value 1, because they never satisfy (events != 1).

    When to Use
    -----------
    - You have a sparse binary detection sequence (e.g., voice-activity detection,
      event markers, onsets in audio) and want to “soften” or broaden each spike
      in time.
    - You need a simple way to create a time-margin or temporal context around
      each detected event so that downstream algorithms (e.g., smoothing, feature
      pooling, or non‐maximum suppression) can see a small window around each event.
    - You want neighbor frames to have diminishing influence based on distance.

    Example Inputs & Outputs
    ------------------------
    Suppose `events = [0, 1, 0, 0, 1, 0, 0, 0]` and `num_neighbors = 2`.
    - Original events at indices 1 and 4 (0-based).
    - After 1st iteration:
        widen_events → [0, 1, 1*0.5, 0, 1, 1*0.5, 0, 0]
                         ^ 1 stays 1  ^ neighbor of 1 becomes 0.5
                                     ^ neighbor of 4 becomes 0.5
      That yields: [0, 1, 0.5, 0, 1, 0.5, 0, 0]
    - After 2nd iteration:
        The max filter on [0,1,0.5,0,1,0.5,0,0] produces
            [1, 1, 1, 0.5, 1, 1, 0.5, 0]
        But then we find all indices where `events != 1` and `widen_events > 0`, which
        would be indices 0, 2, 3, 5, 6. However, some of these are simply propagation
        from a 0.5 neighbor. We multiply all those “newly influenced” positions by 0.5:
            index 0 was 0 → became 1 from neighbor at index 1 → now 1 * 0.5 = 0.5
            index 2 was 0.5 (neighbor of index 1) → now 1 * 0.5 = 0.5  → then ×0.5 = 0.25
            index 3 was 0 → became 0.5 from neighbor at index 2 → 0.5 * 0.5 = 0.25
            index 5 was 0.5 → 1 * 0.5 = 0.5 → then ×0.5 = 0.25
            index 6 was 0 → became 0.5 from neighbor at index 5 → 0.5 * 0.5 = 0.25
        Final 2-step result: [0.5, 1, 0.25, 0.25, 1, 0.25, 0.25, 0]

    So the returned array would be:
        [0.5, 1.0, 0.25, 0.25, 1.0, 0.25, 0.25, 0.0]

    """
    # Make a working copy (float) so we don’t overwrite the original events array.
    # If `events` was integers, conversion to float ensures we can store 0.5, 0.25, etc.
    widen_events = np.array(events, dtype=float)

    # Repeat the spreading procedure num_neighbors times.
    for i in range(num_neighbors):
        # Apply a 1D maximum filter with window size = 3. For each index t, this looks at
        # widen_events[t-1], widen_events[t], widen_events[t+1], takes the max,
        # and writes that back to widen_events[t]. Boundaries are handled by default
        # (i.e., edge values replicate as needed).
        widen_events = maximum_filter1d(widen_events, size=3)

        # Find indices that were not original events (events != 1) but became >0 after filtering.
        # Those positions are newly “affected” by at least one neighbor that originally had value 1
        # or a previous iteration’s value. We down-weight them by 0.5.
        neighbor_indices = np.flatnonzero((np.array(events) != 1) & (widen_events > 0))
        widen_events[neighbor_indices] *= 0.5

    return widen_events


def times_to_mask(times: np.ndarray, T: int, fps: int) -> np.ndarray:
    """
    Convert an array of time instants (in seconds) into a binary mask of length T (frames).
    Any time point outside the valid frame range [0, T) is discarded.

    Args:
        times (np.ndarray): 1D array of shape (N,), time instants in seconds.
        T (int): Total number of frames (mask length).
        fps (int): Frames per second.

    Returns:
        np.ndarray: 1D array of shape (T,), dtype float32, containing 0.0 or 1.0.
                    A value of 1.0 indicates that a time instant maps to that frame index.
    """
    # Convert time instants (sec) to frame indices and round to nearest integer
    # times (shape: (N,)) → frames (shape: (N,))
    frames = np.round(times * fps).astype(int)

    # Initialize mask of length T with zeros
    # mask (shape: (T,))
    mask = np.zeros(T, dtype=np.float32)

    # Determine which frame indices are within [0, T)
    valid = (frames >= 0) & (frames < T)  # valid is a boolean array of shape (N,)

    # Set mask positions corresponding to valid frames to 1.0
    mask[frames[valid]] = 1.0

    return mask


def mask_to_times(mask: np.ndarray, fps: int) -> np.ndarray:
    """
    Convert a binary (or probability) mask back to time instants (in seconds).
    All non-zero values in `mask` are treated as events, their indices are divided by fps
    to get time in seconds. Results are sorted and duplicates removed.

    Args:
        mask (np.ndarray): 1D array of shape (T,), non-zero entries indicate events.
        fps (int): Frames per second.

    Returns:
        np.ndarray: 1D array of shape (M,), where M ≤ T, containing unique, sorted
                    time instants (in seconds) corresponding to mask positions > 0.
                    If no events, returns an empty array of shape (0,).
    """
    # Find indices where mask > 0
    # idxs is a 1D array of shape (M,), containing integer indices
    idxs = np.where(mask > 0)[0]

    # If no indices found, return an empty array
    if idxs.size == 0:
        return np.zeros((0,), dtype=np.float32)

    # Convert frame indices back to time (seconds)
    # idxs (shape: (M,)) → times (shape: (M,))
    times = idxs.astype(np.float32) / float(fps)

    # Sort times and remove duplicates
    # unique_times (shape: (M',)), where M' ≤ M
    unique_times = np.unique(np.sort(times))

    return unique_times


def chord_to_majmin(chord_str):
    """
    Convert a chord label to a simplified 25 class major/minor index, treating any chord
    that contains a major triad as “major” and any chord that contains a minor
    triad as “minor.” This includes seventh chords (e.g., C:7, C:maj7, C:min7),
    since they still contain a clear major or minor triad. Only chords that lack
    a definite major or minor third—such as sus, aug, or dim—will not be classified.

    Parameters
    ----------
    chord_str : str
        Chord label in the format accepted by mir_eval.chord.encode (e.g., 'C:maj7', 'D:min7', 'G:sus4').

    Returns
    -------
    int
        - 0–11: indices for major chords (C major = 0, C# major = 1, … B major = 11)
        - 12–23: indices for minor chords (C minor = 12, C# minor = 13, … B minor = 23)
        - 24: no chord (label 'N' or any chord with no valid root)
        - -1: any chord quality that doesn’t contain a pure major or pure minor triad
             (e.g., suspended (sus), augmented (aug), diminished (dim), or other ambiguous qualities)
             DO NOT compute loss for -1 in 25 classes classification.
    """
    # Encode the chord string into numerical form:
    #   root_number: integer 0–11 for C–B; <0 if no chord (e.g., 'N')
    #   semitone_bitmap: length-12 binary vector indicating which scale degrees are present
    #   bass_number: integer 0–11 for the bass note (not used for major/minor detection)
    root_number, semitone_bitmap, bass_number = encode(chord_str)

    # If there is no valid root (e.g., 'N'), classify as “no chord”
    if root_number < 0:
        return 24

    # Retrieve the reference bitmaps for a major triad and a minor triad
    major_quality = QUALITIES['maj']   # [1,0,0,0,1,0,0,1,0,0,0,0]
    minor_quality = QUALITIES['min']   # [1,0,0,1,0,0,0,1,0,0,0,0]

    # Check if the chord contains at least the three notes of a major triad:
    #   For each “1” in major_quality, semitone_bitmap must also have “1” at that position.
    contains_major = np.all(
        np.logical_and(semitone_bitmap, major_quality) == major_quality
    )

    # Check if the chord contains at least the three notes of a minor triad:
    contains_minor = np.all(
        np.logical_and(semitone_bitmap, minor_quality) == minor_quality
    )

    # If it contains a major triad (and not a minor triad), classify as major (0–11)
    if contains_major and not contains_minor:
        return root_number

    # If it contains a minor triad (and not a major triad), classify as minor (12–23)
    elif contains_minor and not contains_major:
        return root_number + 12

    # Otherwise—e.g., suspended (no clear third), augmented, diminished, or ambiguous—return -1
    return -1

def id2chord_str(chord_id):
    """
    Convert an integer representing a chord index (-1–23) to the corresponding chord string for triads.

    Parameters
    ----------
    chord_id : int
        Chord index, where 0–11 represent major chords (C major = 0, C# major = 1, … B major = 11),
        -1 represents any chord that does not contain a pure major or minor triad (e.g., suspended, augmented, diminished),
        12–23 represent minor chords (C minor = 12, C# minor = 13, … B minor = 23),
        and 24 represents "no chord".

    Returns
    -------
    str
        Chord string in the format 'C:maj', 'D:min', etc. or 'N' for no chord. or 'X' for invalid input.
    """
    # Define the chord roots and qualities
    chord_roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_quality = ':maj'
    minor_quality = ':min'
    
    if chord_id < 0:
        return 'X'
    
    if chord_id == 24:
        return 'N'  # No chord

    # Determine whether the chord is major or minor based on the index
    if chord_id < 12:  # Major chords
        root = chord_roots[chord_id]
        return f'{root}{major_quality}'

    elif chord_id < 24:  # Minor chords
        root = chord_roots[chord_id - 12]
        return f'{root}{minor_quality}'
    
    # If the index is not within the valid range, return None or raise an exception
    raise ValueError(f"Invalid chord index: {chord_id}. Must be in range 0-24.")
    