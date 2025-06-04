import torch
import numpy as np
import mir_eval
from marble.tasks.GTZANBeatTracking.modules import TimeEventFMeasure
from marble.utils.utils import mask_to_times

def test_time_event_fmeasure():
    """
    Test suite for TimeEventFMeasure. Uses synthetic masks to validate behavior.
    """
    label_freq = 10  # 10 frames per second
    tol = 0.07       # 70 ms tolerance

    # Helper: builds a mask of length T with events at given times (in seconds)
    def make_mask(event_times, T, fps):
        mask = np.zeros(T, dtype=np.float32)
        frames = np.round(np.array(event_times) * fps).astype(int)
        valid = (frames >= 0) & (frames < T)
        mask[frames[valid]] = 1.0
        return torch.tensor(mask).unsqueeze(0)  # shape (1, T)

    T = 20  # total frames (2 seconds at 10 fps)

    # 1. Both predicted and reference are empty → F1 = 1.0
    metric1 = TimeEventFMeasure(label_freq=label_freq, tol=tol)
    est1 = torch.zeros((1, T))
    ref1 = torch.zeros((1, T))
    metric1.update(est1, ref1)
    f1_1 = metric1.compute().item()
    print(f"Test 1 (both empty): F1 = {f1_1:.2f} (expected 1.00)")

    # 2. Single matching event at exactly the same time → F1 = 1.0
    metric2 = TimeEventFMeasure(label_freq=label_freq, tol=tol)
    # Event at t = 1.0 s → frame index 10
    est2 = make_mask([1.0], T, label_freq)
    ref2 = make_mask([1.0], T, label_freq)
    metric2.update(est2, ref2)
    f1_2 = metric2.compute().item()
    print(f"Test 2 (perfect match): F1 = {f1_2:.2f} (expected 1.00)")

    # 3. Reference has an event, prediction is empty → F1 = 0.0
    metric3 = TimeEventFMeasure(label_freq=label_freq, tol=tol)
    est3 = torch.zeros((1, T))
    ref3 = make_mask([0.5], T, label_freq)  # event at 0.5 s → frame 5
    metric3.update(est3, ref3)
    f1_3 = metric3.compute().item()
    print(f"Test 3 (ref only): F1 = {f1_3:.2f} (expected 0.00)")

    # 4. Prediction has an event, reference is empty → F1 = 0.0
    metric4 = TimeEventFMeasure(label_freq=label_freq, tol=tol)
    est4 = make_mask([0.5], T, label_freq)  # frame 5
    ref4 = torch.zeros((1, T))
    metric4.update(est4, ref4)
    f1_4 = metric4.compute().item()
    print(f"Test 4 (est only): F1 = {f1_4:.2f} (expected 0.00)")

    # 5a. Two events on each side; one match, one miss → F1 = 0.50
    metric5a = TimeEventFMeasure(label_freq=label_freq, tol=tol)
    # References at t = 1.0 s (frame 10) and t = 1.5 s (frame 15)
    ref5a = make_mask([1.0, 1.5], T, label_freq)
    # Predictions at t = 1.0 s (match) and t = 1.6 s (frame 16, 0.1 s away from 1.5)
    est5a = make_mask([1.0, 1.6], T, label_freq)
    metric5a.update(est5a, ref5a)
    f1_5a = metric5a.compute().item()
    print(f"Test 5a (2 ref vs. 2 pred, one match): F1 = {f1_5a:.2f} (expected 0.50)")

    # 5b. One reference event, two predicted events; only one match → F1 ≈ 0.67
    metric5b = TimeEventFMeasure(label_freq=label_freq, tol=tol)
    # Reference only at t = 1.0 s (frame 10)
    ref5b = make_mask([1.0], T, label_freq)
    # Predictions at t = 1.0 s (match) and t = 1.6 s (non‐match)
    est5b = make_mask([1.0, 1.6], T, label_freq)
    metric5b.update(est5b, ref5b)
    f1_5b = metric5b.compute().item()
    print(f"Test 5b (1 ref vs. 2 pred, one match): F1 = {f1_5b:.2f} (expected ~0.67)")

if __name__ == "__main__":
    test_time_event_fmeasure()
    