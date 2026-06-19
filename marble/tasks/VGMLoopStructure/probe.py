# marble/tasks/VGMLoopStructure/probe.py
#
# ProbeAudioTask is the generic HookTheoryStructure probe (encoder / decoders /
# losses / metrics wired from config).  We subclass it ONLY to add an opt-in
# dump of per-file test predictions for offline confusion-matrix / per-class
# analysis.  The dump is gated on MARBLE_DUMP_TEST_PREDS being set, so default
# behaviour (and every other task re-using the parent) is unchanged.
import os

import torch

from marble.tasks.HookTheoryStructure.probe import ProbeAudioTask as _BaseProbeAudioTask

__all__ = ["ProbeAudioTask"]


class ProbeAudioTask(_BaseProbeAudioTask):
    def on_test_epoch_end(self) -> None:
        # Parent computes + logs the configured test metrics and leaves
        # self._test_file_outputs populated (it does not clear it).
        super().on_test_epoch_end()

        dump_path = os.environ.get("MARBLE_DUMP_TEST_PREDS")
        if not dump_path:
            return

        # Re-aggregate per file exactly as the parent does (mean logit / file).
        file_dict: dict = {}
        for entry in self._test_file_outputs:
            info = file_dict.setdefault(
                entry["uid"], {"logits": [], "label": entry["label"]}
            )
            info["logits"].append(entry["logit"])

        uids, preds, labels = [], [], []
        for uid, info in file_dict.items():
            mean_logit = torch.stack(info["logits"]).mean(dim=0)
            uids.append(str(uid))
            preds.append(int(mean_logit.argmax().item()))
            labels.append(int(info["label"]))

        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
        np.savez(
            dump_path,
            uids=np.array(uids),
            preds=np.array(preds, dtype=np.int64),
            labels=np.array(labels, dtype=np.int64),
        )
        print(f"[dump] wrote {len(uids)} test predictions -> {dump_path}")
