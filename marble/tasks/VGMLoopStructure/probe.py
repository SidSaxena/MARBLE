# marble/tasks/VGMLoopStructure/probe.py
#
# Generic HookTheoryStructure ProbeAudioTask + VGM-specific comprehensive test
# metrics. The parent logs the config metrics (acc, f1_macro); we add per-class
# precision/recall/F1 and a confusion matrix for EVERY run (the class imbalance
# makes macro-F1 + acc insufficient: intro_loop is the hard class). All of it is
# defensive (try/except) so a metrics hiccup can never break the core test run.
# The raw per-file prediction dump stays opt-in via MARBLE_DUMP_TEST_PREDS.
import os

import torch

from marble.tasks.HookTheoryStructure.probe import ProbeAudioTask as _BaseProbeAudioTask

__all__ = ["ProbeAudioTask"]


class ProbeAudioTask(_BaseProbeAudioTask):
    def _aggregate_test(self):
        file_dict: dict = {}
        for entry in self._test_file_outputs:
            info = file_dict.setdefault(entry["uid"], {"logits": [], "label": entry["label"]})
            info["logits"].append(entry["logit"])
        uids, preds, labels = [], [], []
        for uid, info in file_dict.items():
            ml = torch.stack(info["logits"]).mean(dim=0)
            uids.append(str(uid))
            preds.append(int(ml.argmax().item()))
            labels.append(int(info["label"]))
        return uids, preds, labels

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()  # logs configured metrics (acc, f1_macro)
        if not getattr(self, "_test_file_outputs", None):
            return
        uids, preds, labels = self._aggregate_test()

        try:
            from marble.tasks.VGMLoopStructure.datamodule import _VGMLoopStructureAudioBase as _DM
            classes = [_DM.IDX2LABEL[i] for i in sorted(_DM.IDX2LABEL)]
        except Exception:
            classes = [f"class_{i}" for i in range(int(max(labels + preds)) + 1)]

        # Per-class F1 scalars + per-class heatmap + confusion matrix.
        # f1_macro is logged by the YAML metric, so don't double-log it here.
        try:
            from marble.modules.test_metrics import log_classification_test_metrics
            log_classification_test_metrics(self, preds, labels, classes,
                                            log_f1_macro=False)
        except Exception as e:
            print(f"[vgm-metrics] per-class metrics skipped: {e}")

        dump_path = os.environ.get("MARBLE_DUMP_TEST_PREDS")
        if dump_path:
            import numpy as np
            os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
            np.savez(dump_path, uids=np.array(uids),
                     preds=np.array(preds, dtype=np.int64),
                     labels=np.array(labels, dtype=np.int64))
            print(f"[dump] wrote {len(uids)} test predictions -> {dump_path}")
