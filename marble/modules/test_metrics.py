# marble/modules/test_metrics.py
#
# Reusable comprehensive test-time classification metrics for MARBLE probes.
#
# log_classification_test_metrics() takes aggregated per-item predictions and a
# class-name list and logs, on the test split:
#   - test/f1_<class>        per-class F1 scalars (chartable across layers/encoders)
#   - test/f1_macro          macro-F1 scalar (only when log_f1_macro=True, so we
#                            don't double-log tasks whose YAML already logs a
#                            macro F1 — e.g. VGMLoopStructure, HXMSA)
#   - test/per_class_metrics a precision/recall/F1 HEATMAP (wandb.Image)
#   - test/confusion_matrix  the confusion-matrix heatmap (wandb.plot)
#
# Per-class precision/recall are folded into the heatmap instead of logged as
# scalars (a 13-class task would otherwise add 39 summary keys). Everything is
# defensive: a metric/plotting failure prints a warning and never breaks the
# test run.
from __future__ import annotations


def _per_class_heatmap_figure(classes, precision, recall, f1):
    """Return a matplotlib Figure: rows=classes, cols=[precision, recall, f1]."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    mat = np.array([precision, recall, f1], dtype=float).T  # (C, 3)
    height = max(1.5, 0.45 * len(classes) + 1.0)
    fig, ax = plt.subplots(figsize=(4.2, height))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["precision", "recall", "f1"])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Per-class precision / recall / F1 (test)")
    fig.tight_layout()
    return fig


def log_classification_test_metrics(module, preds, labels, classes, *,
                                    log_f1_macro=True, prefix="test"):
    """Log per-class F1 scalars + macro-F1 (optional) + per-class heatmap +
    confusion matrix from aggregated test predictions.

    Parameters
    ----------
    module : the LightningModule (uses ``module.log`` and ``module.logger``)
    preds, labels : iterables of int class indices (per test item)
    classes : list[str] class names, index-ordered (classes[i] is class i)
    log_f1_macro : log ``test/f1_macro`` too. Set False when the task config
        already logs a macro F1 (avoids double-logging the same metric).
    """
    preds = [int(p) for p in preds]
    labels = [int(x) for x in labels]
    idx = list(range(len(classes)))

    pr = rc = f1 = None
    try:
        from sklearn.metrics import f1_score, precision_recall_fscore_support
        pr, rc, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=idx, zero_division=0)
        for i, c in enumerate(classes):
            module.log(f"{prefix}/f1_{c}", float(f1[i]), sync_dist=True)
        if log_f1_macro:
            module.log(
                f"{prefix}/f1_macro",
                float(f1_score(labels, preds, labels=idx, average="macro",
                               zero_division=0)),
                sync_dist=True)
    except Exception as e:  # noqa: BLE001
        print(f"[test-metrics] per-class F1 scalars skipped: {e}")

    try:
        import wandb
        exp = getattr(getattr(module, "logger", None), "experiment", None)
        if exp is not None:
            payload = {}
            if pr is not None:
                payload[f"{prefix}/per_class_metrics"] = wandb.Image(
                    _per_class_heatmap_figure(classes, pr, rc, f1))
            if hasattr(wandb, "plot"):
                payload[f"{prefix}/confusion_matrix"] = wandb.plot.confusion_matrix(
                    y_true=list(labels), preds=list(preds), class_names=list(classes))
            if payload:
                exp.log(payload)
                import matplotlib.pyplot as plt
                plt.close("all")
    except Exception as e:  # noqa: BLE001
        print(f"[test-metrics] heatmap/confusion skipped: {e}")
