"""Tests for marble.utils.sweep_coords.parse_sweep_coords.

The parser canonicalizes a wandb run's (name, job_type, tags) into queryable
sweep coordinates {layer, fold, stage, repr} so the dashboard can group by
layer and average across folds. Cases mirror the real BPS-Motif run shapes:
  - layer test runs:  name "layer-6-test-fold0", job_type "test-fold0"
  - layer fit runs:   name "layer-6-fit",        job_type "fit" (no fold)
  - meanall runs:     name "meanall-fold3-test"  /  "layer-meanall-test"
"""

from marble.utils.sweep_coords import parse_sweep_coords, resolve_coords


def test_layer_test_run_with_fold():
    c = parse_sweep_coords("layer-6-test-fold0", "test-fold0", ["layer-6", "layer-sweep"])
    assert c == {"layer": 6, "fold": 0, "window": None, "stage": "test", "repr": "single"}


def test_layer_fit_run_has_no_fold():
    c = parse_sweep_coords("layer-6-fit", "fit", ["layer-6", "layer-sweep"])
    assert c == {"layer": 6, "fold": None, "window": None, "stage": "fit", "repr": "single"}


def test_meanall_test_run():
    c = parse_sweep_coords("meanall-fold3-test", "test", ["layer-sweep"])
    assert c == {"layer": -1, "fold": 3, "window": None, "stage": "test", "repr": "meanall"}


def test_meanall_fit_run():
    c = parse_sweep_coords("meanall-fold0-fit", "fit", ["layer-sweep"])
    assert c == {"layer": -1, "fold": 0, "window": None, "stage": "fit", "repr": "meanall"}


def test_gen_sweep_meanall_naming():
    # run_sweep_local's meanall convention: layer-meanall-{stage}
    c = parse_sweep_coords("layer-meanall-test", "test", [])
    assert c == {"layer": -1, "fold": None, "window": None, "stage": "test", "repr": "meanall"}


def test_double_digit_layer():
    c = parse_sweep_coords("layer-12-test-fold4", "test-fold4", ["layer-12"])
    assert c == {"layer": 12, "fold": 4, "window": None, "stage": "test", "repr": "single"}


def test_fold_recovered_from_job_type_when_absent_in_name():
    c = parse_sweep_coords("layer-3-test", "test-fold2", ["layer-3"])
    assert c["fold"] == 2
    assert c["layer"] == 3
    assert c["stage"] == "test"


def test_layer_recovered_from_tag_when_name_uninformative():
    c = parse_sweep_coords("some-run-test-fold1", "test-fold1", ["layer-9", "layer-sweep"])
    assert c["layer"] == 9
    assert c["fold"] == 1
    assert c["stage"] == "test"


def test_layer_sweep_tag_is_not_mistaken_for_a_layer():
    # "layer-sweep" must NOT parse as layer number
    c = parse_sweep_coords("layer-0-fit", "fit", ["layer-sweep", "layer-0"])
    assert c["layer"] == 0


def test_missing_everything_is_graceful():
    c = parse_sweep_coords("", None, None)
    assert c == {
        "layer": None,
        "fold": None,
        "window": None,
        "stage": None,
        "repr": "single",
    }


# ---- resolve_coords: applies authoritative overrides (used by the callback) ----


def test_resolve_overrides_fold_for_fit_run():
    # fit run name has no fold, but the datamodule knows fold_idx=2
    c = resolve_coords("layer-6-fit", "fit", ["layer-6"], fold_idx=2)
    assert c == {"layer": 6, "fold": 2, "window": None, "stage": "fit", "repr": "single"}


def test_resolve_fold_idx_zero_is_respected():
    # fold 0 is falsy; must still override (not be skipped)
    c = resolve_coords("layer-6-fit", "fit", ["layer-6"], fold_idx=0)
    assert c["fold"] == 0


def test_resolve_stage_override():
    c = resolve_coords("layer-3-x", None, ["layer-3"], stage="test")
    assert c["stage"] == "test"
    assert c["layer"] == 3


def test_resolve_without_overrides_matches_parse():
    name, jt, tags = "layer-5-test-fold1", "test-fold1", ["layer-5"]
    assert resolve_coords(name, jt, tags) == parse_sweep_coords(name, jt, tags)


# ---- window coord: within-piece window-size sweep (analogous to fold) ----


def test_window_parsed_from_name():
    c = parse_sweep_coords("layer-8-test-window4", "test", ["layer-8"])
    assert c["window"] == 4
    assert c["layer"] == 8
    assert c["stage"] == "test"


def test_window_parsed_from_job_type_and_tag():
    assert parse_sweep_coords("layer-3-test", "test-window12", ["layer-3"])["window"] == 12
    assert parse_sweep_coords("x-test", "test", ["window-24"])["window"] == 24


def test_window_absent_is_none():
    assert parse_sweep_coords("layer-6-test-fold0", "test-fold0", ["layer-6"])["window"] is None


def test_resolve_overrides_window_for_fit_run():
    # fit run name carries no window token, but the datamodule knows window=8
    c = resolve_coords("layer-8-fit", "fit", ["layer-8"], window=8)
    assert c == {"layer": 8, "fold": None, "window": 8, "stage": "fit", "repr": "single"}


def test_resolve_window_and_fold_together():
    c = resolve_coords("layer-2-fit", "fit", ["layer-2"], fold_idx=1, window=16)
    assert c["fold"] == 1 and c["window"] == 16 and c["layer"] == 2
