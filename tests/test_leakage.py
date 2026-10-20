"""Tests that guard against label leakage and temporal contamination."""
import numpy as np
import pandas as pd
import pytest

from tokokita import config as C
from tokokita.features import FEATURE_COLUMNS, LABEL_COLUMN, build_panel, compute_features, make_labels, snapshot_dates
from tokokita.leakage import audit_single_feature_auc, build_leaky_panel, flag_leaky_features
from tokokita.splits import assert_no_temporal_overlap, temporal_split

AS_OF = pd.Timestamp("2024-06-01")


def test_features_ignore_future_transactions(small_ds):
    """Deleting or distorting everything after as_of must not change a single feature value."""
    tx = small_ds.transactions
    base = compute_features(tx, small_ds.customers, AS_OF)
    tampered = tx.copy()
    future = tampered["order_date"] > AS_OF
    assert future.any()
    tampered.loc[future, "order_value"] *= 1000
    tampered.loc[future, "discount_used"] = ~tampered.loc[future, "discount_used"]
    changed = compute_features(tampered, small_ds.customers, AS_OF)
    pd.testing.assert_frame_equal(base, changed)
    dropped = compute_features(tx[~future], small_ds.customers, AS_OF)
    pd.testing.assert_frame_equal(base, dropped)


def test_label_window_is_strictly_after_as_of():
    tx = pd.DataFrame({
        "customer_id": ["on_as_of", "day1", "day30", "day31"],
        "order_date": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-07-01", "2024-07-02"]),
    })
    labels = make_labels(tx, ["on_as_of", "day1", "day30", "day31", "never"], AS_OF, horizon=30)
    # an order ON the as_of date belongs to the feature window, not the label window
    assert labels["on_as_of"] == 1
    assert labels["day1"] == 0 and labels["day30"] == 0      # inside (as_of, as_of+30]
    assert labels["day31"] == 1 and labels["never"] == 1      # outside the window


def test_panel_has_no_overlap_between_feature_and_label_windows(small_ds):
    panel = build_panel(small_ds.transactions, small_ds.customers, snapshot_dates("2023-06-01", "2024-12-01"))
    assert (panel["as_of"] + pd.Timedelta(days=C.HORIZON_DAYS) <= pd.Timestamp(C.DATA_END)).all()
    # no feature column may be (or contain) the label or an identifier
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert not {"customer_id", "as_of"} & set(FEATURE_COLUMNS)
    # every active customer really did order in the 90 days before as_of
    assert (panel["frequency_90d"] >= 1).all()


def test_build_panel_refuses_labels_beyond_data_end(small_ds):
    with pytest.raises(ValueError):
        build_panel(small_ds.transactions, small_ds.customers, [pd.Timestamp("2024-12-15")])


def test_temporal_split_is_purged(small_ds):
    panel = build_panel(small_ds.transactions, small_ds.customers, snapshot_dates("2023-06-01", "2024-12-01"))
    s = temporal_split(panel)
    assert len(s.train) and len(s.val) and len(s.test)
    assert_no_temporal_overlap(s)
    h = pd.Timedelta(days=C.HORIZON_DAYS)
    assert s.train["as_of"].max() + h <= s.val["as_of"].min()     # train labels are known before validation starts
    assert s.val["as_of"].max() + h <= s.test["as_of"].min()      # validation labels are known before test starts
    # purged snapshots are dropped, not reassigned
    assert len(s.train) + len(s.val) + len(s.test) < len(panel) + 1


def test_audit_flags_leaky_features_but_not_clean_ones(small_ds):
    snaps = [pd.Timestamp("2024-10-01"), pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-01")]
    clean = build_panel(small_ds.transactions, small_ds.customers, snaps)
    leaky = build_leaky_panel(small_ds.transactions, small_ds.customers, snaps)
    assert flag_leaky_features(clean) == []
    assert "recency_days" in flag_leaky_features(leaky)
    assert audit_single_feature_auc(leaky, ["recency_days"])["strength"].iloc[0] > 0.99
    assert audit_single_feature_auc(clean)["strength"].max() < 0.9


def test_features_have_no_missing_or_infinite_values(small_ds):
    f = compute_features(small_ds.transactions, small_ds.customers, AS_OF)
    X = f[FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(X).all()
