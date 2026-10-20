"""Purged temporal split.

Rows are snapshots (customer, as_of). Because each label looks ``horizon`` days ahead, a plain
"train before date X" split still leaks: the last training labels are computed from orders that
fall *inside* the validation/test period. The purge rule removes those rows:

    train : as_of + horizon <= val_start      (labels fully known before validation begins)
    val   : val_start <= as_of, as_of + horizon <= test_start
    test  : as_of >= test_start               (never touched during development)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    horizon: int
    val_start: pd.Timestamp
    test_start: pd.Timestamp


def temporal_split(panel: pd.DataFrame, val_start: str = C.VAL_START, test_start: str = C.TEST_START,
                   horizon: int = C.HORIZON_DAYS) -> Split:
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    h = pd.Timedelta(days=horizon)
    label_end = panel["as_of"] + h
    train = panel[label_end <= val_start]
    val = panel[(panel["as_of"] >= val_start) & (label_end <= test_start)]
    test = panel[panel["as_of"] >= test_start]
    return Split(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True),
                 horizon, val_start, test_start)


def assert_no_temporal_overlap(split: Split) -> None:
    """Raise if any label window of an earlier set reaches into the feature period of a later set."""
    h = pd.Timedelta(days=split.horizon)
    if (split.train["as_of"].max() + h) > split.val["as_of"].min():
        raise AssertionError("Training labels overlap the validation feature period.")
    if (split.val["as_of"].max() + h) > split.test["as_of"].min():
        raise AssertionError("Validation labels overlap the test feature period.")
    if split.train["as_of"].max() >= split.val["as_of"].min() or split.val["as_of"].max() >= split.test["as_of"].min():
        raise AssertionError("Sets are not ordered in time.")
