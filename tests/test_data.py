import pandas as pd
import pandas.testing as pdt

from tokokita.data import generate_dataset


def test_generator_is_deterministic():
    a = generate_dataset(n_customers=150, seed=123)
    b = generate_dataset(n_customers=150, seed=123)
    pdt.assert_frame_equal(a.transactions, b.transactions)
    pdt.assert_frame_equal(a.customers, b.customers)
    pdt.assert_frame_equal(a.latent, b.latent)


def test_different_seed_changes_data():
    a = generate_dataset(n_customers=150, seed=1)
    b = generate_dataset(n_customers=150, seed=2)
    assert not a.transactions.equals(b.transactions)


def test_schema_and_integrity(small_ds):
    cu, tx = small_ds.customers, small_ds.transactions
    assert cu["customer_id"].is_unique and tx["order_id"].is_unique
    assert set(tx["customer_id"]) <= set(cu["customer_id"])
    assert tx["order_value"].min() >= 50_000
    assert tx.isna().sum().sum() == 0 and cu.isna().sum().sum() == 0
    # nobody orders before signing up, and nothing is dated after the end of the simulation
    merged = tx.merge(cu[["customer_id", "signup_date"]], on="customer_id")
    assert (merged["order_date"] >= merged["signup_date"]).all()
    assert tx["order_date"].max() <= pd.Timestamp("2024-12-31")


def test_latent_truth_is_separate_from_public_tables(small_ds):
    # the model-facing tables must not carry the simulation's hidden state
    hidden = {"value_tier", "defection_date", "monthly_orders", "wind_down_days", "discount_propensity"}
    assert not hidden & set(small_ds.customers.columns)
    assert not hidden & set(small_ds.transactions.columns)
