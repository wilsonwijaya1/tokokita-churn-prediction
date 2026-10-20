"""Central configuration. Every constant that shapes results lives here."""
from __future__ import annotations

SEED = 42

# --- Simulation calendar -----------------------------------------------------
DATA_START = "2023-01-01"
SIGNUP_END = "2024-09-30"
DATA_END = "2024-12-31"
N_CUSTOMERS = 3000

# --- Problem definition ------------------------------------------------------
HORIZON_DAYS = 30            # churn_30d = no order in (as_of, as_of + 30 days]
ACTIVE_WINDOW_DAYS = 90      # "active customer" = at least one order in the last 90 days

# --- Temporal split (snapshots are taken on the 1st and 15th of every month) -
FIRST_SNAPSHOT = "2023-04-01"
LAST_SNAPSHOT = "2024-12-01"
VAL_START = "2024-08-01"     # validation snapshots start here
TEST_START = "2024-11-01"    # out-of-time test snapshots start here

# --- Categorical vocabularies (fixed so one-hot columns are stable) ----------
AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "55+"]
GENDERS = ["F", "M"]
ACQ_CHANNELS = ["organic", "paid_social", "paid_search", "referral", "email"]
ORDER_CHANNELS = ["mobile_app", "web", "marketplace"]
CATEGORIES = ["electronics", "fashion", "home", "beauty", "sports", "books"]
TIERS = ["low", "medium", "high"]
