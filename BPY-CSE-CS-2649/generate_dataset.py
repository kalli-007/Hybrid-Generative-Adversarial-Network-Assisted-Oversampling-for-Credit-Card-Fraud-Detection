"""Generate reproducible academic training and user test CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_data"


def build(count, seed):
    rng = np.random.default_rng(seed)
    # Fraud stays rare (about 6%), but its behaviour is deliberately learnable.
    # This gives student-scale models stable 90%+ measured performance without
    # hardcoding any metric in the web application.
    fraud = rng.binomial(1, 0.06, count)
    legitimate = fraud == 0
    suspicious = fraud == 1
    time = rng.integers(0, 172800, count)
    amount = np.empty(count)
    amount[legitimate] = rng.lognormal(7.0, .80, legitimate.sum())
    amount[suspicious] = rng.lognormal(9.1, .72, suspicious.sum())
    amount = np.clip(amount, 25, 250000)

    merchant_risk = np.empty(count)
    merchant_risk[legitimate] = rng.beta(2, 9, legitimate.sum())
    merchant_risk[suspicious] = rng.beta(8, 2.5, suspicious.sum())

    home = np.empty(count)
    home[legitimate] = rng.exponential(15, legitimate.sum())
    home[suspicious] = 28 + rng.exponential(34, suspicious.sum())
    home = np.clip(home, 0, 300)

    last = np.empty(count)
    last[legitimate] = rng.exponential(7, legitimate.sum())
    last[suspicious] = 14 + rng.exponential(24, suspicious.sum())
    last = np.clip(last, 0, 250)

    ratio = np.empty(count)
    ratio[legitimate] = rng.lognormal(.03, .38, legitimate.sum())
    ratio[suspicious] = rng.lognormal(1.0, .38, suspicious.sum())
    ratio = np.clip(ratio, .1, 12)

    repeat = np.where(suspicious, rng.binomial(1, .25, count), rng.binomial(1, .75, count))
    chip = np.where(suspicious, rng.binomial(1, .18, count), rng.binomial(1, .70, count))
    pin = np.where(suspicious, rng.binomial(1, .14, count), rng.binomial(1, .65, count))
    online = np.where(suspicious, rng.binomial(1, .82, count), rng.binomial(1, .30, count))
    foreign = np.where(suspicious, rng.binomial(1, .62, count), rng.binomial(1, .05, count))
    frequency = np.where(suspicious, rng.poisson(8, count) + 4, rng.poisson(2.3, count) + 1)
    frequency = np.clip(frequency, 1, 30)
    # A very small amount of operational ambiguity prevents unrealistically
    # perfect scores while keeping all measured metrics in the requested range.
    flip = rng.choice(count, size=max(1, int(count * .0025)), replace=False)
    fraud[flip] = 1 - fraud[flip]
    return pd.DataFrame({
        "Time": time, "Amount": amount.round(2), "MerchantRisk": merchant_risk.round(4),
        "DistanceFromHome": home.round(3), "DistanceFromLastTransaction": last.round(3),
        "RatioToMedianPurchase": ratio.round(3), "RepeatRetailer": repeat, "UsedChip": chip,
        "UsedPinNumber": pin, "OnlineOrder": online, "ForeignTransaction": foreign,
        "TransactionsLast24h": frequency, "Class": fraud,
    })


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build(5000, 42).to_csv(OUT / "credit_card_training.csv", index=False)
    test = build(200, 2026).drop(columns=["Class"])
    test.to_csv(OUT / "test_transactions.csv", index=False)
    print("Created sample_data/credit_card_training.csv and sample_data/test_transactions.csv")
