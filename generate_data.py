import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
accounts = [f"A{i:03d}" for i in range(1, 41)]
rows = []
for _ in range(180):
    source, target = rng.choice(accounts, 2, replace=False)
    amount = round(float(rng.lognormal(mean=4.2, sigma=0.8)), 2)
    rows.append((source, target, amount, f"2026-11-{int(rng.integers(1, 22)):02d}"))

# Planted five-account circular transfer pattern for demonstration
ring = ["A031", "A032", "A033", "A034", "A035"]
for i in range(len(ring)):
    for _ in range(2):
        rows.append((ring[i], ring[(i + 1) % len(ring)],
                     round(float(rng.uniform(800, 1800)), 2), "2026-11-20"))

df = pd.DataFrame(rows, columns=[
    "source_account", "target_account", "amount", "transaction_date"
])
df["synthetic_fraud_flag"] = (
    df["source_account"].isin(ring) & df["target_account"].isin(ring)
).astype(int)
df.to_csv("transactions.csv", index=False)
print(f"Created transactions.csv with {len(df)} rows.")
