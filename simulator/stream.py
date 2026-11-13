"""Replay the held-out (future) test period against the API as a live stream.

python -m simulator.stream --rate 20 --n 5000 --reset
"""
import argparse, time
import pandas as pd, requests


def main(a):
    df = pd.read_csv("artifacts/test_stream.csv")
    if a.n:
        df = df.iloc[: a.n]
    if a.reset:
        requests.post(f"{a.url}/reset").raise_for_status()
    sess, delay = requests.Session(), 1.0 / a.rate
    for i, r in df.iterrows():
        payload = {c: float(r[c]) for c in df.columns if c.startswith("V")}
        payload.update(Time=float(r.Time), Amount=float(r.Amount),
                       transaction_id=f"tx{i:07d}", true_label=int(r.Class))
        sess.post(f"{a.url}/score", json=payload, timeout=10).raise_for_status()
        if i % 200 == 0:
            print(f"sent {i}")
        time.sleep(delay)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--rate", type=float, default=20, help="transactions/second")
    p.add_argument("--n", type=int, default=0, help="limit rows (0 = all)")
    p.add_argument("--reset", action="store_true")
    main(p.parse_args())
