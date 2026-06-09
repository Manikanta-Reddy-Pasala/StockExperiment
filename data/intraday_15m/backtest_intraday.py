#!/usr/bin/env python3
"""Reproduce the intraday MACD day-trade backtest on the 15m CSVs in this folder.
Usage: python3 backtest_intraday.py
Rules: MACD(12,26,9) cross vs signal while below/above zero, 200-period MA trend gate,
entry next bar OPEN (PIT), stop = 200-MA level at entry, target 1.5 R:R, square-off 15:15,
6 bps/side. Per-symbol independent capital; reports total return per side."""
import glob, os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COST = 0.0006; RR = 1.5; SQOFF = "15:15"

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def bt(df, side):
    cl = df["close"]
    macd = ema(cl, 12) - ema(cl, 26); sig = macd.ewm(span=9, adjust=False).mean()
    ma200 = cl.rolling(200).mean()
    cu = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd < 0)
    cd = (macd < sig) & (macd.shift(1) >= sig.shift(1)) & (macd > 0)
    rets = []; pos = None
    hm = df["datetime"].str[11:16]; day = df["datetime"].str[:10]
    for i in range(1, len(df)):
        o, h, l = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i]
        if pos:
            ex = None
            if side == "long":
                if l <= pos["stop"]: ex = pos["stop"]
                elif h >= pos["tgt"]: ex = pos["tgt"]
                elif bool(cd.iloc[i-1]): ex = o
            else:
                if h >= pos["stop"]: ex = pos["stop"]
                elif l <= pos["tgt"]: ex = pos["tgt"]
                elif bool(cu.iloc[i-1]): ex = o
            if ex is None and (hm.iloc[i] >= SQOFF or day.iloc[i] != pos["day"]): ex = o
            if ex is not None:
                r = (ex/pos["entry"]-1) if side == "long" else (pos["entry"]/ex-1)
                rets.append(r - 2*COST); pos = None
        if pos is None and hm.iloc[i] < SQOFF:
            trig = cu.iloc[i-1] if side == "long" else cd.iloc[i-1]
            m = ma200.iloc[i-1]
            if bool(trig) and pd.notna(m):
                px = o
                if side == "long" and px > m:
                    pos = dict(entry=px, stop=m, tgt=px+RR*(px-m), day=day.iloc[i])
                elif side == "short" and px < m:
                    pos = dict(entry=px, stop=m, tgt=px-RR*(m-px), day=day.iloc[i])
    if not rets: return None
    eq = 1.0
    for r in rets: eq *= (1+r)
    wr = 100*sum(1 for r in rets if r > 0)/len(rets)
    return dict(trades=len(rets), wr=round(wr,1), total_ret_pct=round((eq-1)*100,1))

res = {}
for f in sorted(glob.glob(os.path.join(HERE, "*_15m.csv"))):
    sym = os.path.basename(f).split("_")[0]
    df = pd.read_csv(f)
    for c in ("open","high","low","close"): df[c] = pd.to_numeric(df[c])
    res[sym] = dict(bars=len(df), long=bt(df, "long"), short=bt(df, "short"))
print(json.dumps(res, indent=1))
ls = [v["long"]["total_ret_pct"] for v in res.values() if v["long"]]
ss = [v["short"]["total_ret_pct"] for v in res.values() if v["short"]]
print(f"\nAVG long {sum(ls)/len(ls):+.1f}%  |  AVG short {sum(ss)/len(ss):+.1f}%  -> NO EDGE")
