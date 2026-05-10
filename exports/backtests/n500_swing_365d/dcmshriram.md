# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1231.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -1.13% / 0.00%
- **Sum % (uncompounded):** -4.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.13% | -4.5% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.13% | -4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | -1.13% | -4.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 05:30:00 | 1278.50 | 1190.20 | 1219.45 | Stage2 pullback-breakout RSI=62 vol=3.9x ATR=35.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 05:30:00 | 1350.46 | 1192.70 | 1237.14 | T1 booked 50% @ 1350.46 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 1278.50 | 1194.89 | 1248.98 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2025-11-21 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 05:30:00 | 1268.50 | 1199.31 | 1229.50 | Stage2 pullback-breakout RSI=57 vol=8.4x ATR=41.08 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 1206.88 | 1199.33 | 1226.82 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2025-12-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 05:30:00 | 1273.80 | 1202.22 | 1225.48 | Stage2 pullback-breakout RSI=58 vol=12.4x ATR=44.89 |
| Stop hit — per-position SL triggered | 2025-12-18 05:30:00 | 1206.46 | 1203.83 | 1228.73 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-27 05:30:00 | 1278.50 | 2025-10-29 05:30:00 | 1350.46 | PARTIAL | 0.50 | 5.63% |
| BUY | retest1 | 2025-10-27 05:30:00 | 1278.50 | 2025-10-31 05:30:00 | 1278.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-21 05:30:00 | 1268.50 | 2025-11-24 05:30:00 | 1206.88 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2025-12-11 05:30:00 | 1273.80 | 2025-12-18 05:30:00 | 1206.46 | STOP_HIT | 1.00 | -5.29% |
