# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (415 bars)
- **Last close:** 1668.70
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
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 0.50% / -1.68%
- **Sum % (uncompounded):** 1.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.50% | 1.5% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.50% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.50% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 05:30:00 | 1259.25 | 1143.50 | 1194.19 | Stage2 pullback-breakout RSI=67 vol=1.6x ATR=33.28 |
| Stop hit — per-position SL triggered | 2024-07-18 05:30:00 | 1238.10 | 1154.29 | 1233.50 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-08-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 05:30:00 | 1242.93 | 1160.89 | 1217.02 | Stage2 pullback-breakout RSI=57 vol=2.4x ATR=35.35 |
| Stop hit — per-position SL triggered | 2024-08-14 05:30:00 | 1189.91 | 1165.77 | 1228.59 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-08-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 05:30:00 | 1349.35 | 1168.36 | 1241.39 | Stage2 pullback-breakout RSI=66 vol=4.5x ATR=50.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 05:30:00 | 1449.92 | 1179.98 | 1299.61 | T1 booked 50% @ 1449.92 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 05:30:00 | 1259.25 | 2024-07-18 05:30:00 | 1238.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2024-08-06 05:30:00 | 1242.93 | 2024-08-14 05:30:00 | 1189.91 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2024-08-19 05:30:00 | 1349.35 | 2024-08-27 05:30:00 | 1449.92 | PARTIAL | 0.50 | 7.45% |
