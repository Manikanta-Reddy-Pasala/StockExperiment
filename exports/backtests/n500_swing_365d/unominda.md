# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1178.90
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** 0.44% / -3.32%
- **Sum % (uncompounded):** 3.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.44% | 3.1% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.44% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | 0.44% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 05:30:00 | 1109.80 | 1006.49 | 1057.34 | Stage2 pullback-breakout RSI=67 vol=1.8x ATR=33.63 |
| Stop hit — per-position SL triggered | 2025-07-11 05:30:00 | 1073.00 | 1014.97 | 1081.16 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-08-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 05:30:00 | 1103.10 | 1024.89 | 1075.34 | Stage2 pullback-breakout RSI=58 vol=4.1x ATR=30.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 05:30:00 | 1163.23 | 1031.84 | 1101.43 | T1 booked 50% @ 1163.23 |
| Target hit | 2025-09-26 05:30:00 | 1268.90 | 1094.76 | 1285.90 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-10-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 05:30:00 | 1366.30 | 1107.97 | 1303.72 | Stage2 pullback-breakout RSI=68 vol=2.8x ATR=34.27 |
| Stop hit — per-position SL triggered | 2025-10-08 05:30:00 | 1314.90 | 1109.99 | 1304.45 | SL hit (bars_held=1) |

### Cycle 4 — BUY (started 2025-11-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 05:30:00 | 1319.30 | 1132.39 | 1244.88 | Stage2 pullback-breakout RSI=63 vol=3.1x ATR=40.36 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 1279.50 | 1148.40 | 1278.26 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2025-11-26 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 05:30:00 | 1320.10 | 1151.46 | 1282.85 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=34.62 |
| Stop hit — per-position SL triggered | 2025-12-03 05:30:00 | 1268.17 | 1158.59 | 1287.95 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2026-01-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 05:30:00 | 1321.20 | 1178.99 | 1277.07 | Stage2 pullback-breakout RSI=62 vol=3.0x ATR=29.79 |
| Stop hit — per-position SL triggered | 2026-01-08 05:30:00 | 1276.51 | 1184.17 | 1287.56 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-27 05:30:00 | 1109.80 | 2025-07-11 05:30:00 | 1073.00 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest1 | 2025-08-05 05:30:00 | 1103.10 | 2025-08-18 05:30:00 | 1163.23 | PARTIAL | 0.50 | 5.45% |
| BUY | retest1 | 2025-08-05 05:30:00 | 1103.10 | 2025-09-26 05:30:00 | 1268.90 | TARGET_HIT | 0.50 | 15.03% |
| BUY | retest1 | 2025-10-07 05:30:00 | 1366.30 | 2025-10-08 05:30:00 | 1314.90 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest1 | 2025-11-10 05:30:00 | 1319.30 | 2025-11-24 05:30:00 | 1279.50 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest1 | 2025-11-26 05:30:00 | 1320.10 | 2025-12-03 05:30:00 | 1268.17 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest1 | 2026-01-02 05:30:00 | 1321.20 | 2026-01-08 05:30:00 | 1276.51 | STOP_HIT | 1.00 | -3.38% |
