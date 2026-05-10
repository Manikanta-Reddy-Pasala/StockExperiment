# SBFC Finance Ltd. (SBFC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 98.60
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 13
- **Target hits / Stop hits / Partials:** 3 / 13 / 6
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 2.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.25% | 2.7% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.25% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.04% | -0.5% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.04% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 9 | 40.9% | 3 | 13 | 6 | 0.10% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 97.20 | 96.22 | 0.00 | ORB-long ORB[94.95,95.75] vol=1.9x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-02-17 15:20:00 | 97.07 | 96.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 96.17 | 96.96 | 0.00 | ORB-short ORB[96.91,97.50] vol=1.6x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-02-18 10:00:00 | 96.42 | 96.78 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 96.33 | 96.59 | 0.00 | ORB-short ORB[96.52,97.26] vol=1.9x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:05:00 | 96.03 | 96.46 | 0.00 | T1 1.5R @ 96.03 |
| Stop hit — per-position SL triggered | 2026-02-19 12:15:00 | 96.33 | 96.45 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:15:00 | 97.17 | 96.55 | 0.00 | ORB-long ORB[95.81,96.95] vol=1.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-02-24 12:45:00 | 96.79 | 96.79 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 96.70 | 96.33 | 0.00 | ORB-long ORB[96.05,96.62] vol=3.8x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:00:00 | 97.19 | 96.65 | 0.00 | T1 1.5R @ 97.19 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 96.70 | 96.84 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:55:00 | 95.13 | 95.59 | 0.00 | ORB-short ORB[95.39,96.69] vol=1.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:35:00 | 94.63 | 95.19 | 0.00 | T1 1.5R @ 94.63 |
| Stop hit — per-position SL triggered | 2026-02-27 12:45:00 | 95.13 | 94.97 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 94.77 | 93.69 | 0.00 | ORB-long ORB[92.49,93.52] vol=4.9x ATR=0.39 |
| Stop hit — per-position SL triggered | 2026-03-06 10:35:00 | 94.38 | 93.76 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:30:00 | 86.98 | 87.37 | 0.00 | ORB-short ORB[87.11,88.00] vol=1.9x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-03-19 09:50:00 | 87.33 | 87.34 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:40:00 | 92.91 | 92.18 | 0.00 | ORB-long ORB[91.42,92.30] vol=4.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-04-13 09:45:00 | 92.37 | 92.33 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 97.20 | 96.28 | 0.00 | ORB-long ORB[95.06,95.94] vol=3.4x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:25:00 | 97.84 | 97.24 | 0.00 | T1 1.5R @ 97.84 |
| Target hit | 2026-04-16 11:55:00 | 99.13 | 99.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 97.53 | 98.16 | 0.00 | ORB-short ORB[98.05,99.00] vol=2.4x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-04-22 12:40:00 | 97.96 | 97.65 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:20:00 | 95.17 | 95.64 | 0.00 | ORB-short ORB[95.41,96.75] vol=5.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-04-24 10:35:00 | 95.43 | 95.62 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:10:00 | 94.40 | 94.63 | 0.00 | ORB-short ORB[94.52,95.07] vol=1.5x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:25:00 | 94.16 | 94.61 | 0.00 | T1 1.5R @ 94.16 |
| Target hit | 2026-04-28 14:20:00 | 94.30 | 94.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 92.89 | 93.56 | 0.00 | ORB-short ORB[93.09,93.97] vol=2.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 93.17 | 93.56 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:55:00 | 94.93 | 94.56 | 0.00 | ORB-long ORB[93.68,94.71] vol=2.0x ATR=0.38 |
| Stop hit — per-position SL triggered | 2026-05-06 10:50:00 | 94.55 | 94.66 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 95.97 | 95.66 | 0.00 | ORB-long ORB[94.87,95.93] vol=2.6x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:40:00 | 96.49 | 95.84 | 0.00 | T1 1.5R @ 96.49 |
| Target hit | 2026-05-07 14:50:00 | 96.90 | 97.09 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 10:30:00 | 97.20 | 2026-02-17 15:20:00 | 97.07 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2026-02-18 09:45:00 | 96.17 | 2026-02-18 10:00:00 | 96.42 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 11:00:00 | 96.33 | 2026-02-19 12:05:00 | 96.03 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-19 11:00:00 | 96.33 | 2026-02-19 12:15:00 | 96.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:15:00 | 97.17 | 2026-02-24 12:45:00 | 96.79 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-26 09:30:00 | 96.70 | 2026-02-26 10:00:00 | 97.19 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-02-26 09:30:00 | 96.70 | 2026-02-26 11:30:00 | 96.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 09:55:00 | 95.13 | 2026-02-27 10:35:00 | 94.63 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-27 09:55:00 | 95.13 | 2026-02-27 12:45:00 | 95.13 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 10:30:00 | 94.77 | 2026-03-06 10:35:00 | 94.38 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-03-19 09:30:00 | 86.98 | 2026-03-19 09:50:00 | 87.33 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-13 09:40:00 | 92.91 | 2026-04-13 09:45:00 | 92.37 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-04-16 09:30:00 | 97.20 | 2026-04-16 10:25:00 | 97.84 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-16 09:30:00 | 97.20 | 2026-04-16 11:55:00 | 99.13 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2026-04-22 09:45:00 | 97.53 | 2026-04-22 12:40:00 | 97.96 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-04-24 10:20:00 | 95.17 | 2026-04-24 10:35:00 | 95.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-28 11:10:00 | 94.40 | 2026-04-28 11:25:00 | 94.16 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-04-28 11:10:00 | 94.40 | 2026-04-28 14:20:00 | 94.30 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2026-05-05 11:15:00 | 92.89 | 2026-05-05 11:25:00 | 93.17 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-06 09:55:00 | 94.93 | 2026-05-06 10:50:00 | 94.55 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-07 09:35:00 | 95.97 | 2026-05-07 09:40:00 | 96.49 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-07 09:35:00 | 95.97 | 2026-05-07 14:50:00 | 96.90 | TARGET_HIT | 0.50 | 0.97% |
