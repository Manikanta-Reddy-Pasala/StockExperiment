# Aadhar Housing Finance Ltd. (AADHARHFC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 502.75
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 11
- **Target hits / Stop hits / Partials:** 3 / 11 / 6
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 4.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.35% | 3.9% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.35% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.06% | 0.5% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.06% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.22% | 4.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 478.25 | 477.42 | 0.00 | ORB-long ORB[474.70,478.00] vol=6.1x ATR=1.55 |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 476.70 | 477.51 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 474.25 | 475.38 | 0.00 | ORB-short ORB[475.50,479.75] vol=5.1x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:05:00 | 473.22 | 475.34 | 0.00 | T1 1.5R @ 473.22 |
| Stop hit — per-position SL triggered | 2026-02-10 13:40:00 | 474.25 | 474.47 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 464.60 | 467.43 | 0.00 | ORB-short ORB[466.05,472.85] vol=3.3x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-02-16 10:45:00 | 466.27 | 467.10 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 458.80 | 459.94 | 0.00 | ORB-short ORB[459.05,462.95] vol=1.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:55:00 | 457.01 | 459.42 | 0.00 | T1 1.5R @ 457.01 |
| Target hit | 2026-02-18 11:00:00 | 457.15 | 456.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 480.40 | 477.06 | 0.00 | ORB-long ORB[474.65,480.00] vol=6.7x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-02-20 11:10:00 | 478.91 | 477.68 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 441.75 | 444.97 | 0.00 | ORB-short ORB[447.60,452.00] vol=4.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 443.72 | 442.16 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 11:05:00 | 451.70 | 450.01 | 0.00 | ORB-long ORB[446.95,450.90] vol=8.7x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-03-06 12:30:00 | 450.27 | 450.36 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:10:00 | 463.70 | 459.38 | 0.00 | ORB-long ORB[451.25,458.00] vol=5.2x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 11:15:00 | 467.86 | 460.57 | 0.00 | T1 1.5R @ 467.86 |
| Target hit | 2026-03-16 11:45:00 | 464.45 | 464.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-24 10:50:00 | 447.85 | 444.86 | 0.00 | ORB-long ORB[442.00,447.00] vol=4.0x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 11:00:00 | 451.28 | 445.58 | 0.00 | T1 1.5R @ 451.28 |
| Target hit | 2026-03-24 15:20:00 | 462.65 | 454.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 500.60 | 497.54 | 0.00 | ORB-long ORB[490.90,495.00] vol=4.6x ATR=2.56 |
| Stop hit — per-position SL triggered | 2026-04-17 09:35:00 | 498.04 | 497.50 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 488.40 | 493.91 | 0.00 | ORB-short ORB[491.00,497.95] vol=3.1x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:35:00 | 485.42 | 490.98 | 0.00 | T1 1.5R @ 485.42 |
| Stop hit — per-position SL triggered | 2026-04-22 13:00:00 | 488.40 | 490.84 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 496.60 | 494.59 | 0.00 | ORB-long ORB[492.05,496.00] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-04-28 10:10:00 | 495.02 | 495.48 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 496.60 | 492.09 | 0.00 | ORB-long ORB[486.80,493.90] vol=2.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:55:00 | 499.24 | 494.93 | 0.00 | T1 1.5R @ 499.24 |
| Stop hit — per-position SL triggered | 2026-05-04 10:05:00 | 496.60 | 495.91 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 499.65 | 501.01 | 0.00 | ORB-short ORB[501.05,505.00] vol=2.3x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-05-08 11:20:00 | 501.02 | 501.06 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 478.25 | 2026-02-09 11:15:00 | 476.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-10 11:00:00 | 474.25 | 2026-02-10 11:05:00 | 473.22 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2026-02-10 11:00:00 | 474.25 | 2026-02-10 13:40:00 | 474.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-16 10:40:00 | 464.60 | 2026-02-16 10:45:00 | 466.27 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 09:40:00 | 458.80 | 2026-02-18 09:55:00 | 457.01 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-18 09:40:00 | 458.80 | 2026-02-18 11:00:00 | 457.15 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-20 10:55:00 | 480.40 | 2026-02-20 11:10:00 | 478.91 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-25 09:45:00 | 441.75 | 2026-02-25 09:50:00 | 443.72 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-06 11:05:00 | 451.70 | 2026-03-06 12:30:00 | 450.27 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-16 11:10:00 | 463.70 | 2026-03-16 11:15:00 | 467.86 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2026-03-16 11:10:00 | 463.70 | 2026-03-16 11:45:00 | 464.45 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-03-24 10:50:00 | 447.85 | 2026-03-24 11:00:00 | 451.28 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-03-24 10:50:00 | 447.85 | 2026-03-24 15:20:00 | 462.65 | TARGET_HIT | 0.50 | 3.30% |
| BUY | retest1 | 2026-04-17 09:30:00 | 500.60 | 2026-04-17 09:35:00 | 498.04 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-04-22 10:50:00 | 488.40 | 2026-04-22 12:35:00 | 485.42 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-22 10:50:00 | 488.40 | 2026-04-22 13:00:00 | 488.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 09:35:00 | 496.60 | 2026-04-28 10:10:00 | 495.02 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-04 09:30:00 | 496.60 | 2026-05-04 09:55:00 | 499.24 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-05-04 09:30:00 | 496.60 | 2026-05-04 10:05:00 | 496.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:15:00 | 499.65 | 2026-05-08 11:20:00 | 501.02 | STOP_HIT | 1.00 | -0.27% |
