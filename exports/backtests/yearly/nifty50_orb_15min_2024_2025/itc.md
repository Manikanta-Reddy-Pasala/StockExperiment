# ITC (ITC)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-07-09 15:25:00 (3021 bars)
- **Last close:** 451.90
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 15
- **Target hits / Stop hits / Partials:** 2 / 15 / 5
- **Avg / median % per leg:** 0.00% / -0.16%
- **Sum % (uncompounded):** 0.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 0 | 12 | 3 | -0.04% | -0.6% |
| BUY @ 2nd Alert (retest1) | 15 | 3 | 20.0% | 0 | 12 | 3 | -0.04% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.09% | 0.6% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.09% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 7 | 31.8% | 2 | 15 | 5 | 0.00% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 432.60 | 430.10 | 0.00 | ORB-long ORB[429.00,431.80] vol=1.9x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-05-14 11:20:00 | 431.71 | 430.15 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 426.65 | 428.64 | 0.00 | ORB-short ORB[428.10,430.45] vol=2.3x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-05-16 10:15:00 | 427.72 | 427.52 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:45:00 | 435.55 | 432.47 | 0.00 | ORB-long ORB[428.90,433.45] vol=2.0x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-05-17 11:50:00 | 434.47 | 433.04 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 439.30 | 437.57 | 0.00 | ORB-long ORB[435.40,437.75] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-05-22 09:40:00 | 438.42 | 438.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 426.70 | 428.50 | 0.00 | ORB-short ORB[427.20,430.95] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-05-30 09:35:00 | 427.57 | 428.26 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 428.50 | 427.76 | 0.00 | ORB-long ORB[425.15,428.15] vol=2.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-05-31 09:55:00 | 427.60 | 427.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-03 11:15:00 | 429.35 | 431.63 | 0.00 | ORB-short ORB[432.00,434.90] vol=1.8x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-06-03 12:05:00 | 430.26 | 431.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 10:00:00 | 429.65 | 423.75 | 0.00 | ORB-long ORB[418.05,423.90] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 10:35:00 | 433.84 | 426.66 | 0.00 | T1 1.5R @ 433.84 |
| Stop hit — per-position SL triggered | 2024-06-05 11:55:00 | 429.65 | 429.29 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:00:00 | 436.90 | 433.54 | 0.00 | ORB-long ORB[431.10,436.30] vol=1.9x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-06-06 10:40:00 | 435.27 | 434.23 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:05:00 | 436.60 | 434.21 | 0.00 | ORB-long ORB[431.10,436.15] vol=1.7x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-06-07 11:25:00 | 435.44 | 434.43 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:40:00 | 421.90 | 423.38 | 0.00 | ORB-short ORB[422.30,425.30] vol=1.6x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 12:40:00 | 420.60 | 422.48 | 0.00 | T1 1.5R @ 420.60 |
| Target hit | 2024-06-21 15:20:00 | 419.35 | 420.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:45:00 | 422.55 | 423.26 | 0.00 | ORB-short ORB[422.65,424.00] vol=2.1x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:05:00 | 421.75 | 423.01 | 0.00 | T1 1.5R @ 421.75 |
| Target hit | 2024-06-25 12:20:00 | 421.70 | 421.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:40:00 | 425.40 | 423.63 | 0.00 | ORB-long ORB[422.55,424.05] vol=3.2x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-06-26 11:55:00 | 424.69 | 424.39 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:10:00 | 429.00 | 427.81 | 0.00 | ORB-long ORB[425.50,428.70] vol=4.2x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-07-03 11:20:00 | 428.30 | 427.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:50:00 | 430.15 | 428.75 | 0.00 | ORB-long ORB[427.05,430.00] vol=3.5x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 11:10:00 | 431.08 | 429.08 | 0.00 | T1 1.5R @ 431.08 |
| Stop hit — per-position SL triggered | 2024-07-04 11:30:00 | 430.15 | 429.22 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:50:00 | 437.05 | 435.22 | 0.00 | ORB-long ORB[433.65,436.00] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:55:00 | 438.47 | 435.69 | 0.00 | T1 1.5R @ 438.47 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 437.05 | 435.81 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 449.15 | 446.42 | 0.00 | ORB-long ORB[444.50,448.00] vol=2.2x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-07-09 10:35:00 | 447.97 | 446.91 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 11:15:00 | 432.60 | 2024-05-14 11:20:00 | 431.71 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-16 09:30:00 | 426.65 | 2024-05-16 10:15:00 | 427.72 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-17 10:45:00 | 435.55 | 2024-05-17 11:50:00 | 434.47 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-22 09:35:00 | 439.30 | 2024-05-22 09:40:00 | 438.42 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-05-30 09:30:00 | 426.70 | 2024-05-30 09:35:00 | 427.57 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-31 09:45:00 | 428.50 | 2024-05-31 09:55:00 | 427.60 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-03 11:15:00 | 429.35 | 2024-06-03 12:05:00 | 430.26 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-05 10:00:00 | 429.65 | 2024-06-05 10:35:00 | 433.84 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2024-06-05 10:00:00 | 429.65 | 2024-06-05 11:55:00 | 429.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-06 10:00:00 | 436.90 | 2024-06-06 10:40:00 | 435.27 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-07 11:05:00 | 436.60 | 2024-06-07 11:25:00 | 435.44 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-21 10:40:00 | 421.90 | 2024-06-21 12:40:00 | 420.60 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-06-21 10:40:00 | 421.90 | 2024-06-21 15:20:00 | 419.35 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-06-25 10:45:00 | 422.55 | 2024-06-25 11:05:00 | 421.75 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2024-06-25 10:45:00 | 422.55 | 2024-06-25 12:20:00 | 421.70 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-06-26 10:40:00 | 425.40 | 2024-06-26 11:55:00 | 424.69 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-07-03 11:10:00 | 429.00 | 2024-07-03 11:20:00 | 428.30 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-07-04 10:50:00 | 430.15 | 2024-07-04 11:10:00 | 431.08 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-04 10:50:00 | 430.15 | 2024-07-04 11:30:00 | 430.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-08 09:50:00 | 437.05 | 2024-07-08 09:55:00 | 438.47 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-07-08 09:50:00 | 437.05 | 2024-07-08 10:00:00 | 437.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 10:15:00 | 449.15 | 2024-07-09 10:35:00 | 447.97 | STOP_HIT | 1.00 | -0.26% |
