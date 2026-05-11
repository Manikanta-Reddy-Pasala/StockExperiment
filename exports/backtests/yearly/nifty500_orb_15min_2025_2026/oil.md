# Oil India Ltd. (OIL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-09-05 15:25:00 (6225 bars)
- **Last close:** 396.50
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 8 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 24
- **Target hits / Stop hits / Partials:** 8 / 24 / 12
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 7.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 10 | 41.7% | 4 | 14 | 6 | 0.20% | 4.8% |
| BUY @ 2nd Alert (retest1) | 24 | 10 | 41.7% | 4 | 14 | 6 | 0.20% | 4.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 10 | 50.0% | 4 | 10 | 6 | 0.12% | 2.3% |
| SELL @ 2nd Alert (retest1) | 20 | 10 | 50.0% | 4 | 10 | 6 | 0.12% | 2.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 44 | 20 | 45.5% | 8 | 24 | 12 | 0.16% | 7.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 11:00:00 | 429.95 | 426.87 | 0.00 | ORB-long ORB[424.20,429.40] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-05-19 11:05:00 | 428.67 | 427.12 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 10:00:00 | 430.50 | 426.97 | 0.00 | ORB-long ORB[423.00,427.95] vol=2.5x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-05-20 10:05:00 | 428.63 | 427.20 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:15:00 | 430.80 | 428.13 | 0.00 | ORB-long ORB[426.40,430.00] vol=2.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-05-27 11:40:00 | 429.92 | 428.55 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 10:00:00 | 436.35 | 433.68 | 0.00 | ORB-long ORB[430.15,434.30] vol=3.2x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:15:00 | 438.76 | 434.70 | 0.00 | T1 1.5R @ 438.76 |
| Stop hit — per-position SL triggered | 2025-05-29 10:30:00 | 436.35 | 435.17 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:10:00 | 419.30 | 424.49 | 0.00 | ORB-short ORB[425.50,428.40] vol=1.9x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 12:20:00 | 417.68 | 423.49 | 0.00 | T1 1.5R @ 417.68 |
| Stop hit — per-position SL triggered | 2025-06-03 12:30:00 | 419.30 | 423.35 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:55:00 | 428.85 | 427.12 | 0.00 | ORB-long ORB[422.40,427.60] vol=3.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-06-05 11:10:00 | 427.78 | 427.47 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 424.15 | 423.21 | 0.00 | ORB-long ORB[420.15,423.50] vol=1.7x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 09:45:00 | 426.17 | 423.78 | 0.00 | T1 1.5R @ 426.17 |
| Target hit | 2025-06-06 11:50:00 | 424.60 | 424.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2025-06-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:40:00 | 429.00 | 427.03 | 0.00 | ORB-long ORB[424.95,428.50] vol=2.5x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-06-09 11:05:00 | 428.00 | 427.40 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:00:00 | 439.00 | 436.46 | 0.00 | ORB-long ORB[434.35,437.45] vol=2.3x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-06-10 10:10:00 | 437.85 | 436.77 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 09:50:00 | 440.45 | 441.81 | 0.00 | ORB-short ORB[441.05,444.60] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-06-26 10:45:00 | 442.01 | 441.59 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:15:00 | 434.15 | 437.13 | 0.00 | ORB-short ORB[435.20,440.80] vol=2.4x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-06-30 11:20:00 | 435.24 | 437.05 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 431.70 | 435.19 | 0.00 | ORB-short ORB[435.60,438.00] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-07-01 11:35:00 | 432.66 | 434.06 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:35:00 | 433.45 | 435.31 | 0.00 | ORB-short ORB[433.80,437.00] vol=1.5x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:35:00 | 431.62 | 434.27 | 0.00 | T1 1.5R @ 431.62 |
| Stop hit — per-position SL triggered | 2025-07-02 11:10:00 | 433.45 | 433.96 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 442.10 | 440.65 | 0.00 | ORB-long ORB[436.60,441.40] vol=2.4x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 09:40:00 | 443.80 | 441.21 | 0.00 | T1 1.5R @ 443.80 |
| Target hit | 2025-07-03 15:20:00 | 452.90 | 448.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2025-07-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:55:00 | 442.25 | 445.70 | 0.00 | ORB-short ORB[445.65,449.80] vol=1.5x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:10:00 | 440.80 | 445.12 | 0.00 | T1 1.5R @ 440.80 |
| Target hit | 2025-07-10 15:20:00 | 437.15 | 440.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 448.35 | 447.31 | 0.00 | ORB-long ORB[445.50,448.10] vol=1.9x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-07-17 09:35:00 | 447.44 | 447.37 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 10:10:00 | 451.85 | 450.22 | 0.00 | ORB-long ORB[445.45,451.45] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 450.58 | 450.27 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:30:00 | 452.30 | 448.14 | 0.00 | ORB-long ORB[443.75,448.90] vol=2.0x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-07-21 10:35:00 | 451.06 | 448.48 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:45:00 | 448.75 | 450.98 | 0.00 | ORB-short ORB[450.35,453.00] vol=1.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-07-22 11:35:00 | 449.67 | 450.68 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 11:15:00 | 454.95 | 452.38 | 0.00 | ORB-long ORB[448.00,454.40] vol=3.1x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-07-23 11:25:00 | 454.06 | 452.51 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:45:00 | 449.75 | 451.33 | 0.00 | ORB-short ORB[451.10,455.00] vol=1.5x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 448.02 | 449.78 | 0.00 | T1 1.5R @ 448.02 |
| Target hit | 2025-07-24 13:45:00 | 449.20 | 448.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 11:15:00 | 435.00 | 433.07 | 0.00 | ORB-long ORB[429.40,434.00] vol=4.1x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 13:05:00 | 436.68 | 433.76 | 0.00 | T1 1.5R @ 436.68 |
| Target hit | 2025-07-29 15:20:00 | 440.30 | 437.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:15:00 | 444.70 | 441.49 | 0.00 | ORB-long ORB[438.35,443.30] vol=2.3x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-07-31 11:30:00 | 443.41 | 441.65 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 424.15 | 425.66 | 0.00 | ORB-short ORB[424.25,429.90] vol=1.9x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 09:50:00 | 422.43 | 424.91 | 0.00 | T1 1.5R @ 422.43 |
| Target hit | 2025-08-05 10:40:00 | 423.00 | 422.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 11:15:00 | 431.75 | 430.61 | 0.00 | ORB-long ORB[427.60,431.50] vol=6.3x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:25:00 | 433.27 | 430.99 | 0.00 | T1 1.5R @ 433.27 |
| Stop hit — per-position SL triggered | 2025-08-07 11:50:00 | 431.75 | 431.40 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:15:00 | 425.30 | 428.00 | 0.00 | ORB-short ORB[428.75,433.30] vol=1.8x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:40:00 | 423.41 | 427.43 | 0.00 | T1 1.5R @ 423.41 |
| Target hit | 2025-08-11 15:05:00 | 424.00 | 423.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2025-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 10:10:00 | 404.40 | 405.70 | 0.00 | ORB-short ORB[404.85,409.35] vol=3.1x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-08-20 11:00:00 | 405.18 | 405.39 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 409.35 | 409.70 | 0.00 | ORB-short ORB[409.65,413.75] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-08-22 13:25:00 | 410.34 | 409.60 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 11:15:00 | 405.80 | 407.14 | 0.00 | ORB-short ORB[406.70,410.10] vol=1.9x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-08-25 11:25:00 | 406.53 | 407.16 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 11:05:00 | 385.60 | 386.75 | 0.00 | ORB-short ORB[387.00,391.00] vol=1.9x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-08-29 11:20:00 | 386.58 | 386.70 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:50:00 | 394.00 | 392.16 | 0.00 | ORB-long ORB[388.80,392.30] vol=1.9x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:15:00 | 395.67 | 392.48 | 0.00 | T1 1.5R @ 395.67 |
| Target hit | 2025-09-01 15:20:00 | 400.60 | 395.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:55:00 | 396.35 | 394.71 | 0.00 | ORB-long ORB[393.10,395.85] vol=5.1x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-09-05 11:40:00 | 395.30 | 395.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-19 11:00:00 | 429.95 | 2025-05-19 11:05:00 | 428.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-20 10:00:00 | 430.50 | 2025-05-20 10:05:00 | 428.63 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-27 11:15:00 | 430.80 | 2025-05-27 11:40:00 | 429.92 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-05-29 10:00:00 | 436.35 | 2025-05-29 10:15:00 | 438.76 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-05-29 10:00:00 | 436.35 | 2025-05-29 10:30:00 | 436.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 11:10:00 | 419.30 | 2025-06-03 12:20:00 | 417.68 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-03 11:10:00 | 419.30 | 2025-06-03 12:30:00 | 419.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 10:55:00 | 428.85 | 2025-06-05 11:10:00 | 427.78 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-06 09:35:00 | 424.15 | 2025-06-06 09:45:00 | 426.17 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-06 09:35:00 | 424.15 | 2025-06-06 11:50:00 | 424.60 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-06-09 10:40:00 | 429.00 | 2025-06-09 11:05:00 | 428.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-10 10:00:00 | 439.00 | 2025-06-10 10:10:00 | 437.85 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-26 09:50:00 | 440.45 | 2025-06-26 10:45:00 | 442.01 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-30 11:15:00 | 434.15 | 2025-06-30 11:20:00 | 435.24 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-01 10:50:00 | 431.70 | 2025-07-01 11:35:00 | 432.66 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-02 09:35:00 | 433.45 | 2025-07-02 10:35:00 | 431.62 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-02 09:35:00 | 433.45 | 2025-07-02 11:10:00 | 433.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 09:35:00 | 442.10 | 2025-07-03 09:40:00 | 443.80 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-07-03 09:35:00 | 442.10 | 2025-07-03 15:20:00 | 452.90 | TARGET_HIT | 0.50 | 2.44% |
| SELL | retest1 | 2025-07-10 10:55:00 | 442.25 | 2025-07-10 11:10:00 | 440.80 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-10 10:55:00 | 442.25 | 2025-07-10 15:20:00 | 437.15 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-07-17 09:30:00 | 448.35 | 2025-07-17 09:35:00 | 447.44 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-18 10:10:00 | 451.85 | 2025-07-18 10:15:00 | 450.58 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-21 10:30:00 | 452.30 | 2025-07-21 10:35:00 | 451.06 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 10:45:00 | 448.75 | 2025-07-22 11:35:00 | 449.67 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-23 11:15:00 | 454.95 | 2025-07-23 11:25:00 | 454.06 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-24 09:45:00 | 449.75 | 2025-07-24 11:15:00 | 448.02 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-24 09:45:00 | 449.75 | 2025-07-24 13:45:00 | 449.20 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-07-29 11:15:00 | 435.00 | 2025-07-29 13:05:00 | 436.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-29 11:15:00 | 435.00 | 2025-07-29 15:20:00 | 440.30 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-07-31 11:15:00 | 444.70 | 2025-07-31 11:30:00 | 443.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-05 09:30:00 | 424.15 | 2025-08-05 09:50:00 | 422.43 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-08-05 09:30:00 | 424.15 | 2025-08-05 10:40:00 | 423.00 | TARGET_HIT | 0.50 | 0.27% |
| BUY | retest1 | 2025-08-07 11:15:00 | 431.75 | 2025-08-07 11:25:00 | 433.27 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-08-07 11:15:00 | 431.75 | 2025-08-07 11:50:00 | 431.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-11 11:15:00 | 425.30 | 2025-08-11 11:40:00 | 423.41 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-11 11:15:00 | 425.30 | 2025-08-11 15:05:00 | 424.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-20 10:10:00 | 404.40 | 2025-08-20 11:00:00 | 405.18 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-22 11:00:00 | 409.35 | 2025-08-22 13:25:00 | 410.34 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-25 11:15:00 | 405.80 | 2025-08-25 11:25:00 | 406.53 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-29 11:05:00 | 385.60 | 2025-08-29 11:20:00 | 386.58 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-01 10:50:00 | 394.00 | 2025-09-01 11:15:00 | 395.67 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-09-01 10:50:00 | 394.00 | 2025-09-01 15:20:00 | 400.60 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-09-05 10:55:00 | 396.35 | 2025-09-05 11:40:00 | 395.30 | STOP_HIT | 1.00 | -0.26% |
