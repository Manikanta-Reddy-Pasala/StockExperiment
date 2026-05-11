# Tata Motors Passenger Vehicles Ltd. (TMPV)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18458 bars)
- **Last close:** 355.50
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 19 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 60
- **Target hits / Stop hits / Partials:** 19 / 60 / 27
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 16.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 21 | 36.8% | 9 | 36 | 12 | 0.10% | 5.7% |
| BUY @ 2nd Alert (retest1) | 57 | 21 | 36.8% | 9 | 36 | 12 | 0.10% | 5.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 49 | 25 | 51.0% | 10 | 24 | 15 | 0.23% | 11.1% |
| SELL @ 2nd Alert (retest1) | 49 | 25 | 51.0% | 10 | 24 | 15 | 0.23% | 11.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 106 | 46 | 43.4% | 19 | 60 | 27 | 0.16% | 16.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:55:00 | 429.76 | 426.54 | 0.00 | ORB-long ORB[421.91,428.09] vol=1.6x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 10:00:00 | 432.17 | 427.51 | 0.00 | T1 1.5R @ 432.17 |
| Target hit | 2025-05-15 15:20:00 | 441.82 | 435.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:30:00 | 443.94 | 441.71 | 0.00 | ORB-long ORB[439.73,442.52] vol=1.9x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:55:00 | 446.00 | 443.11 | 0.00 | T1 1.5R @ 446.00 |
| Target hit | 2025-05-16 10:40:00 | 444.15 | 444.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-05-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 10:05:00 | 440.91 | 438.87 | 0.00 | ORB-long ORB[436.36,440.45] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-05-29 10:25:00 | 439.81 | 439.10 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:40:00 | 436.64 | 439.23 | 0.00 | ORB-short ORB[438.48,441.55] vol=1.6x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:45:00 | 435.08 | 438.76 | 0.00 | T1 1.5R @ 435.08 |
| Target hit | 2025-05-30 14:05:00 | 435.61 | 435.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2025-06-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 10:35:00 | 430.27 | 432.01 | 0.00 | ORB-short ORB[431.70,435.36] vol=1.8x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:10:00 | 428.88 | 431.52 | 0.00 | T1 1.5R @ 428.88 |
| Target hit | 2025-06-03 15:20:00 | 426.21 | 429.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:45:00 | 426.82 | 428.96 | 0.00 | ORB-short ORB[428.09,432.36] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-06-05 11:00:00 | 427.73 | 428.84 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 425.91 | 427.93 | 0.00 | ORB-short ORB[427.42,430.73] vol=1.8x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-06-06 10:05:00 | 426.88 | 427.24 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:45:00 | 439.27 | 436.88 | 0.00 | ORB-long ORB[432.18,438.00] vol=2.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-06-09 10:00:00 | 437.92 | 437.25 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:40:00 | 439.30 | 437.97 | 0.00 | ORB-long ORB[433.55,439.21] vol=4.3x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-06-10 09:45:00 | 438.39 | 438.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 440.39 | 442.16 | 0.00 | ORB-short ORB[442.18,447.09] vol=2.3x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:35:00 | 439.12 | 441.77 | 0.00 | T1 1.5R @ 439.12 |
| Target hit | 2025-06-12 15:20:00 | 432.97 | 437.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:45:00 | 414.97 | 412.58 | 0.00 | ORB-long ORB[409.15,412.73] vol=1.5x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-06-24 12:20:00 | 413.96 | 413.35 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 11:00:00 | 411.64 | 410.35 | 0.00 | ORB-long ORB[409.27,411.27] vol=2.5x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 410.89 | 410.48 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 09:30:00 | 418.48 | 417.03 | 0.00 | ORB-long ORB[415.27,418.18] vol=2.0x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-06-30 09:35:00 | 417.74 | 417.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:05:00 | 416.76 | 419.19 | 0.00 | ORB-short ORB[419.48,423.27] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 417.62 | 419.09 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:10:00 | 414.24 | 413.11 | 0.00 | ORB-long ORB[411.00,413.64] vol=2.1x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 11:35:00 | 415.13 | 413.38 | 0.00 | T1 1.5R @ 415.13 |
| Stop hit — per-position SL triggered | 2025-07-17 13:55:00 | 414.24 | 414.39 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:40:00 | 412.91 | 413.33 | 0.00 | ORB-short ORB[412.97,415.67] vol=1.6x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-07-18 10:50:00 | 413.70 | 413.35 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:30:00 | 408.48 | 410.32 | 0.00 | ORB-short ORB[409.09,413.33] vol=1.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 09:40:00 | 407.17 | 409.53 | 0.00 | T1 1.5R @ 407.17 |
| Stop hit — per-position SL triggered | 2025-07-21 09:45:00 | 408.48 | 409.43 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 411.21 | 414.18 | 0.00 | ORB-short ORB[413.33,417.48] vol=2.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 13:15:00 | 409.73 | 411.74 | 0.00 | T1 1.5R @ 409.73 |
| Target hit | 2025-07-22 15:20:00 | 408.21 | 410.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:50:00 | 419.21 | 420.62 | 0.00 | ORB-short ORB[421.24,425.03] vol=2.3x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 12:20:00 | 417.85 | 420.18 | 0.00 | T1 1.5R @ 417.85 |
| Target hit | 2025-07-25 15:20:00 | 416.36 | 418.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:15:00 | 394.88 | 395.72 | 0.00 | ORB-short ORB[395.52,398.76] vol=3.6x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-08-06 11:50:00 | 395.58 | 395.60 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:40:00 | 413.00 | 410.95 | 0.00 | ORB-long ORB[407.82,412.64] vol=1.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-08-18 09:55:00 | 411.44 | 411.13 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 413.97 | 410.90 | 0.00 | ORB-long ORB[407.94,411.03] vol=2.2x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:05:00 | 415.64 | 412.52 | 0.00 | T1 1.5R @ 415.64 |
| Target hit | 2025-08-19 15:20:00 | 424.45 | 421.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 10:55:00 | 417.82 | 420.22 | 0.00 | ORB-short ORB[420.00,424.73] vol=2.0x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-08-20 11:10:00 | 418.68 | 420.06 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:15:00 | 413.15 | 414.38 | 0.00 | ORB-short ORB[414.18,419.09] vol=3.0x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 12:20:00 | 411.97 | 414.00 | 0.00 | T1 1.5R @ 411.97 |
| Stop hit — per-position SL triggered | 2025-08-22 14:20:00 | 413.15 | 413.56 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:35:00 | 415.45 | 413.95 | 0.00 | ORB-long ORB[412.30,414.97] vol=1.5x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-08-25 10:05:00 | 414.70 | 414.58 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:15:00 | 409.42 | 407.94 | 0.00 | ORB-long ORB[406.39,408.91] vol=1.9x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:40:00 | 410.40 | 408.33 | 0.00 | T1 1.5R @ 410.40 |
| Target hit | 2025-09-01 15:20:00 | 418.06 | 413.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:00:00 | 436.24 | 433.47 | 0.00 | ORB-long ORB[430.45,435.73] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-09-10 11:05:00 | 435.48 | 433.59 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:30:00 | 434.67 | 432.46 | 0.00 | ORB-long ORB[428.70,434.15] vol=2.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-09-12 09:50:00 | 433.71 | 433.35 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:00:00 | 435.61 | 433.54 | 0.00 | ORB-long ORB[432.27,433.91] vol=2.3x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-09-16 10:10:00 | 434.94 | 433.78 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 436.67 | 435.76 | 0.00 | ORB-long ORB[432.91,436.36] vol=3.4x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-09-17 10:40:00 | 435.58 | 436.09 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:30:00 | 400.48 | 404.97 | 0.00 | ORB-short ORB[403.88,409.09] vol=1.8x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:40:00 | 398.51 | 404.36 | 0.00 | T1 1.5R @ 398.51 |
| Stop hit — per-position SL triggered | 2025-09-25 11:00:00 | 400.48 | 403.68 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:40:00 | 407.58 | 405.69 | 0.00 | ORB-long ORB[401.42,406.48] vol=2.0x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:45:00 | 409.84 | 406.13 | 0.00 | T1 1.5R @ 409.84 |
| Target hit | 2025-09-26 13:25:00 | 408.61 | 408.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-09-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 09:40:00 | 409.21 | 411.15 | 0.00 | ORB-short ORB[409.85,415.15] vol=1.8x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 410.60 | 410.62 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:55:00 | 420.55 | 417.09 | 0.00 | ORB-long ORB[411.64,415.94] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-10-01 10:05:00 | 419.15 | 417.68 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:55:00 | 399.65 | 395.75 | 0.00 | ORB-long ORB[389.55,395.45] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-10-16 11:00:00 | 398.28 | 395.83 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 401.50 | 399.63 | 0.00 | ORB-long ORB[396.15,400.65] vol=2.2x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 400.46 | 399.85 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:45:00 | 402.15 | 399.60 | 0.00 | ORB-long ORB[396.60,400.40] vol=3.1x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-10-20 10:50:00 | 401.02 | 399.67 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:40:00 | 409.35 | 407.60 | 0.00 | ORB-long ORB[403.65,408.20] vol=2.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-10-27 10:50:00 | 408.44 | 408.19 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:35:00 | 413.35 | 411.33 | 0.00 | ORB-long ORB[409.75,411.90] vol=1.7x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-10-28 09:55:00 | 412.27 | 412.11 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:45:00 | 414.80 | 411.58 | 0.00 | ORB-long ORB[407.05,412.65] vol=2.0x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-10-29 10:55:00 | 413.95 | 411.89 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:35:00 | 416.90 | 415.00 | 0.00 | ORB-long ORB[411.65,414.75] vol=3.2x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-10-31 09:40:00 | 415.99 | 415.34 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:35:00 | 409.25 | 412.74 | 0.00 | ORB-short ORB[413.25,417.50] vol=1.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-11-04 10:40:00 | 410.18 | 412.55 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 403.80 | 404.97 | 0.00 | ORB-short ORB[404.15,407.20] vol=1.7x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-11-07 09:35:00 | 404.74 | 404.93 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:45:00 | 410.80 | 408.97 | 0.00 | ORB-long ORB[406.00,409.70] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-11-10 10:00:00 | 409.75 | 409.24 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 406.10 | 408.36 | 0.00 | ORB-short ORB[407.50,411.40] vol=3.3x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:10:00 | 404.70 | 407.32 | 0.00 | T1 1.5R @ 404.70 |
| Target hit | 2025-11-12 15:20:00 | 402.30 | 404.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 11:15:00 | 393.95 | 395.92 | 0.00 | ORB-short ORB[394.30,399.60] vol=2.9x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-11-14 11:25:00 | 394.93 | 395.67 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:30:00 | 361.50 | 359.94 | 0.00 | ORB-long ORB[358.00,360.80] vol=1.9x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-11-21 09:40:00 | 360.62 | 360.15 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 355.90 | 358.12 | 0.00 | ORB-short ORB[357.00,359.75] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-11-28 09:55:00 | 356.75 | 357.60 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:00:00 | 361.95 | 360.74 | 0.00 | ORB-long ORB[358.70,361.75] vol=1.8x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 10:05:00 | 363.17 | 361.08 | 0.00 | T1 1.5R @ 363.17 |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 361.95 | 361.26 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 356.60 | 357.92 | 0.00 | ORB-short ORB[357.10,361.70] vol=2.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 357.22 | 357.67 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:50:00 | 351.80 | 354.20 | 0.00 | ORB-short ORB[353.70,356.20] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-08 10:00:00 | 352.60 | 353.88 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:40:00 | 342.95 | 344.88 | 0.00 | ORB-short ORB[343.80,348.65] vol=2.0x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-12-09 10:05:00 | 344.08 | 344.31 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 342.30 | 345.04 | 0.00 | ORB-short ORB[344.20,347.15] vol=2.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-12-10 11:20:00 | 343.08 | 344.83 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:30:00 | 344.25 | 345.11 | 0.00 | ORB-short ORB[344.40,347.45] vol=1.6x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:45:00 | 343.06 | 344.77 | 0.00 | T1 1.5R @ 343.06 |
| Stop hit — per-position SL triggered | 2025-12-15 09:50:00 | 344.25 | 344.72 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 349.80 | 347.95 | 0.00 | ORB-long ORB[345.55,349.20] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 348.94 | 348.15 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:10:00 | 363.75 | 360.78 | 0.00 | ORB-long ORB[357.65,361.15] vol=2.2x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:25:00 | 364.82 | 361.54 | 0.00 | T1 1.5R @ 364.82 |
| Stop hit — per-position SL triggered | 2025-12-23 13:05:00 | 363.75 | 362.97 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:15:00 | 359.55 | 361.70 | 0.00 | ORB-short ORB[362.00,364.90] vol=1.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-24 11:45:00 | 360.35 | 360.93 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 356.00 | 357.80 | 0.00 | ORB-short ORB[357.00,360.40] vol=1.7x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:20:00 | 355.07 | 357.52 | 0.00 | T1 1.5R @ 355.07 |
| Stop hit — per-position SL triggered | 2025-12-26 12:05:00 | 356.00 | 357.10 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:50:00 | 364.40 | 362.67 | 0.00 | ORB-long ORB[360.65,362.85] vol=2.0x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:55:00 | 365.57 | 363.06 | 0.00 | T1 1.5R @ 365.57 |
| Target hit | 2025-12-31 15:20:00 | 367.60 | 365.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2026-01-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:50:00 | 369.95 | 367.90 | 0.00 | ORB-long ORB[366.05,369.65] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-01-01 10:00:00 | 368.84 | 368.46 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 359.70 | 362.82 | 0.00 | ORB-short ORB[362.00,365.35] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 360.52 | 362.32 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:00:00 | 353.15 | 351.40 | 0.00 | ORB-long ORB[347.10,352.30] vol=1.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-01-14 11:35:00 | 352.25 | 351.71 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:20:00 | 355.50 | 353.44 | 0.00 | ORB-long ORB[349.65,353.95] vol=1.6x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:25:00 | 357.00 | 353.74 | 0.00 | T1 1.5R @ 357.00 |
| Target hit | 2026-01-16 13:40:00 | 355.90 | 356.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — SELL (started 2026-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:40:00 | 341.75 | 343.46 | 0.00 | ORB-short ORB[342.65,346.60] vol=1.5x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:55:00 | 340.19 | 342.28 | 0.00 | T1 1.5R @ 340.19 |
| Target hit | 2026-01-20 15:20:00 | 337.95 | 339.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2026-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 09:55:00 | 349.80 | 347.99 | 0.00 | ORB-long ORB[344.90,349.15] vol=1.5x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-01-23 10:40:00 | 348.78 | 348.75 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-02-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:10:00 | 358.85 | 353.44 | 0.00 | ORB-long ORB[347.10,349.95] vol=1.9x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 357.54 | 355.49 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 378.55 | 373.87 | 0.00 | ORB-long ORB[369.60,373.70] vol=1.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2026-02-09 11:05:00 | 377.32 | 374.22 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 377.10 | 379.38 | 0.00 | ORB-short ORB[377.75,382.50] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-02-10 09:35:00 | 378.20 | 379.34 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:45:00 | 385.95 | 384.28 | 0.00 | ORB-long ORB[382.10,385.60] vol=2.4x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-02-11 12:00:00 | 384.80 | 384.70 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 379.00 | 377.03 | 0.00 | ORB-long ORB[373.55,377.00] vol=2.2x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 378.09 | 377.32 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:00:00 | 380.95 | 379.46 | 0.00 | ORB-long ORB[376.10,380.25] vol=4.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:35:00 | 382.87 | 379.95 | 0.00 | T1 1.5R @ 382.87 |
| Target hit | 2026-02-25 13:25:00 | 381.65 | 381.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2026-02-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:55:00 | 385.35 | 383.85 | 0.00 | ORB-long ORB[380.10,383.35] vol=5.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:05:00 | 386.82 | 384.30 | 0.00 | T1 1.5R @ 386.82 |
| Target hit | 2026-02-26 13:55:00 | 388.75 | 388.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 339.70 | 343.76 | 0.00 | ORB-short ORB[343.55,347.00] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:00:00 | 338.28 | 342.21 | 0.00 | T1 1.5R @ 338.28 |
| Target hit | 2026-03-11 15:20:00 | 334.95 | 339.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 317.90 | 320.70 | 0.00 | ORB-short ORB[320.05,324.15] vol=1.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:45:00 | 316.18 | 319.92 | 0.00 | T1 1.5R @ 316.18 |
| Target hit | 2026-03-13 15:10:00 | 314.35 | 314.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — BUY (started 2026-03-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:45:00 | 327.15 | 324.06 | 0.00 | ORB-long ORB[319.90,324.25] vol=2.8x ATR=1.36 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 325.79 | 324.54 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-04-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:45:00 | 360.65 | 354.65 | 0.00 | ORB-long ORB[349.25,354.15] vol=2.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 359.32 | 354.97 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-04-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-20 09:30:00 | 352.80 | 355.54 | 0.00 | ORB-short ORB[353.60,358.30] vol=1.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-04-20 09:50:00 | 354.14 | 355.00 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:55:00 | 355.80 | 358.10 | 0.00 | ORB-short ORB[356.35,361.20] vol=1.8x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 12:50:00 | 354.07 | 356.86 | 0.00 | T1 1.5R @ 354.07 |
| Target hit | 2026-04-23 15:20:00 | 351.50 | 355.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 343.10 | 341.85 | 0.00 | ORB-long ORB[339.00,342.65] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-05-05 10:00:00 | 341.98 | 342.08 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:55:00 | 429.76 | 2025-05-15 10:00:00 | 432.17 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-05-15 09:55:00 | 429.76 | 2025-05-15 15:20:00 | 441.82 | TARGET_HIT | 0.50 | 2.81% |
| BUY | retest1 | 2025-05-16 09:30:00 | 443.94 | 2025-05-16 09:55:00 | 446.00 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-05-16 09:30:00 | 443.94 | 2025-05-16 10:40:00 | 444.15 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-05-29 10:05:00 | 440.91 | 2025-05-29 10:25:00 | 439.81 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-30 09:40:00 | 436.64 | 2025-05-30 09:45:00 | 435.08 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-05-30 09:40:00 | 436.64 | 2025-05-30 14:05:00 | 435.61 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-06-03 10:35:00 | 430.27 | 2025-06-03 11:10:00 | 428.88 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-06-03 10:35:00 | 430.27 | 2025-06-03 15:20:00 | 426.21 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-06-05 10:45:00 | 426.82 | 2025-06-05 11:00:00 | 427.73 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-06 09:35:00 | 425.91 | 2025-06-06 10:05:00 | 426.88 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-09 09:45:00 | 439.27 | 2025-06-09 10:00:00 | 437.92 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-10 09:40:00 | 439.30 | 2025-06-10 09:45:00 | 438.39 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-12 11:10:00 | 440.39 | 2025-06-12 11:35:00 | 439.12 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-12 11:10:00 | 440.39 | 2025-06-12 15:20:00 | 432.97 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2025-06-24 10:45:00 | 414.97 | 2025-06-24 12:20:00 | 413.96 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-26 11:00:00 | 411.64 | 2025-06-26 11:15:00 | 410.89 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-30 09:30:00 | 418.48 | 2025-06-30 09:35:00 | 417.74 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-11 11:05:00 | 416.76 | 2025-07-11 11:10:00 | 417.62 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-17 11:10:00 | 414.24 | 2025-07-17 11:35:00 | 415.13 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-07-17 11:10:00 | 414.24 | 2025-07-17 13:55:00 | 414.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:40:00 | 412.91 | 2025-07-18 10:50:00 | 413.70 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-21 09:30:00 | 408.48 | 2025-07-21 09:40:00 | 407.17 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-21 09:30:00 | 408.48 | 2025-07-21 09:45:00 | 408.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 09:30:00 | 411.21 | 2025-07-22 13:15:00 | 409.73 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-22 09:30:00 | 411.21 | 2025-07-22 15:20:00 | 408.21 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-07-25 10:50:00 | 419.21 | 2025-07-25 12:20:00 | 417.85 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-25 10:50:00 | 419.21 | 2025-07-25 15:20:00 | 416.36 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-08-06 11:15:00 | 394.88 | 2025-08-06 11:50:00 | 395.58 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-08-18 09:40:00 | 413.00 | 2025-08-18 09:55:00 | 411.44 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-08-19 09:50:00 | 413.97 | 2025-08-19 10:05:00 | 415.64 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-19 09:50:00 | 413.97 | 2025-08-19 15:20:00 | 424.45 | TARGET_HIT | 0.50 | 2.53% |
| SELL | retest1 | 2025-08-20 10:55:00 | 417.82 | 2025-08-20 11:10:00 | 418.68 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-22 11:15:00 | 413.15 | 2025-08-22 12:20:00 | 411.97 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-22 11:15:00 | 413.15 | 2025-08-22 14:20:00 | 413.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-25 09:35:00 | 415.45 | 2025-08-25 10:05:00 | 414.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-01 11:15:00 | 409.42 | 2025-09-01 11:40:00 | 410.40 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-01 11:15:00 | 409.42 | 2025-09-01 15:20:00 | 418.06 | TARGET_HIT | 0.50 | 2.11% |
| BUY | retest1 | 2025-09-10 11:00:00 | 436.24 | 2025-09-10 11:05:00 | 435.48 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-12 09:30:00 | 434.67 | 2025-09-12 09:50:00 | 433.71 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-16 10:00:00 | 435.61 | 2025-09-16 10:10:00 | 434.94 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-17 09:35:00 | 436.67 | 2025-09-17 10:40:00 | 435.58 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-25 10:30:00 | 400.48 | 2025-09-25 10:40:00 | 398.51 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-09-25 10:30:00 | 400.48 | 2025-09-25 11:00:00 | 400.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-26 09:40:00 | 407.58 | 2025-09-26 09:45:00 | 409.84 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-09-26 09:40:00 | 407.58 | 2025-09-26 13:25:00 | 408.61 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-09-29 09:40:00 | 409.21 | 2025-09-29 10:15:00 | 410.60 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-01 09:55:00 | 420.55 | 2025-10-01 10:05:00 | 419.15 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-16 10:55:00 | 399.65 | 2025-10-16 11:00:00 | 398.28 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-17 09:35:00 | 401.50 | 2025-10-17 09:40:00 | 400.46 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-20 10:45:00 | 402.15 | 2025-10-20 10:50:00 | 401.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-27 09:40:00 | 409.35 | 2025-10-27 10:50:00 | 408.44 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-28 09:35:00 | 413.35 | 2025-10-28 09:55:00 | 412.27 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-29 10:45:00 | 414.80 | 2025-10-29 10:55:00 | 413.95 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-31 09:35:00 | 416.90 | 2025-10-31 09:40:00 | 415.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-04 10:35:00 | 409.25 | 2025-11-04 10:40:00 | 410.18 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-07 09:30:00 | 403.80 | 2025-11-07 09:35:00 | 404.74 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-10 09:45:00 | 410.80 | 2025-11-10 10:00:00 | 409.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-12 10:00:00 | 406.10 | 2025-11-12 10:10:00 | 404.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-12 10:00:00 | 406.10 | 2025-11-12 15:20:00 | 402.30 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-11-14 11:15:00 | 393.95 | 2025-11-14 11:25:00 | 394.93 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-21 09:30:00 | 361.50 | 2025-11-21 09:40:00 | 360.62 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-28 09:45:00 | 355.90 | 2025-11-28 09:55:00 | 356.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-01 10:00:00 | 361.95 | 2025-12-01 10:05:00 | 363.17 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-01 10:00:00 | 361.95 | 2025-12-01 10:15:00 | 361.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 11:05:00 | 356.60 | 2025-12-03 11:35:00 | 357.22 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-08 09:50:00 | 351.80 | 2025-12-08 10:00:00 | 352.60 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-09 09:40:00 | 342.95 | 2025-12-09 10:05:00 | 344.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-10 11:00:00 | 342.30 | 2025-12-10 11:20:00 | 343.08 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-15 09:30:00 | 344.25 | 2025-12-15 09:45:00 | 343.06 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-15 09:30:00 | 344.25 | 2025-12-15 09:50:00 | 344.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 09:30:00 | 349.80 | 2025-12-17 09:40:00 | 348.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-23 11:10:00 | 363.75 | 2025-12-23 11:25:00 | 364.82 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-23 11:10:00 | 363.75 | 2025-12-23 13:05:00 | 363.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-24 10:15:00 | 359.55 | 2025-12-24 11:45:00 | 360.35 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-26 11:10:00 | 356.00 | 2025-12-26 11:20:00 | 355.07 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-26 11:10:00 | 356.00 | 2025-12-26 12:05:00 | 356.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:50:00 | 364.40 | 2025-12-31 10:55:00 | 365.57 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-31 10:50:00 | 364.40 | 2025-12-31 15:20:00 | 367.60 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2026-01-01 09:50:00 | 369.95 | 2026-01-01 10:00:00 | 368.84 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-08 11:15:00 | 359.70 | 2026-01-08 11:35:00 | 360.52 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-14 11:00:00 | 353.15 | 2026-01-14 11:35:00 | 352.25 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-16 10:20:00 | 355.50 | 2026-01-16 10:25:00 | 357.00 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-16 10:20:00 | 355.50 | 2026-01-16 13:40:00 | 355.90 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2026-01-20 09:40:00 | 341.75 | 2026-01-20 10:55:00 | 340.19 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-01-20 09:40:00 | 341.75 | 2026-01-20 15:20:00 | 337.95 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2026-01-23 09:55:00 | 349.80 | 2026-01-23 10:40:00 | 348.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-01 10:10:00 | 358.85 | 2026-02-01 10:15:00 | 357.54 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-09 10:55:00 | 378.55 | 2026-02-09 11:05:00 | 377.32 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-10 09:30:00 | 377.10 | 2026-02-10 09:35:00 | 378.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-11 10:45:00 | 385.95 | 2026-02-11 12:00:00 | 384.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-17 10:25:00 | 379.00 | 2026-02-17 10:30:00 | 378.09 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:00:00 | 380.95 | 2026-02-25 10:35:00 | 382.87 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-25 10:00:00 | 380.95 | 2026-02-25 13:25:00 | 381.65 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-02-26 09:55:00 | 385.35 | 2026-02-26 10:05:00 | 386.82 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-26 09:55:00 | 385.35 | 2026-02-26 13:55:00 | 388.75 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2026-03-11 11:15:00 | 339.70 | 2026-03-11 13:00:00 | 338.28 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-11 11:15:00 | 339.70 | 2026-03-11 15:20:00 | 334.95 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2026-03-13 09:35:00 | 317.90 | 2026-03-13 09:45:00 | 316.18 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-03-13 09:35:00 | 317.90 | 2026-03-13 15:10:00 | 314.35 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2026-03-18 09:45:00 | 327.15 | 2026-03-18 09:55:00 | 325.79 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-15 10:45:00 | 360.65 | 2026-04-15 10:50:00 | 359.32 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-20 09:30:00 | 352.80 | 2026-04-20 09:50:00 | 354.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-04-23 10:55:00 | 355.80 | 2026-04-23 12:50:00 | 354.07 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-23 10:55:00 | 355.80 | 2026-04-23 15:20:00 | 351.50 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2026-05-05 09:40:00 | 343.10 | 2026-05-05 10:00:00 | 341.98 | STOP_HIT | 1.00 | -0.33% |
