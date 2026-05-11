# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 1044.50
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
| ENTRY1 | 95 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 16 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 79
- **Target hits / Stop hits / Partials:** 16 / 79 / 36
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 17.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 27 | 39.1% | 8 | 42 | 19 | 0.08% | 5.6% |
| BUY @ 2nd Alert (retest1) | 69 | 27 | 39.1% | 8 | 42 | 19 | 0.08% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 25 | 40.3% | 8 | 37 | 17 | 0.19% | 12.0% |
| SELL @ 2nd Alert (retest1) | 62 | 25 | 40.3% | 8 | 37 | 17 | 0.19% | 12.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 131 | 52 | 39.7% | 16 | 79 | 36 | 0.13% | 17.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:35:00 | 402.00 | 404.12 | 0.00 | ORB-short ORB[402.60,408.40] vol=1.5x ATR=1.32 |
| Stop hit — per-position SL triggered | 2023-05-19 10:20:00 | 403.32 | 402.87 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 11:15:00 | 412.90 | 410.22 | 0.00 | ORB-long ORB[406.90,409.70] vol=2.6x ATR=0.83 |
| Stop hit — per-position SL triggered | 2023-05-23 11:25:00 | 412.07 | 410.46 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:20:00 | 410.55 | 407.46 | 0.00 | ORB-long ORB[403.10,406.75] vol=1.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 10:50:00 | 412.40 | 408.60 | 0.00 | T1 1.5R @ 412.40 |
| Target hit | 2023-05-26 15:20:00 | 413.00 | 411.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2023-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:50:00 | 421.30 | 419.25 | 0.00 | ORB-long ORB[415.50,420.45] vol=2.0x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 10:15:00 | 423.24 | 420.37 | 0.00 | T1 1.5R @ 423.24 |
| Target hit | 2023-05-29 14:00:00 | 421.75 | 421.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2023-06-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 11:00:00 | 425.10 | 423.89 | 0.00 | ORB-long ORB[421.00,424.85] vol=2.5x ATR=0.95 |
| Stop hit — per-position SL triggered | 2023-06-08 11:10:00 | 424.15 | 423.95 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:45:00 | 418.90 | 416.64 | 0.00 | ORB-long ORB[414.00,418.00] vol=1.6x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:55:00 | 420.09 | 417.04 | 0.00 | T1 1.5R @ 420.09 |
| Stop hit — per-position SL triggered | 2023-06-12 12:20:00 | 418.90 | 417.92 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 09:35:00 | 425.90 | 428.08 | 0.00 | ORB-short ORB[426.50,430.00] vol=2.2x ATR=1.25 |
| Stop hit — per-position SL triggered | 2023-06-14 09:40:00 | 427.15 | 427.99 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:35:00 | 430.15 | 428.62 | 0.00 | ORB-long ORB[426.15,429.45] vol=2.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-06-16 09:55:00 | 429.00 | 429.26 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:50:00 | 427.10 | 425.34 | 0.00 | ORB-long ORB[424.00,426.45] vol=2.3x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:00:00 | 428.61 | 425.95 | 0.00 | T1 1.5R @ 428.61 |
| Target hit | 2023-06-20 14:05:00 | 429.70 | 429.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2023-06-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:30:00 | 410.90 | 413.06 | 0.00 | ORB-short ORB[411.60,416.95] vol=1.9x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-06-23 09:45:00 | 412.32 | 412.77 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:35:00 | 412.20 | 410.61 | 0.00 | ORB-long ORB[408.20,411.30] vol=2.4x ATR=1.06 |
| Stop hit — per-position SL triggered | 2023-06-26 10:05:00 | 411.14 | 411.02 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 11:15:00 | 422.05 | 419.49 | 0.00 | ORB-long ORB[414.65,419.85] vol=4.0x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-06-30 12:20:00 | 421.27 | 419.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-07-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 10:00:00 | 424.05 | 425.23 | 0.00 | ORB-short ORB[424.30,428.95] vol=1.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2023-07-04 10:10:00 | 425.18 | 425.19 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:40:00 | 421.60 | 424.52 | 0.00 | ORB-short ORB[422.00,427.40] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-07-05 10:45:00 | 422.69 | 424.30 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:55:00 | 420.40 | 418.95 | 0.00 | ORB-long ORB[417.15,419.55] vol=1.5x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-07-06 10:00:00 | 419.25 | 419.00 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 11:05:00 | 420.75 | 422.65 | 0.00 | ORB-short ORB[422.10,424.40] vol=3.1x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 11:25:00 | 419.24 | 422.10 | 0.00 | T1 1.5R @ 419.24 |
| Stop hit — per-position SL triggered | 2023-07-07 12:45:00 | 420.75 | 420.87 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:05:00 | 430.95 | 427.26 | 0.00 | ORB-long ORB[423.00,427.75] vol=4.1x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-07-10 10:15:00 | 429.68 | 428.19 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:55:00 | 436.80 | 434.72 | 0.00 | ORB-long ORB[430.20,436.60] vol=1.5x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 11:10:00 | 439.23 | 435.78 | 0.00 | T1 1.5R @ 439.23 |
| Stop hit — per-position SL triggered | 2023-07-13 11:50:00 | 436.80 | 436.02 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:30:00 | 444.80 | 442.99 | 0.00 | ORB-long ORB[439.60,444.40] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-07-14 09:35:00 | 443.29 | 443.17 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:40:00 | 451.35 | 447.59 | 0.00 | ORB-long ORB[443.40,447.80] vol=2.3x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-07-17 09:50:00 | 449.87 | 448.09 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:40:00 | 439.70 | 441.98 | 0.00 | ORB-short ORB[441.50,444.40] vol=1.8x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-07-19 09:50:00 | 440.66 | 441.80 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:55:00 | 442.10 | 440.38 | 0.00 | ORB-long ORB[438.25,442.00] vol=1.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-07-20 11:05:00 | 441.23 | 440.49 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:30:00 | 438.80 | 437.14 | 0.00 | ORB-long ORB[435.00,438.35] vol=2.4x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-25 09:50:00 | 440.39 | 438.22 | 0.00 | T1 1.5R @ 440.39 |
| Target hit | 2023-07-25 11:45:00 | 439.90 | 440.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2023-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 11:00:00 | 459.75 | 456.91 | 0.00 | ORB-long ORB[453.00,458.65] vol=3.0x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 11:10:00 | 461.72 | 457.66 | 0.00 | T1 1.5R @ 461.72 |
| Stop hit — per-position SL triggered | 2023-07-31 11:50:00 | 459.75 | 458.32 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 10:20:00 | 460.80 | 457.84 | 0.00 | ORB-long ORB[455.50,460.00] vol=1.7x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-08-02 10:25:00 | 459.47 | 457.93 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 09:30:00 | 446.75 | 449.30 | 0.00 | ORB-short ORB[448.30,451.75] vol=2.0x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-08-03 09:40:00 | 448.48 | 448.50 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:00:00 | 459.50 | 462.48 | 0.00 | ORB-short ORB[460.10,466.00] vol=1.5x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 10:25:00 | 456.24 | 461.44 | 0.00 | T1 1.5R @ 456.24 |
| Stop hit — per-position SL triggered | 2023-08-04 11:50:00 | 459.50 | 460.00 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:30:00 | 460.40 | 463.67 | 0.00 | ORB-short ORB[462.40,466.60] vol=1.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2023-08-08 10:40:00 | 461.65 | 463.54 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:25:00 | 452.75 | 455.50 | 0.00 | ORB-short ORB[453.65,458.15] vol=1.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-08-09 10:40:00 | 454.23 | 455.23 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:50:00 | 460.50 | 464.00 | 0.00 | ORB-short ORB[463.60,470.00] vol=2.1x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-08-11 10:00:00 | 461.98 | 463.63 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-16 09:30:00 | 441.35 | 443.73 | 0.00 | ORB-short ORB[442.00,448.00] vol=2.2x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 10:45:00 | 439.03 | 441.79 | 0.00 | T1 1.5R @ 439.03 |
| Stop hit — per-position SL triggered | 2023-08-16 11:50:00 | 441.35 | 441.45 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 441.00 | 442.71 | 0.00 | ORB-short ORB[441.55,446.15] vol=2.3x ATR=0.69 |
| Stop hit — per-position SL triggered | 2023-08-17 11:25:00 | 441.69 | 442.62 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 11:15:00 | 457.00 | 460.67 | 0.00 | ORB-short ORB[462.30,465.90] vol=1.6x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 12:00:00 | 455.44 | 459.77 | 0.00 | T1 1.5R @ 455.44 |
| Stop hit — per-position SL triggered | 2023-08-24 12:40:00 | 457.00 | 459.25 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 10:40:00 | 452.05 | 449.94 | 0.00 | ORB-long ORB[447.35,450.95] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 11:15:00 | 453.27 | 450.78 | 0.00 | T1 1.5R @ 453.27 |
| Target hit | 2023-08-29 15:20:00 | 456.25 | 454.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2023-09-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 11:00:00 | 476.50 | 478.80 | 0.00 | ORB-short ORB[478.85,484.90] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2023-09-06 12:15:00 | 477.47 | 477.88 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 09:40:00 | 469.90 | 472.02 | 0.00 | ORB-short ORB[470.15,475.45] vol=1.7x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-09-08 10:05:00 | 471.36 | 471.68 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 09:55:00 | 477.60 | 478.88 | 0.00 | ORB-short ORB[478.00,480.80] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-09-11 10:05:00 | 478.79 | 478.85 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:55:00 | 482.75 | 484.88 | 0.00 | ORB-short ORB[486.00,490.90] vol=2.0x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 10:10:00 | 479.95 | 484.46 | 0.00 | T1 1.5R @ 479.95 |
| Target hit | 2023-09-12 15:20:00 | 478.00 | 480.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2023-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:30:00 | 473.90 | 475.80 | 0.00 | ORB-short ORB[474.50,479.00] vol=2.1x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-09-22 09:35:00 | 475.26 | 475.70 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 11:05:00 | 473.55 | 475.64 | 0.00 | ORB-short ORB[474.50,478.80] vol=3.0x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 12:05:00 | 471.90 | 474.64 | 0.00 | T1 1.5R @ 471.90 |
| Stop hit — per-position SL triggered | 2023-09-25 12:20:00 | 473.55 | 474.25 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 11:10:00 | 472.10 | 471.34 | 0.00 | ORB-long ORB[466.20,471.00] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-09-26 11:50:00 | 471.11 | 471.49 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 11:15:00 | 470.25 | 472.75 | 0.00 | ORB-short ORB[471.30,476.45] vol=2.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-10-06 13:15:00 | 471.16 | 471.79 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 10:55:00 | 468.65 | 470.99 | 0.00 | ORB-short ORB[471.25,474.35] vol=2.0x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-10-10 11:05:00 | 469.66 | 470.84 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 11:00:00 | 485.70 | 481.71 | 0.00 | ORB-long ORB[477.50,482.00] vol=3.8x ATR=1.70 |
| Stop hit — per-position SL triggered | 2023-10-13 11:05:00 | 484.00 | 481.83 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:10:00 | 480.10 | 477.82 | 0.00 | ORB-long ORB[475.30,480.00] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2023-10-16 10:20:00 | 478.73 | 477.98 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-10-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 09:30:00 | 472.00 | 474.50 | 0.00 | ORB-short ORB[473.50,478.45] vol=3.3x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-10-19 09:45:00 | 473.73 | 474.18 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 11:00:00 | 468.00 | 469.73 | 0.00 | ORB-short ORB[470.00,473.95] vol=2.4x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 11:20:00 | 466.37 | 469.40 | 0.00 | T1 1.5R @ 466.37 |
| Target hit | 2023-10-23 15:20:00 | 455.70 | 463.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2023-11-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 10:50:00 | 460.35 | 459.94 | 0.00 | ORB-long ORB[456.00,459.95] vol=2.9x ATR=1.24 |
| Stop hit — per-position SL triggered | 2023-11-01 11:05:00 | 459.11 | 459.90 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 10:10:00 | 488.40 | 484.66 | 0.00 | ORB-long ORB[480.30,484.60] vol=1.6x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:15:00 | 490.43 | 485.48 | 0.00 | T1 1.5R @ 490.43 |
| Stop hit — per-position SL triggered | 2023-11-07 10:30:00 | 488.40 | 486.37 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:50:00 | 488.95 | 485.95 | 0.00 | ORB-long ORB[481.10,486.60] vol=4.2x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 09:55:00 | 491.07 | 487.15 | 0.00 | T1 1.5R @ 491.07 |
| Stop hit — per-position SL triggered | 2023-11-13 10:20:00 | 488.95 | 488.53 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 11:05:00 | 505.75 | 503.07 | 0.00 | ORB-long ORB[500.15,505.55] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-11-20 11:15:00 | 504.66 | 503.26 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:30:00 | 508.90 | 506.55 | 0.00 | ORB-long ORB[501.90,507.95] vol=3.0x ATR=1.49 |
| Stop hit — per-position SL triggered | 2023-11-21 09:35:00 | 507.41 | 506.75 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 10:45:00 | 492.30 | 495.67 | 0.00 | ORB-short ORB[497.60,503.00] vol=1.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-11-23 10:50:00 | 493.44 | 495.57 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 11:15:00 | 507.50 | 506.03 | 0.00 | ORB-long ORB[502.55,505.00] vol=1.8x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 11:25:00 | 509.00 | 506.27 | 0.00 | T1 1.5R @ 509.00 |
| Target hit | 2023-11-24 13:15:00 | 508.30 | 508.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2023-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:50:00 | 518.00 | 515.23 | 0.00 | ORB-long ORB[509.80,516.60] vol=2.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2023-11-28 09:55:00 | 516.53 | 515.40 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-11-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:20:00 | 523.00 | 520.29 | 0.00 | ORB-long ORB[518.15,521.00] vol=1.8x ATR=1.26 |
| Stop hit — per-position SL triggered | 2023-11-29 10:30:00 | 521.74 | 520.50 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 10:50:00 | 517.00 | 518.27 | 0.00 | ORB-short ORB[517.45,520.30] vol=2.0x ATR=1.29 |
| Stop hit — per-position SL triggered | 2023-12-01 13:25:00 | 518.29 | 517.56 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 09:40:00 | 526.00 | 523.59 | 0.00 | ORB-long ORB[519.35,524.75] vol=4.1x ATR=1.93 |
| Stop hit — per-position SL triggered | 2023-12-04 09:50:00 | 524.07 | 523.79 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:05:00 | 521.30 | 519.06 | 0.00 | ORB-long ORB[515.45,520.00] vol=3.2x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 10:30:00 | 523.50 | 520.07 | 0.00 | T1 1.5R @ 523.50 |
| Stop hit — per-position SL triggered | 2023-12-05 11:30:00 | 521.30 | 522.80 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-12-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 09:30:00 | 523.75 | 521.32 | 0.00 | ORB-long ORB[516.05,521.90] vol=2.1x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-12-08 09:50:00 | 522.31 | 522.00 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 10:35:00 | 532.70 | 527.35 | 0.00 | ORB-long ORB[524.00,529.20] vol=6.6x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 11:00:00 | 535.33 | 529.91 | 0.00 | T1 1.5R @ 535.33 |
| Stop hit — per-position SL triggered | 2023-12-12 13:55:00 | 532.70 | 532.46 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:40:00 | 567.90 | 565.51 | 0.00 | ORB-long ORB[559.00,567.35] vol=2.1x ATR=2.49 |
| Stop hit — per-position SL triggered | 2023-12-22 11:40:00 | 565.41 | 566.91 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:35:00 | 587.80 | 584.08 | 0.00 | ORB-long ORB[580.30,585.35] vol=1.8x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 09:50:00 | 590.25 | 586.18 | 0.00 | T1 1.5R @ 590.25 |
| Target hit | 2023-12-27 15:20:00 | 605.95 | 598.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2023-12-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:25:00 | 613.85 | 610.05 | 0.00 | ORB-long ORB[608.00,612.55] vol=1.8x ATR=2.11 |
| Stop hit — per-position SL triggered | 2023-12-28 10:55:00 | 611.74 | 610.33 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:00:00 | 618.55 | 614.92 | 0.00 | ORB-long ORB[611.55,616.00] vol=1.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-01-01 10:10:00 | 616.77 | 615.40 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 601.90 | 605.54 | 0.00 | ORB-short ORB[603.40,611.40] vol=2.5x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 09:40:00 | 598.49 | 603.38 | 0.00 | T1 1.5R @ 598.49 |
| Target hit | 2024-01-03 15:20:00 | 593.45 | 598.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2024-01-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:50:00 | 584.50 | 587.90 | 0.00 | ORB-short ORB[586.00,592.50] vol=1.9x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 10:20:00 | 581.58 | 586.12 | 0.00 | T1 1.5R @ 581.58 |
| Target hit | 2024-01-08 15:20:00 | 577.50 | 581.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2024-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:30:00 | 580.00 | 582.36 | 0.00 | ORB-short ORB[581.05,586.00] vol=1.9x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 09:45:00 | 577.64 | 580.87 | 0.00 | T1 1.5R @ 577.64 |
| Target hit | 2024-01-15 14:50:00 | 576.30 | 576.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2024-01-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 10:25:00 | 560.75 | 557.88 | 0.00 | ORB-long ORB[555.50,559.35] vol=1.7x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-01-19 10:55:00 | 559.23 | 558.27 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 10:00:00 | 556.20 | 559.68 | 0.00 | ORB-short ORB[557.10,561.75] vol=1.7x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:20:00 | 553.80 | 558.59 | 0.00 | T1 1.5R @ 553.80 |
| Target hit | 2024-01-23 15:20:00 | 541.25 | 550.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — SELL (started 2024-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:10:00 | 576.45 | 578.64 | 0.00 | ORB-short ORB[576.80,580.30] vol=1.9x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 11:15:00 | 574.43 | 578.39 | 0.00 | T1 1.5R @ 574.43 |
| Stop hit — per-position SL triggered | 2024-02-01 11:45:00 | 576.45 | 577.77 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-02-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:40:00 | 579.20 | 576.40 | 0.00 | ORB-long ORB[573.10,577.70] vol=1.5x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 12:05:00 | 581.48 | 578.03 | 0.00 | T1 1.5R @ 581.48 |
| Stop hit — per-position SL triggered | 2024-02-02 12:10:00 | 579.20 | 578.07 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-02-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 10:45:00 | 596.30 | 591.42 | 0.00 | ORB-long ORB[588.20,594.30] vol=2.2x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-02-07 11:00:00 | 594.03 | 591.99 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-02-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 09:45:00 | 513.30 | 510.82 | 0.00 | ORB-long ORB[507.60,512.70] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-02-15 09:50:00 | 511.51 | 510.88 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 09:40:00 | 513.15 | 514.80 | 0.00 | ORB-short ORB[514.35,518.90] vol=2.1x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-02-19 10:05:00 | 514.64 | 514.08 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:05:00 | 509.70 | 514.37 | 0.00 | ORB-short ORB[516.00,519.75] vol=2.0x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 10:40:00 | 507.49 | 512.15 | 0.00 | T1 1.5R @ 507.49 |
| Target hit | 2024-02-26 15:20:00 | 505.20 | 507.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 09:55:00 | 507.90 | 506.48 | 0.00 | ORB-long ORB[504.90,507.15] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-02-27 10:00:00 | 506.86 | 506.73 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-02-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:50:00 | 500.00 | 503.59 | 0.00 | ORB-short ORB[501.45,507.20] vol=2.4x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-02-29 11:05:00 | 501.71 | 503.12 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-03-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:05:00 | 518.20 | 521.08 | 0.00 | ORB-short ORB[521.00,526.50] vol=1.7x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-03-15 11:15:00 | 519.89 | 521.01 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:45:00 | 543.25 | 538.97 | 0.00 | ORB-long ORB[534.25,541.10] vol=2.2x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-03-21 09:55:00 | 540.67 | 539.73 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-03-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:55:00 | 545.70 | 542.67 | 0.00 | ORB-long ORB[538.10,543.50] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-03-22 11:35:00 | 544.30 | 543.15 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-03-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 09:50:00 | 558.00 | 558.88 | 0.00 | ORB-short ORB[558.65,562.95] vol=2.3x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 10:00:00 | 555.84 | 558.60 | 0.00 | T1 1.5R @ 555.84 |
| Stop hit — per-position SL triggered | 2024-03-28 10:15:00 | 558.00 | 558.15 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-04-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:55:00 | 568.75 | 570.23 | 0.00 | ORB-short ORB[569.20,575.00] vol=2.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-04-05 10:15:00 | 570.73 | 570.12 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-04-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:45:00 | 582.65 | 578.22 | 0.00 | ORB-long ORB[574.00,579.20] vol=3.0x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 09:55:00 | 585.24 | 581.03 | 0.00 | T1 1.5R @ 585.24 |
| Target hit | 2024-04-09 13:30:00 | 588.85 | 589.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 85 — BUY (started 2024-04-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:00:00 | 599.00 | 596.49 | 0.00 | ORB-long ORB[591.05,598.00] vol=1.6x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 10:20:00 | 602.24 | 597.44 | 0.00 | T1 1.5R @ 602.24 |
| Stop hit — per-position SL triggered | 2024-04-10 10:45:00 | 599.00 | 598.04 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:50:00 | 596.90 | 602.70 | 0.00 | ORB-short ORB[600.90,609.05] vol=2.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-04-12 09:55:00 | 599.09 | 602.64 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 10:15:00 | 620.75 | 614.50 | 0.00 | ORB-long ORB[606.80,615.45] vol=3.2x ATR=2.64 |
| Stop hit — per-position SL triggered | 2024-04-16 10:20:00 | 618.11 | 614.84 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-04-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 10:30:00 | 619.55 | 615.94 | 0.00 | ORB-long ORB[614.30,618.40] vol=2.1x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 11:05:00 | 622.40 | 617.25 | 0.00 | T1 1.5R @ 622.40 |
| Stop hit — per-position SL triggered | 2024-04-18 13:35:00 | 619.55 | 620.15 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 10:00:00 | 614.50 | 616.63 | 0.00 | ORB-short ORB[617.05,621.35] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-04-22 10:10:00 | 616.82 | 616.49 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2024-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 10:00:00 | 611.45 | 613.34 | 0.00 | ORB-short ORB[612.65,618.75] vol=2.0x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-04-23 10:10:00 | 613.20 | 613.22 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 616.70 | 612.47 | 0.00 | ORB-long ORB[608.70,615.40] vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 09:35:00 | 619.29 | 613.28 | 0.00 | T1 1.5R @ 619.29 |
| Stop hit — per-position SL triggered | 2024-04-24 09:40:00 | 616.70 | 613.56 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 09:55:00 | 646.35 | 650.45 | 0.00 | ORB-short ORB[649.80,655.40] vol=1.7x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 10:10:00 | 642.80 | 649.37 | 0.00 | T1 1.5R @ 642.80 |
| Stop hit — per-position SL triggered | 2024-04-29 10:45:00 | 646.35 | 647.96 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 10:55:00 | 645.95 | 649.51 | 0.00 | ORB-short ORB[649.70,653.70] vol=1.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 11:35:00 | 643.72 | 648.22 | 0.00 | T1 1.5R @ 643.72 |
| Stop hit — per-position SL triggered | 2024-04-30 12:20:00 | 645.95 | 647.50 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-05-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 09:35:00 | 645.40 | 648.01 | 0.00 | ORB-short ORB[645.55,655.05] vol=3.8x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-05-03 09:45:00 | 648.14 | 647.95 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:20:00 | 634.45 | 639.77 | 0.00 | ORB-short ORB[638.25,646.90] vol=1.6x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:50:00 | 631.09 | 638.39 | 0.00 | T1 1.5R @ 631.09 |
| Target hit | 2024-05-07 15:20:00 | 620.95 | 625.07 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 09:35:00 | 402.00 | 2023-05-19 10:20:00 | 403.32 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-05-23 11:15:00 | 412.90 | 2023-05-23 11:25:00 | 412.07 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-05-26 10:20:00 | 410.55 | 2023-05-26 10:50:00 | 412.40 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-05-26 10:20:00 | 410.55 | 2023-05-26 15:20:00 | 413.00 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2023-05-29 09:50:00 | 421.30 | 2023-05-29 10:15:00 | 423.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-05-29 09:50:00 | 421.30 | 2023-05-29 14:00:00 | 421.75 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2023-06-08 11:00:00 | 425.10 | 2023-06-08 11:10:00 | 424.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-12 10:45:00 | 418.90 | 2023-06-12 10:55:00 | 420.09 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-06-12 10:45:00 | 418.90 | 2023-06-12 12:20:00 | 418.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 09:35:00 | 425.90 | 2023-06-14 09:40:00 | 427.15 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-16 09:35:00 | 430.15 | 2023-06-16 09:55:00 | 429.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-06-20 09:50:00 | 427.10 | 2023-06-20 10:00:00 | 428.61 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-06-20 09:50:00 | 427.10 | 2023-06-20 14:05:00 | 429.70 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2023-06-23 09:30:00 | 410.90 | 2023-06-23 09:45:00 | 412.32 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-06-26 09:35:00 | 412.20 | 2023-06-26 10:05:00 | 411.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-30 11:15:00 | 422.05 | 2023-06-30 12:20:00 | 421.27 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-04 10:00:00 | 424.05 | 2023-07-04 10:10:00 | 425.18 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-07-05 10:40:00 | 421.60 | 2023-07-05 10:45:00 | 422.69 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-06 09:55:00 | 420.40 | 2023-07-06 10:00:00 | 419.25 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-07-07 11:05:00 | 420.75 | 2023-07-07 11:25:00 | 419.24 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-07-07 11:05:00 | 420.75 | 2023-07-07 12:45:00 | 420.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-10 10:05:00 | 430.95 | 2023-07-10 10:15:00 | 429.68 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-07-13 09:55:00 | 436.80 | 2023-07-13 11:10:00 | 439.23 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-07-13 09:55:00 | 436.80 | 2023-07-13 11:50:00 | 436.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 09:30:00 | 444.80 | 2023-07-14 09:35:00 | 443.29 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-07-17 09:40:00 | 451.35 | 2023-07-17 09:50:00 | 449.87 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-07-19 09:40:00 | 439.70 | 2023-07-19 09:50:00 | 440.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-20 10:55:00 | 442.10 | 2023-07-20 11:05:00 | 441.23 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-07-25 09:30:00 | 438.80 | 2023-07-25 09:50:00 | 440.39 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-07-25 09:30:00 | 438.80 | 2023-07-25 11:45:00 | 439.90 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2023-07-31 11:00:00 | 459.75 | 2023-07-31 11:10:00 | 461.72 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-07-31 11:00:00 | 459.75 | 2023-07-31 11:50:00 | 459.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-02 10:20:00 | 460.80 | 2023-08-02 10:25:00 | 459.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-08-03 09:30:00 | 446.75 | 2023-08-03 09:40:00 | 448.48 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-08-04 10:00:00 | 459.50 | 2023-08-04 10:25:00 | 456.24 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2023-08-04 10:00:00 | 459.50 | 2023-08-04 11:50:00 | 459.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-08 10:30:00 | 460.40 | 2023-08-08 10:40:00 | 461.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-09 10:25:00 | 452.75 | 2023-08-09 10:40:00 | 454.23 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-11 09:50:00 | 460.50 | 2023-08-11 10:00:00 | 461.98 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-08-16 09:30:00 | 441.35 | 2023-08-16 10:45:00 | 439.03 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-08-16 09:30:00 | 441.35 | 2023-08-16 11:50:00 | 441.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-17 11:15:00 | 441.00 | 2023-08-17 11:25:00 | 441.69 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-24 11:15:00 | 457.00 | 2023-08-24 12:00:00 | 455.44 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-08-24 11:15:00 | 457.00 | 2023-08-24 12:40:00 | 457.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-29 10:40:00 | 452.05 | 2023-08-29 11:15:00 | 453.27 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-08-29 10:40:00 | 452.05 | 2023-08-29 15:20:00 | 456.25 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2023-09-06 11:00:00 | 476.50 | 2023-09-06 12:15:00 | 477.47 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-08 09:40:00 | 469.90 | 2023-09-08 10:05:00 | 471.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-09-11 09:55:00 | 477.60 | 2023-09-11 10:05:00 | 478.79 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-09-12 09:55:00 | 482.75 | 2023-09-12 10:10:00 | 479.95 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-09-12 09:55:00 | 482.75 | 2023-09-12 15:20:00 | 478.00 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2023-09-22 09:30:00 | 473.90 | 2023-09-22 09:35:00 | 475.26 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-25 11:05:00 | 473.55 | 2023-09-25 12:05:00 | 471.90 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-09-25 11:05:00 | 473.55 | 2023-09-25 12:20:00 | 473.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-26 11:10:00 | 472.10 | 2023-09-26 11:50:00 | 471.11 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-06 11:15:00 | 470.25 | 2023-10-06 13:15:00 | 471.16 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-10 10:55:00 | 468.65 | 2023-10-10 11:05:00 | 469.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-10-13 11:00:00 | 485.70 | 2023-10-13 11:05:00 | 484.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-10-16 10:10:00 | 480.10 | 2023-10-16 10:20:00 | 478.73 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-10-19 09:30:00 | 472.00 | 2023-10-19 09:45:00 | 473.73 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-10-23 11:00:00 | 468.00 | 2023-10-23 11:20:00 | 466.37 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-10-23 11:00:00 | 468.00 | 2023-10-23 15:20:00 | 455.70 | TARGET_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2023-11-01 10:50:00 | 460.35 | 2023-11-01 11:05:00 | 459.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-11-07 10:10:00 | 488.40 | 2023-11-07 10:15:00 | 490.43 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-11-07 10:10:00 | 488.40 | 2023-11-07 10:30:00 | 488.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-13 09:50:00 | 488.95 | 2023-11-13 09:55:00 | 491.07 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-11-13 09:50:00 | 488.95 | 2023-11-13 10:20:00 | 488.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-20 11:05:00 | 505.75 | 2023-11-20 11:15:00 | 504.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-21 09:30:00 | 508.90 | 2023-11-21 09:35:00 | 507.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-23 10:45:00 | 492.30 | 2023-11-23 10:50:00 | 493.44 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-11-24 11:15:00 | 507.50 | 2023-11-24 11:25:00 | 509.00 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-24 11:15:00 | 507.50 | 2023-11-24 13:15:00 | 508.30 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2023-11-28 09:50:00 | 518.00 | 2023-11-28 09:55:00 | 516.53 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-29 10:20:00 | 523.00 | 2023-11-29 10:30:00 | 521.74 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-12-01 10:50:00 | 517.00 | 2023-12-01 13:25:00 | 518.29 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-12-04 09:40:00 | 526.00 | 2023-12-04 09:50:00 | 524.07 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-05 10:05:00 | 521.30 | 2023-12-05 10:30:00 | 523.50 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-12-05 10:05:00 | 521.30 | 2023-12-05 11:30:00 | 521.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-08 09:30:00 | 523.75 | 2023-12-08 09:50:00 | 522.31 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-12 10:35:00 | 532.70 | 2023-12-12 11:00:00 | 535.33 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-12 10:35:00 | 532.70 | 2023-12-12 13:55:00 | 532.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 09:40:00 | 567.90 | 2023-12-22 11:40:00 | 565.41 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-12-27 09:35:00 | 587.80 | 2023-12-27 09:50:00 | 590.25 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-12-27 09:35:00 | 587.80 | 2023-12-27 15:20:00 | 605.95 | TARGET_HIT | 0.50 | 3.09% |
| BUY | retest1 | 2023-12-28 10:25:00 | 613.85 | 2023-12-28 10:55:00 | 611.74 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-01-01 10:00:00 | 618.55 | 2024-01-01 10:10:00 | 616.77 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-03 09:30:00 | 601.90 | 2024-01-03 09:40:00 | 598.49 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-01-03 09:30:00 | 601.90 | 2024-01-03 15:20:00 | 593.45 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2024-01-08 09:50:00 | 584.50 | 2024-01-08 10:20:00 | 581.58 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-01-08 09:50:00 | 584.50 | 2024-01-08 15:20:00 | 577.50 | TARGET_HIT | 0.50 | 1.20% |
| SELL | retest1 | 2024-01-15 09:30:00 | 580.00 | 2024-01-15 09:45:00 | 577.64 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-01-15 09:30:00 | 580.00 | 2024-01-15 14:50:00 | 576.30 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2024-01-19 10:25:00 | 560.75 | 2024-01-19 10:55:00 | 559.23 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-23 10:00:00 | 556.20 | 2024-01-23 10:20:00 | 553.80 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-23 10:00:00 | 556.20 | 2024-01-23 15:20:00 | 541.25 | TARGET_HIT | 0.50 | 2.69% |
| SELL | retest1 | 2024-02-01 11:10:00 | 576.45 | 2024-02-01 11:15:00 | 574.43 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-02-01 11:10:00 | 576.45 | 2024-02-01 11:45:00 | 576.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 10:40:00 | 579.20 | 2024-02-02 12:05:00 | 581.48 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-02-02 10:40:00 | 579.20 | 2024-02-02 12:10:00 | 579.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-07 10:45:00 | 596.30 | 2024-02-07 11:00:00 | 594.03 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-02-15 09:45:00 | 513.30 | 2024-02-15 09:50:00 | 511.51 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-19 09:40:00 | 513.15 | 2024-02-19 10:05:00 | 514.64 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-26 10:05:00 | 509.70 | 2024-02-26 10:40:00 | 507.49 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-02-26 10:05:00 | 509.70 | 2024-02-26 15:20:00 | 505.20 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2024-02-27 09:55:00 | 507.90 | 2024-02-27 10:00:00 | 506.86 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-02-29 10:50:00 | 500.00 | 2024-02-29 11:05:00 | 501.71 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-15 11:05:00 | 518.20 | 2024-03-15 11:15:00 | 519.89 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-21 09:45:00 | 543.25 | 2024-03-21 09:55:00 | 540.67 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-03-22 10:55:00 | 545.70 | 2024-03-22 11:35:00 | 544.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-03-28 09:50:00 | 558.00 | 2024-03-28 10:00:00 | 555.84 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-03-28 09:50:00 | 558.00 | 2024-03-28 10:15:00 | 558.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-05 09:55:00 | 568.75 | 2024-04-05 10:15:00 | 570.73 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-04-09 09:45:00 | 582.65 | 2024-04-09 09:55:00 | 585.24 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-04-09 09:45:00 | 582.65 | 2024-04-09 13:30:00 | 588.85 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2024-04-10 10:00:00 | 599.00 | 2024-04-10 10:20:00 | 602.24 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-04-10 10:00:00 | 599.00 | 2024-04-10 10:45:00 | 599.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-12 09:50:00 | 596.90 | 2024-04-12 09:55:00 | 599.09 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-16 10:15:00 | 620.75 | 2024-04-16 10:20:00 | 618.11 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-04-18 10:30:00 | 619.55 | 2024-04-18 11:05:00 | 622.40 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-04-18 10:30:00 | 619.55 | 2024-04-18 13:35:00 | 619.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-22 10:00:00 | 614.50 | 2024-04-22 10:10:00 | 616.82 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-04-23 10:00:00 | 611.45 | 2024-04-23 10:10:00 | 613.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-24 09:30:00 | 616.70 | 2024-04-24 09:35:00 | 619.29 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-24 09:30:00 | 616.70 | 2024-04-24 09:40:00 | 616.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-29 09:55:00 | 646.35 | 2024-04-29 10:10:00 | 642.80 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-04-29 09:55:00 | 646.35 | 2024-04-29 10:45:00 | 646.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-30 10:55:00 | 645.95 | 2024-04-30 11:35:00 | 643.72 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-04-30 10:55:00 | 645.95 | 2024-04-30 12:20:00 | 645.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-03 09:35:00 | 645.40 | 2024-05-03 09:45:00 | 648.14 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-05-07 10:20:00 | 634.45 | 2024-05-07 10:50:00 | 631.09 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-05-07 10:20:00 | 634.45 | 2024-05-07 15:20:00 | 620.95 | TARGET_HIT | 0.50 | 2.13% |
