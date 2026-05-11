# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 381.00
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
| ENTRY1 | 90 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 20 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 129 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 70
- **Target hits / Stop hits / Partials:** 20 / 70 / 39
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 15.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 36 | 53.7% | 13 | 31 | 23 | 0.17% | 11.1% |
| BUY @ 2nd Alert (retest1) | 67 | 36 | 53.7% | 13 | 31 | 23 | 0.17% | 11.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 23 | 37.1% | 7 | 39 | 16 | 0.06% | 3.9% |
| SELL @ 2nd Alert (retest1) | 62 | 23 | 37.1% | 7 | 39 | 16 | 0.06% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 129 | 59 | 45.7% | 20 | 70 | 39 | 0.12% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 10:50:00 | 421.50 | 422.76 | 0.00 | ORB-short ORB[422.06,423.98] vol=2.0x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 12:05:00 | 420.34 | 422.06 | 0.00 | T1 1.5R @ 420.34 |
| Target hit | 2025-05-14 15:20:00 | 418.02 | 419.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2025-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:45:00 | 413.78 | 415.46 | 0.00 | ORB-short ORB[415.66,418.62] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-05-15 10:25:00 | 414.68 | 414.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:30:00 | 410.56 | 412.16 | 0.00 | ORB-short ORB[411.02,414.80] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-05-22 09:35:00 | 411.37 | 412.09 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 417.00 | 415.80 | 0.00 | ORB-long ORB[413.16,416.26] vol=2.2x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 11:30:00 | 418.16 | 416.72 | 0.00 | T1 1.5R @ 418.16 |
| Stop hit — per-position SL triggered | 2025-05-23 12:10:00 | 417.00 | 417.17 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 11:15:00 | 413.14 | 414.04 | 0.00 | ORB-short ORB[413.48,416.08] vol=3.1x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-05-28 12:30:00 | 413.94 | 413.73 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:30:00 | 412.98 | 415.16 | 0.00 | ORB-short ORB[414.56,417.40] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 12:15:00 | 411.57 | 414.04 | 0.00 | T1 1.5R @ 411.57 |
| Stop hit — per-position SL triggered | 2025-05-29 12:55:00 | 412.98 | 413.53 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 410.16 | 412.42 | 0.00 | ORB-short ORB[411.44,415.16] vol=1.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-06-03 09:35:00 | 411.00 | 412.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 407.40 | 408.17 | 0.00 | ORB-short ORB[407.42,408.78] vol=2.3x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-06-06 10:20:00 | 408.42 | 408.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 424.22 | 421.51 | 0.00 | ORB-long ORB[418.00,424.20] vol=1.7x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:40:00 | 426.15 | 422.56 | 0.00 | T1 1.5R @ 426.15 |
| Target hit | 2025-06-09 15:20:00 | 427.86 | 425.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:35:00 | 430.16 | 429.20 | 0.00 | ORB-long ORB[427.64,430.02] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 09:55:00 | 431.37 | 429.71 | 0.00 | T1 1.5R @ 431.37 |
| Target hit | 2025-06-10 13:55:00 | 431.52 | 431.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2025-06-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 11:00:00 | 430.42 | 428.56 | 0.00 | ORB-long ORB[427.08,430.04] vol=3.2x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:05:00 | 431.40 | 429.13 | 0.00 | T1 1.5R @ 431.40 |
| Stop hit — per-position SL triggered | 2025-06-11 12:30:00 | 430.42 | 430.13 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:50:00 | 427.46 | 428.31 | 0.00 | ORB-short ORB[428.04,429.76] vol=2.2x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:00:00 | 426.50 | 427.54 | 0.00 | T1 1.5R @ 426.50 |
| Target hit | 2025-06-12 15:20:00 | 424.96 | 426.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:30:00 | 424.84 | 423.08 | 0.00 | ORB-long ORB[417.76,421.98] vol=1.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 11:05:00 | 426.30 | 423.61 | 0.00 | T1 1.5R @ 426.30 |
| Target hit | 2025-06-16 15:20:00 | 428.42 | 426.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 11:15:00 | 430.48 | 429.54 | 0.00 | ORB-long ORB[426.62,430.00] vol=2.3x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-06-19 11:20:00 | 429.80 | 429.56 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:55:00 | 432.74 | 431.54 | 0.00 | ORB-long ORB[430.84,432.60] vol=2.0x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 12:10:00 | 433.80 | 432.17 | 0.00 | T1 1.5R @ 433.80 |
| Target hit | 2025-06-23 15:20:00 | 436.60 | 434.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 435.34 | 434.14 | 0.00 | ORB-long ORB[432.12,433.96] vol=2.9x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:30:00 | 436.61 | 434.46 | 0.00 | T1 1.5R @ 436.61 |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 435.34 | 434.78 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 10:50:00 | 424.40 | 425.55 | 0.00 | ORB-short ORB[424.64,427.46] vol=2.0x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:30:00 | 423.45 | 425.29 | 0.00 | T1 1.5R @ 423.45 |
| Target hit | 2025-07-04 12:45:00 | 424.20 | 424.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2025-07-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:35:00 | 427.22 | 426.15 | 0.00 | ORB-long ORB[424.36,426.40] vol=1.8x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:00:00 | 428.27 | 426.48 | 0.00 | T1 1.5R @ 428.27 |
| Stop hit — per-position SL triggered | 2025-07-07 11:10:00 | 427.22 | 426.53 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:45:00 | 447.56 | 446.82 | 0.00 | ORB-long ORB[445.24,447.00] vol=2.5x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-07-10 10:45:00 | 446.84 | 447.05 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:10:00 | 445.02 | 443.63 | 0.00 | ORB-long ORB[440.80,443.38] vol=1.6x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-07-15 11:25:00 | 444.21 | 443.72 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:00:00 | 433.16 | 434.45 | 0.00 | ORB-short ORB[434.74,437.58] vol=2.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-07-16 11:05:00 | 434.06 | 434.43 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:10:00 | 434.62 | 436.22 | 0.00 | ORB-short ORB[435.00,437.78] vol=2.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-07-17 12:25:00 | 435.37 | 435.22 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:10:00 | 426.92 | 429.09 | 0.00 | ORB-short ORB[429.56,433.60] vol=2.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-07-18 11:40:00 | 427.92 | 428.77 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:50:00 | 430.98 | 429.96 | 0.00 | ORB-long ORB[426.04,429.40] vol=1.7x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 11:30:00 | 432.44 | 430.43 | 0.00 | T1 1.5R @ 432.44 |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 430.98 | 430.81 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:45:00 | 432.02 | 432.43 | 0.00 | ORB-short ORB[432.04,434.20] vol=4.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-07-23 09:50:00 | 432.82 | 432.50 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:40:00 | 395.66 | 394.82 | 0.00 | ORB-long ORB[393.20,395.64] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-07-30 09:45:00 | 394.79 | 394.80 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:55:00 | 388.20 | 389.10 | 0.00 | ORB-short ORB[388.24,390.96] vol=5.2x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-07-31 10:35:00 | 389.03 | 388.79 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:45:00 | 394.64 | 396.81 | 0.00 | ORB-short ORB[397.50,399.90] vol=1.9x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 11:40:00 | 393.60 | 395.64 | 0.00 | T1 1.5R @ 393.60 |
| Target hit | 2025-08-08 15:20:00 | 389.90 | 392.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-08-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:45:00 | 395.14 | 394.74 | 0.00 | ORB-long ORB[392.44,394.18] vol=6.0x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:55:00 | 395.95 | 394.95 | 0.00 | T1 1.5R @ 395.95 |
| Target hit | 2025-08-13 12:00:00 | 396.08 | 396.75 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — BUY (started 2025-08-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:55:00 | 403.24 | 401.84 | 0.00 | ORB-long ORB[399.42,402.46] vol=1.7x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-08-18 11:05:00 | 402.47 | 401.96 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:35:00 | 403.74 | 401.97 | 0.00 | ORB-long ORB[398.50,401.82] vol=4.1x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:55:00 | 405.08 | 402.92 | 0.00 | T1 1.5R @ 405.08 |
| Target hit | 2025-08-19 15:20:00 | 405.94 | 404.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-08-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:40:00 | 396.00 | 397.21 | 0.00 | ORB-short ORB[396.64,399.76] vol=1.6x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-08-25 10:45:00 | 396.70 | 397.15 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:55:00 | 393.24 | 394.23 | 0.00 | ORB-short ORB[394.48,396.78] vol=3.2x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 393.77 | 394.15 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:55:00 | 394.36 | 393.20 | 0.00 | ORB-long ORB[391.26,393.18] vol=1.5x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 393.76 | 393.35 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:50:00 | 388.82 | 390.03 | 0.00 | ORB-short ORB[389.88,391.46] vol=2.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 12:35:00 | 387.81 | 388.86 | 0.00 | T1 1.5R @ 387.81 |
| Stop hit — per-position SL triggered | 2025-09-05 12:45:00 | 388.82 | 388.81 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:05:00 | 393.46 | 394.22 | 0.00 | ORB-short ORB[393.60,396.32] vol=4.3x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-09-15 11:10:00 | 393.99 | 394.13 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:15:00 | 400.26 | 398.56 | 0.00 | ORB-long ORB[394.64,398.20] vol=2.4x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-09-16 10:25:00 | 399.60 | 398.76 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 407.60 | 406.31 | 0.00 | ORB-long ORB[404.06,407.06] vol=1.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 09:55:00 | 408.84 | 406.95 | 0.00 | T1 1.5R @ 408.84 |
| Stop hit — per-position SL triggered | 2025-09-17 10:10:00 | 407.60 | 407.11 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:30:00 | 407.42 | 409.21 | 0.00 | ORB-short ORB[408.74,410.92] vol=1.5x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:50:00 | 406.44 | 408.66 | 0.00 | T1 1.5R @ 406.44 |
| Target hit | 2025-09-19 15:20:00 | 406.26 | 406.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2025-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:55:00 | 409.60 | 408.60 | 0.00 | ORB-long ORB[407.28,409.58] vol=9.8x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-09-24 11:00:00 | 408.93 | 408.64 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:55:00 | 404.00 | 405.42 | 0.00 | ORB-short ORB[404.68,407.50] vol=1.9x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-09-25 11:20:00 | 404.72 | 405.17 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:40:00 | 398.66 | 400.32 | 0.00 | ORB-short ORB[399.84,403.84] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-09-26 12:10:00 | 399.36 | 399.50 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:30:00 | 406.04 | 403.94 | 0.00 | ORB-long ORB[397.20,400.50] vol=2.5x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-10-01 10:35:00 | 404.77 | 403.98 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:15:00 | 419.72 | 416.80 | 0.00 | ORB-long ORB[412.40,417.32] vol=1.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-10-03 11:00:00 | 418.50 | 417.80 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:55:00 | 423.72 | 424.83 | 0.00 | ORB-short ORB[423.80,426.44] vol=2.6x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 12:30:00 | 422.63 | 424.35 | 0.00 | T1 1.5R @ 422.63 |
| Stop hit — per-position SL triggered | 2025-10-08 12:40:00 | 423.72 | 424.33 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:50:00 | 420.66 | 421.81 | 0.00 | ORB-short ORB[421.62,425.64] vol=4.1x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-10-09 10:00:00 | 421.56 | 421.69 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:35:00 | 429.72 | 429.14 | 0.00 | ORB-long ORB[427.20,429.60] vol=11.9x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 14:35:00 | 430.96 | 429.52 | 0.00 | T1 1.5R @ 430.96 |
| Target hit | 2025-10-13 15:20:00 | 430.56 | 429.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 09:35:00 | 430.38 | 431.24 | 0.00 | ORB-short ORB[431.00,433.18] vol=3.4x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-10-15 09:45:00 | 431.30 | 431.19 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:05:00 | 427.00 | 427.98 | 0.00 | ORB-short ORB[427.60,430.60] vol=2.3x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-10-30 11:20:00 | 427.66 | 427.90 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:10:00 | 421.66 | 422.81 | 0.00 | ORB-short ORB[421.80,424.22] vol=1.5x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:20:00 | 420.44 | 422.26 | 0.00 | T1 1.5R @ 420.44 |
| Stop hit — per-position SL triggered | 2025-11-04 10:30:00 | 421.66 | 422.22 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:10:00 | 415.72 | 416.93 | 0.00 | ORB-short ORB[416.32,420.60] vol=2.0x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-11-11 11:10:00 | 416.55 | 416.52 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 416.04 | 414.75 | 0.00 | ORB-long ORB[412.60,415.40] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 415.25 | 414.87 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:35:00 | 421.54 | 420.28 | 0.00 | ORB-long ORB[418.48,420.72] vol=3.1x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:50:00 | 423.15 | 420.89 | 0.00 | T1 1.5R @ 423.15 |
| Stop hit — per-position SL triggered | 2025-11-17 10:10:00 | 421.54 | 421.52 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:50:00 | 418.64 | 419.29 | 0.00 | ORB-short ORB[419.38,421.56] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-11-27 10:25:00 | 419.43 | 419.07 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:05:00 | 425.40 | 423.26 | 0.00 | ORB-long ORB[420.44,423.80] vol=3.3x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-11-28 10:20:00 | 424.45 | 423.40 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:30:00 | 428.68 | 427.38 | 0.00 | ORB-long ORB[425.62,427.78] vol=2.0x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 09:35:00 | 429.84 | 428.42 | 0.00 | T1 1.5R @ 429.84 |
| Target hit | 2025-12-01 12:05:00 | 429.64 | 430.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 57 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:15:00 | 424.82 | 425.63 | 0.00 | ORB-short ORB[425.04,427.74] vol=2.4x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-12-03 10:45:00 | 425.58 | 425.27 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 11:10:00 | 426.88 | 427.72 | 0.00 | ORB-short ORB[427.04,429.36] vol=8.3x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:45:00 | 425.88 | 427.55 | 0.00 | T1 1.5R @ 425.88 |
| Stop hit — per-position SL triggered | 2025-12-04 14:50:00 | 426.88 | 426.66 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:10:00 | 430.84 | 428.39 | 0.00 | ORB-long ORB[423.82,427.42] vol=3.2x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:15:00 | 432.12 | 428.75 | 0.00 | T1 1.5R @ 432.12 |
| Target hit | 2025-12-11 14:15:00 | 437.22 | 437.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 11:15:00 | 437.20 | 435.69 | 0.00 | ORB-long ORB[434.68,437.06] vol=2.4x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 12:15:00 | 438.26 | 436.23 | 0.00 | T1 1.5R @ 438.26 |
| Stop hit — per-position SL triggered | 2025-12-16 12:30:00 | 437.20 | 436.40 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:55:00 | 435.10 | 436.00 | 0.00 | ORB-short ORB[435.54,438.64] vol=2.2x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 12:00:00 | 434.05 | 435.63 | 0.00 | T1 1.5R @ 434.05 |
| Stop hit — per-position SL triggered | 2025-12-17 14:50:00 | 435.10 | 434.63 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 09:50:00 | 433.24 | 433.94 | 0.00 | ORB-short ORB[433.60,435.50] vol=1.6x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-12-19 10:00:00 | 434.00 | 433.81 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:05:00 | 433.00 | 432.03 | 0.00 | ORB-long ORB[429.94,431.76] vol=2.5x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:00:00 | 434.03 | 432.47 | 0.00 | T1 1.5R @ 434.03 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 433.00 | 432.51 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 435.14 | 434.49 | 0.00 | ORB-long ORB[432.36,435.12] vol=3.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-12-24 11:05:00 | 434.53 | 434.58 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:15:00 | 432.04 | 433.47 | 0.00 | ORB-short ORB[433.00,436.00] vol=3.0x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-12-29 11:50:00 | 432.68 | 433.28 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:10:00 | 429.28 | 429.73 | 0.00 | ORB-short ORB[429.42,432.20] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-12-30 10:25:00 | 429.92 | 429.66 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:35:00 | 438.66 | 441.37 | 0.00 | ORB-short ORB[439.40,445.60] vol=1.8x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:05:00 | 437.02 | 440.58 | 0.00 | T1 1.5R @ 437.02 |
| Target hit | 2026-01-06 15:20:00 | 429.34 | 433.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 425.00 | 425.97 | 0.00 | ORB-short ORB[425.16,428.60] vol=1.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 424.00 | 425.66 | 0.00 | T1 1.5R @ 424.00 |
| Stop hit — per-position SL triggered | 2026-01-08 11:50:00 | 425.00 | 425.52 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:40:00 | 428.50 | 426.49 | 0.00 | ORB-long ORB[422.80,428.00] vol=1.9x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-01-22 09:50:00 | 427.15 | 427.12 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:20:00 | 426.60 | 427.05 | 0.00 | ORB-short ORB[427.00,429.80] vol=3.4x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-01-23 10:35:00 | 427.70 | 426.83 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 10:25:00 | 410.20 | 409.31 | 0.00 | ORB-long ORB[407.00,410.10] vol=2.7x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-01-29 10:35:00 | 409.14 | 409.32 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 413.20 | 410.78 | 0.00 | ORB-long ORB[407.05,410.10] vol=5.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-02-01 11:05:00 | 412.06 | 410.98 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:10:00 | 410.40 | 411.36 | 0.00 | ORB-short ORB[410.75,413.95] vol=1.5x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 10:30:00 | 408.95 | 411.06 | 0.00 | T1 1.5R @ 408.95 |
| Target hit | 2026-02-05 15:00:00 | 408.45 | 408.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — BUY (started 2026-02-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:10:00 | 417.05 | 412.83 | 0.00 | ORB-long ORB[410.35,414.50] vol=3.5x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 11:20:00 | 418.81 | 413.40 | 0.00 | T1 1.5R @ 418.81 |
| Target hit | 2026-02-06 15:20:00 | 422.35 | 418.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 432.40 | 430.60 | 0.00 | ORB-long ORB[429.10,431.50] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:45:00 | 433.81 | 431.48 | 0.00 | T1 1.5R @ 433.81 |
| Stop hit — per-position SL triggered | 2026-02-10 12:55:00 | 432.40 | 431.96 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:50:00 | 426.75 | 428.58 | 0.00 | ORB-short ORB[427.45,431.05] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-02-12 10:55:00 | 427.65 | 428.46 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:50:00 | 421.65 | 422.39 | 0.00 | ORB-short ORB[421.70,425.50] vol=2.1x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 422.47 | 422.30 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:05:00 | 418.40 | 417.32 | 0.00 | ORB-long ORB[415.00,417.75] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:15:00 | 419.80 | 417.60 | 0.00 | T1 1.5R @ 419.80 |
| Target hit | 2026-02-20 15:20:00 | 421.30 | 420.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-02-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:45:00 | 426.45 | 426.18 | 0.00 | ORB-long ORB[420.35,426.40] vol=2.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:00:00 | 427.81 | 426.25 | 0.00 | T1 1.5R @ 427.81 |
| Target hit | 2026-02-23 15:20:00 | 430.70 | 429.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 369.05 | 370.87 | 0.00 | ORB-short ORB[370.40,373.60] vol=2.1x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:20:00 | 367.51 | 370.39 | 0.00 | T1 1.5R @ 367.51 |
| Stop hit — per-position SL triggered | 2026-03-13 11:45:00 | 369.05 | 369.96 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-03-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:55:00 | 373.55 | 371.89 | 0.00 | ORB-long ORB[368.60,372.30] vol=2.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 372.47 | 372.01 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:50:00 | 373.00 | 373.22 | 0.00 | ORB-short ORB[373.10,376.00] vol=2.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-03-18 11:20:00 | 374.01 | 373.14 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 363.90 | 364.56 | 0.00 | ORB-short ORB[364.60,368.85] vol=3.9x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-03-27 11:45:00 | 365.04 | 364.32 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 10:55:00 | 357.15 | 360.04 | 0.00 | ORB-short ORB[358.90,364.05] vol=2.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-04-01 11:20:00 | 358.39 | 359.73 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 376.50 | 375.78 | 0.00 | ORB-long ORB[372.60,375.35] vol=1.8x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:25:00 | 378.49 | 376.22 | 0.00 | T1 1.5R @ 378.49 |
| Target hit | 2026-04-10 12:35:00 | 377.15 | 377.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 86 — BUY (started 2026-04-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:45:00 | 383.90 | 382.00 | 0.00 | ORB-long ORB[377.55,380.60] vol=1.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-04-29 11:10:00 | 382.95 | 382.36 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:35:00 | 380.85 | 378.48 | 0.00 | ORB-long ORB[375.30,379.15] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-04-30 11:10:00 | 379.70 | 379.08 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 374.70 | 372.18 | 0.00 | ORB-long ORB[369.20,371.65] vol=6.3x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-05-05 11:40:00 | 373.61 | 372.60 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:55:00 | 372.30 | 373.14 | 0.00 | ORB-short ORB[372.60,375.10] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:00:00 | 371.35 | 373.06 | 0.00 | T1 1.5R @ 371.35 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 372.30 | 372.93 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 379.40 | 377.39 | 0.00 | ORB-long ORB[374.30,377.60] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-05-07 11:40:00 | 378.31 | 377.61 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 10:50:00 | 421.50 | 2025-05-14 12:05:00 | 420.34 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-05-14 10:50:00 | 421.50 | 2025-05-14 15:20:00 | 418.02 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-05-15 09:45:00 | 413.78 | 2025-05-15 10:25:00 | 414.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-22 09:30:00 | 410.56 | 2025-05-22 09:35:00 | 411.37 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-05-23 10:45:00 | 417.00 | 2025-05-23 11:30:00 | 418.16 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-05-23 10:45:00 | 417.00 | 2025-05-23 12:10:00 | 417.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-28 11:15:00 | 413.14 | 2025-05-28 12:30:00 | 413.94 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-05-29 10:30:00 | 412.98 | 2025-05-29 12:15:00 | 411.57 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-05-29 10:30:00 | 412.98 | 2025-05-29 12:55:00 | 412.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 09:30:00 | 410.16 | 2025-06-03 09:35:00 | 411.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-06 10:15:00 | 407.40 | 2025-06-06 10:20:00 | 408.42 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-09 09:30:00 | 424.22 | 2025-06-09 09:40:00 | 426.15 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-09 09:30:00 | 424.22 | 2025-06-09 15:20:00 | 427.86 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2025-06-10 09:35:00 | 430.16 | 2025-06-10 09:55:00 | 431.37 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-06-10 09:35:00 | 430.16 | 2025-06-10 13:55:00 | 431.52 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-06-11 11:00:00 | 430.42 | 2025-06-11 11:05:00 | 431.40 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-06-11 11:00:00 | 430.42 | 2025-06-11 12:30:00 | 430.42 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 10:50:00 | 427.46 | 2025-06-12 13:00:00 | 426.50 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-06-12 10:50:00 | 427.46 | 2025-06-12 15:20:00 | 424.96 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-06-16 10:30:00 | 424.84 | 2025-06-16 11:05:00 | 426.30 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-06-16 10:30:00 | 424.84 | 2025-06-16 15:20:00 | 428.42 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2025-06-19 11:15:00 | 430.48 | 2025-06-19 11:20:00 | 429.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-06-23 10:55:00 | 432.74 | 2025-06-23 12:10:00 | 433.80 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-06-23 10:55:00 | 432.74 | 2025-06-23 15:20:00 | 436.60 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-07-01 10:15:00 | 435.34 | 2025-07-01 10:30:00 | 436.61 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-07-01 10:15:00 | 435.34 | 2025-07-01 11:15:00 | 435.34 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-04 10:50:00 | 424.40 | 2025-07-04 11:30:00 | 423.45 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-04 10:50:00 | 424.40 | 2025-07-04 12:45:00 | 424.20 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-07-07 10:35:00 | 427.22 | 2025-07-07 11:00:00 | 428.27 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-07-07 10:35:00 | 427.22 | 2025-07-07 11:10:00 | 427.22 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-10 09:45:00 | 447.56 | 2025-07-10 10:45:00 | 446.84 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-15 11:10:00 | 445.02 | 2025-07-15 11:25:00 | 444.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-16 11:00:00 | 433.16 | 2025-07-16 11:05:00 | 434.06 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-17 11:10:00 | 434.62 | 2025-07-17 12:25:00 | 435.37 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-18 11:10:00 | 426.92 | 2025-07-18 11:40:00 | 427.92 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-21 10:50:00 | 430.98 | 2025-07-21 11:30:00 | 432.44 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-07-21 10:50:00 | 430.98 | 2025-07-21 12:15:00 | 430.98 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:45:00 | 432.02 | 2025-07-23 09:50:00 | 432.82 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-30 09:40:00 | 395.66 | 2025-07-30 09:45:00 | 394.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-31 09:55:00 | 388.20 | 2025-07-31 10:35:00 | 389.03 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-08 10:45:00 | 394.64 | 2025-08-08 11:40:00 | 393.60 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-08-08 10:45:00 | 394.64 | 2025-08-08 15:20:00 | 389.90 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-08-13 10:45:00 | 395.14 | 2025-08-13 10:55:00 | 395.95 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2025-08-13 10:45:00 | 395.14 | 2025-08-13 12:00:00 | 396.08 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-08-18 10:55:00 | 403.24 | 2025-08-18 11:05:00 | 402.47 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-19 10:35:00 | 403.74 | 2025-08-19 11:55:00 | 405.08 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-08-19 10:35:00 | 403.74 | 2025-08-19 15:20:00 | 405.94 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-08-25 10:40:00 | 396.00 | 2025-08-25 10:45:00 | 396.70 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-26 10:55:00 | 393.24 | 2025-08-26 11:15:00 | 393.77 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-01 10:55:00 | 394.36 | 2025-09-01 11:15:00 | 393.76 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-05 10:50:00 | 388.82 | 2025-09-05 12:35:00 | 387.81 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-09-05 10:50:00 | 388.82 | 2025-09-05 12:45:00 | 388.82 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-15 11:05:00 | 393.46 | 2025-09-15 11:10:00 | 393.99 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-09-16 10:15:00 | 400.26 | 2025-09-16 10:25:00 | 399.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-17 09:30:00 | 407.60 | 2025-09-17 09:55:00 | 408.84 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-17 09:30:00 | 407.60 | 2025-09-17 10:10:00 | 407.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-19 10:30:00 | 407.42 | 2025-09-19 11:50:00 | 406.44 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-09-19 10:30:00 | 407.42 | 2025-09-19 15:20:00 | 406.26 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-24 10:55:00 | 409.60 | 2025-09-24 11:00:00 | 408.93 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-09-25 10:55:00 | 404.00 | 2025-09-25 11:20:00 | 404.72 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-26 10:40:00 | 398.66 | 2025-09-26 12:10:00 | 399.36 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-01 10:30:00 | 406.04 | 2025-10-01 10:35:00 | 404.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-03 10:15:00 | 419.72 | 2025-10-03 11:00:00 | 418.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-08 10:55:00 | 423.72 | 2025-10-08 12:30:00 | 422.63 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-10-08 10:55:00 | 423.72 | 2025-10-08 12:40:00 | 423.72 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-09 09:50:00 | 420.66 | 2025-10-09 10:00:00 | 421.56 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-13 10:35:00 | 429.72 | 2025-10-13 14:35:00 | 430.96 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-13 10:35:00 | 429.72 | 2025-10-13 15:20:00 | 430.56 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2025-10-15 09:35:00 | 430.38 | 2025-10-15 09:45:00 | 431.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-30 11:05:00 | 427.00 | 2025-10-30 11:20:00 | 427.66 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-04 10:10:00 | 421.66 | 2025-11-04 10:20:00 | 420.44 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-04 10:10:00 | 421.66 | 2025-11-04 10:30:00 | 421.66 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:10:00 | 415.72 | 2025-11-11 11:10:00 | 416.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-14 10:50:00 | 416.04 | 2025-11-14 11:15:00 | 415.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-17 09:35:00 | 421.54 | 2025-11-17 09:50:00 | 423.15 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-17 09:35:00 | 421.54 | 2025-11-17 10:10:00 | 421.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 09:50:00 | 418.64 | 2025-11-27 10:25:00 | 419.43 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-28 10:05:00 | 425.40 | 2025-11-28 10:20:00 | 424.45 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-01 09:30:00 | 428.68 | 2025-12-01 09:35:00 | 429.84 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-12-01 09:30:00 | 428.68 | 2025-12-01 12:05:00 | 429.64 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-03 10:15:00 | 424.82 | 2025-12-03 10:45:00 | 425.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-04 11:10:00 | 426.88 | 2025-12-04 11:45:00 | 425.88 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-04 11:10:00 | 426.88 | 2025-12-04 14:50:00 | 426.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:10:00 | 430.84 | 2025-12-11 10:15:00 | 432.12 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-11 10:10:00 | 430.84 | 2025-12-11 14:15:00 | 437.22 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2025-12-16 11:15:00 | 437.20 | 2025-12-16 12:15:00 | 438.26 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-16 11:15:00 | 437.20 | 2025-12-16 12:30:00 | 437.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-17 10:55:00 | 435.10 | 2025-12-17 12:00:00 | 434.05 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-17 10:55:00 | 435.10 | 2025-12-17 14:50:00 | 435.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 09:50:00 | 433.24 | 2025-12-19 10:00:00 | 434.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-23 10:05:00 | 433.00 | 2025-12-23 11:00:00 | 434.03 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-23 10:05:00 | 433.00 | 2025-12-23 11:15:00 | 433.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 10:55:00 | 435.14 | 2025-12-24 11:05:00 | 434.53 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-12-29 11:15:00 | 432.04 | 2025-12-29 11:50:00 | 432.68 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-30 10:10:00 | 429.28 | 2025-12-30 10:25:00 | 429.92 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-01-06 09:35:00 | 438.66 | 2026-01-06 10:05:00 | 437.02 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-06 09:35:00 | 438.66 | 2026-01-06 15:20:00 | 429.34 | TARGET_HIT | 0.50 | 2.12% |
| SELL | retest1 | 2026-01-08 11:00:00 | 425.00 | 2026-01-08 11:20:00 | 424.00 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-01-08 11:00:00 | 425.00 | 2026-01-08 11:50:00 | 425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-22 09:40:00 | 428.50 | 2026-01-22 09:50:00 | 427.15 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-23 10:20:00 | 426.60 | 2026-01-23 10:35:00 | 427.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-29 10:25:00 | 410.20 | 2026-01-29 10:35:00 | 409.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-01 11:00:00 | 413.20 | 2026-02-01 11:05:00 | 412.06 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-05 10:10:00 | 410.40 | 2026-02-05 10:30:00 | 408.95 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-05 10:10:00 | 410.40 | 2026-02-05 15:00:00 | 408.45 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-06 11:10:00 | 417.05 | 2026-02-06 11:20:00 | 418.81 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-06 11:10:00 | 417.05 | 2026-02-06 15:20:00 | 422.35 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2026-02-10 10:55:00 | 432.40 | 2026-02-10 11:45:00 | 433.81 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-10 10:55:00 | 432.40 | 2026-02-10 12:55:00 | 432.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 10:50:00 | 426.75 | 2026-02-12 10:55:00 | 427.65 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-17 10:50:00 | 421.65 | 2026-02-17 11:15:00 | 422.47 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-20 10:05:00 | 418.40 | 2026-02-20 10:15:00 | 419.80 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-20 10:05:00 | 418.40 | 2026-02-20 15:20:00 | 421.30 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-02-23 10:45:00 | 426.45 | 2026-02-23 11:00:00 | 427.81 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-23 10:45:00 | 426.45 | 2026-02-23 15:20:00 | 430.70 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2026-03-13 10:50:00 | 369.05 | 2026-03-13 11:20:00 | 367.51 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-13 10:50:00 | 369.05 | 2026-03-13 11:45:00 | 369.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:55:00 | 373.55 | 2026-03-17 11:05:00 | 372.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-18 10:50:00 | 373.00 | 2026-03-18 11:20:00 | 374.01 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-27 11:05:00 | 363.90 | 2026-03-27 11:45:00 | 365.04 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-01 10:55:00 | 357.15 | 2026-04-01 11:20:00 | 358.39 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-10 10:05:00 | 376.50 | 2026-04-10 10:25:00 | 378.49 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-10 10:05:00 | 376.50 | 2026-04-10 12:35:00 | 377.15 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-04-29 10:45:00 | 383.90 | 2026-04-29 11:10:00 | 382.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-30 10:35:00 | 380.85 | 2026-04-30 11:10:00 | 379.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-05 11:00:00 | 374.70 | 2026-05-05 11:40:00 | 373.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-06 10:55:00 | 372.30 | 2026-05-06 11:00:00 | 371.35 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-05-06 10:55:00 | 372.30 | 2026-05-06 11:05:00 | 372.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 11:10:00 | 379.40 | 2026-05-07 11:40:00 | 378.31 | STOP_HIT | 1.00 | -0.29% |
