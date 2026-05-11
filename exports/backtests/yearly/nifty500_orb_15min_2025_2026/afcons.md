# Afcons Infrastructure Ltd. (AFCONS)

## Backtest Summary

- **Window:** 2025-06-10 09:15:00 → 2026-05-08 15:25:00 (16888 bars)
- **Last close:** 340.40
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
| ENTRY1 | 71 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 16 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 99 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 55
- **Target hits / Stop hits / Partials:** 16 / 55 / 28
- **Avg / median % per leg:** 0.27% / 0.00%
- **Sum % (uncompounded):** 27.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 15 | 40.5% | 5 | 22 | 10 | 0.43% | 16.1% |
| BUY @ 2nd Alert (retest1) | 37 | 15 | 40.5% | 5 | 22 | 10 | 0.43% | 16.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 29 | 46.8% | 11 | 33 | 18 | 0.18% | 11.0% |
| SELL @ 2nd Alert (retest1) | 62 | 29 | 46.8% | 11 | 33 | 18 | 0.18% | 11.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 99 | 44 | 44.4% | 16 | 55 | 28 | 0.27% | 27.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 432.65 | 436.38 | 0.00 | ORB-short ORB[435.20,441.50] vol=1.9x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-06-16 09:35:00 | 434.00 | 436.11 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 431.55 | 435.46 | 0.00 | ORB-short ORB[437.05,440.95] vol=1.5x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-07-01 11:45:00 | 432.59 | 434.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:30:00 | 429.35 | 431.47 | 0.00 | ORB-short ORB[430.15,435.35] vol=2.5x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-07-02 09:35:00 | 430.28 | 431.33 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-07-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:55:00 | 434.30 | 431.76 | 0.00 | ORB-long ORB[430.20,433.70] vol=2.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-07-03 11:25:00 | 433.33 | 432.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-07-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:10:00 | 427.95 | 425.14 | 0.00 | ORB-long ORB[421.25,426.30] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-07-09 11:20:00 | 426.96 | 425.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:15:00 | 422.45 | 424.45 | 0.00 | ORB-short ORB[424.15,427.40] vol=3.7x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:45:00 | 420.67 | 423.22 | 0.00 | T1 1.5R @ 420.67 |
| Target hit | 2025-07-11 12:55:00 | 422.40 | 421.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2025-07-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:55:00 | 422.85 | 423.80 | 0.00 | ORB-short ORB[423.20,425.70] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 12:05:00 | 421.64 | 423.19 | 0.00 | T1 1.5R @ 421.64 |
| Target hit | 2025-07-16 15:20:00 | 419.55 | 420.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:00:00 | 419.30 | 421.81 | 0.00 | ORB-short ORB[419.50,423.10] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 419.82 | 421.79 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:15:00 | 423.40 | 425.83 | 0.00 | ORB-short ORB[425.25,430.10] vol=2.3x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-07-23 11:30:00 | 424.55 | 425.67 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:35:00 | 401.70 | 400.56 | 0.00 | ORB-long ORB[397.80,401.00] vol=2.5x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:15:00 | 403.34 | 401.26 | 0.00 | T1 1.5R @ 403.34 |
| Target hit | 2025-08-01 14:00:00 | 403.55 | 404.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2025-08-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:40:00 | 403.90 | 404.50 | 0.00 | ORB-short ORB[404.50,406.85] vol=3.2x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 404.82 | 404.31 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:55:00 | 419.70 | 422.68 | 0.00 | ORB-short ORB[422.30,426.00] vol=3.0x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:00:00 | 417.82 | 421.80 | 0.00 | T1 1.5R @ 417.82 |
| Stop hit — per-position SL triggered | 2025-08-12 13:55:00 | 419.70 | 419.14 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-08-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:55:00 | 418.60 | 420.73 | 0.00 | ORB-short ORB[421.55,424.75] vol=1.8x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:10:00 | 416.32 | 420.14 | 0.00 | T1 1.5R @ 416.32 |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 418.60 | 420.08 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-08-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:30:00 | 430.80 | 427.15 | 0.00 | ORB-long ORB[424.20,428.25] vol=1.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-08-19 10:35:00 | 429.51 | 427.28 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:50:00 | 433.85 | 430.50 | 0.00 | ORB-long ORB[427.55,431.55] vol=4.4x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-08-21 10:05:00 | 432.38 | 431.63 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 431.65 | 429.01 | 0.00 | ORB-long ORB[425.05,430.70] vol=4.2x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-08-22 09:50:00 | 430.15 | 429.20 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:05:00 | 434.00 | 431.02 | 0.00 | ORB-long ORB[428.20,432.80] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-08-25 10:10:00 | 432.79 | 431.18 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:00:00 | 425.10 | 422.36 | 0.00 | ORB-long ORB[419.80,423.00] vol=6.1x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:20:00 | 426.44 | 422.89 | 0.00 | T1 1.5R @ 426.44 |
| Stop hit — per-position SL triggered | 2025-09-01 11:35:00 | 425.10 | 423.08 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:40:00 | 431.60 | 430.01 | 0.00 | ORB-long ORB[424.65,428.80] vol=17.1x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 09:45:00 | 433.95 | 430.26 | 0.00 | T1 1.5R @ 433.95 |
| Target hit | 2025-09-03 15:00:00 | 434.30 | 434.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2025-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:30:00 | 444.60 | 443.43 | 0.00 | ORB-long ORB[440.00,443.35] vol=6.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-09-10 09:35:00 | 443.18 | 443.28 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:30:00 | 452.00 | 449.76 | 0.00 | ORB-long ORB[446.30,449.80] vol=4.0x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 09:35:00 | 453.82 | 450.98 | 0.00 | T1 1.5R @ 453.82 |
| Stop hit — per-position SL triggered | 2025-09-11 09:45:00 | 452.00 | 451.65 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:20:00 | 450.35 | 452.14 | 0.00 | ORB-short ORB[450.55,455.90] vol=1.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:35:00 | 448.04 | 451.44 | 0.00 | T1 1.5R @ 448.04 |
| Target hit | 2025-09-12 15:05:00 | 449.65 | 449.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2025-09-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:50:00 | 452.00 | 449.49 | 0.00 | ORB-long ORB[444.00,449.20] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-09-15 11:45:00 | 450.60 | 450.01 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:25:00 | 458.30 | 456.16 | 0.00 | ORB-long ORB[451.65,455.80] vol=2.1x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-09-17 10:35:00 | 457.12 | 456.29 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:00:00 | 450.15 | 451.48 | 0.00 | ORB-short ORB[451.00,455.40] vol=3.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 451.24 | 451.39 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:10:00 | 453.05 | 455.61 | 0.00 | ORB-short ORB[456.00,460.00] vol=1.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-09-23 11:55:00 | 453.91 | 455.33 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:35:00 | 449.85 | 452.39 | 0.00 | ORB-short ORB[451.20,454.90] vol=2.5x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:55:00 | 448.04 | 451.42 | 0.00 | T1 1.5R @ 448.04 |
| Target hit | 2025-09-25 15:20:00 | 443.80 | 447.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-10-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 11:10:00 | 450.50 | 454.32 | 0.00 | ORB-short ORB[451.00,455.95] vol=1.7x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:45:00 | 448.72 | 453.55 | 0.00 | T1 1.5R @ 448.72 |
| Stop hit — per-position SL triggered | 2025-10-03 13:05:00 | 450.50 | 451.99 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:50:00 | 444.70 | 446.86 | 0.00 | ORB-short ORB[445.35,450.10] vol=1.5x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-10-06 09:55:00 | 445.99 | 446.77 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:45:00 | 459.55 | 461.17 | 0.00 | ORB-short ORB[462.40,465.40] vol=5.8x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 457.58 | 460.37 | 0.00 | T1 1.5R @ 457.58 |
| Target hit | 2025-10-14 15:20:00 | 453.80 | 455.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2025-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:50:00 | 458.30 | 456.25 | 0.00 | ORB-long ORB[451.00,455.70] vol=18.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-10-15 11:05:00 | 456.67 | 456.30 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:05:00 | 452.95 | 454.24 | 0.00 | ORB-short ORB[453.85,459.00] vol=2.0x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 12:10:00 | 451.28 | 454.07 | 0.00 | T1 1.5R @ 451.28 |
| Target hit | 2025-10-17 15:20:00 | 444.80 | 450.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:55:00 | 453.75 | 455.54 | 0.00 | ORB-short ORB[453.90,457.70] vol=1.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 12:25:00 | 451.81 | 454.91 | 0.00 | T1 1.5R @ 451.81 |
| Stop hit — per-position SL triggered | 2025-10-30 14:30:00 | 453.75 | 454.32 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-11-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:05:00 | 454.00 | 452.96 | 0.00 | ORB-long ORB[447.35,453.85] vol=8.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-11-03 10:20:00 | 452.71 | 453.01 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-11-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:45:00 | 448.45 | 449.70 | 0.00 | ORB-short ORB[449.15,451.80] vol=2.4x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:50:00 | 446.87 | 449.47 | 0.00 | T1 1.5R @ 446.87 |
| Target hit | 2025-11-04 15:20:00 | 444.85 | 445.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-11-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:50:00 | 440.10 | 441.25 | 0.00 | ORB-short ORB[440.80,445.40] vol=2.4x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:00:00 | 438.66 | 440.79 | 0.00 | T1 1.5R @ 438.66 |
| Stop hit — per-position SL triggered | 2025-11-10 10:45:00 | 440.10 | 440.55 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 430.50 | 433.21 | 0.00 | ORB-short ORB[432.30,437.10] vol=2.0x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 09:35:00 | 428.50 | 432.32 | 0.00 | T1 1.5R @ 428.50 |
| Target hit | 2025-11-11 15:20:00 | 417.20 | 417.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-11-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 11:05:00 | 410.10 | 411.70 | 0.00 | ORB-short ORB[410.65,415.40] vol=2.4x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-11-13 12:55:00 | 411.35 | 410.52 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:55:00 | 399.60 | 401.80 | 0.00 | ORB-short ORB[402.25,405.00] vol=2.3x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-11-18 10:00:00 | 400.47 | 401.69 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:45:00 | 405.00 | 403.67 | 0.00 | ORB-long ORB[400.10,404.95] vol=3.1x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-11-26 10:55:00 | 404.13 | 403.70 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:40:00 | 405.35 | 407.10 | 0.00 | ORB-short ORB[406.05,409.60] vol=1.7x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-11-27 09:45:00 | 406.13 | 407.04 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-12-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:25:00 | 396.65 | 397.40 | 0.00 | ORB-short ORB[398.00,400.75] vol=6.6x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:35:00 | 395.30 | 397.16 | 0.00 | T1 1.5R @ 395.30 |
| Target hit | 2025-12-08 13:40:00 | 394.85 | 394.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-12-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:05:00 | 389.25 | 385.32 | 0.00 | ORB-long ORB[383.45,387.35] vol=1.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:25:00 | 391.26 | 385.55 | 0.00 | T1 1.5R @ 391.26 |
| Target hit | 2025-12-09 15:20:00 | 399.10 | 394.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-12-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:50:00 | 404.00 | 401.86 | 0.00 | ORB-long ORB[400.65,403.60] vol=2.2x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-12-12 10:00:00 | 402.71 | 401.91 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-12-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:05:00 | 397.40 | 398.58 | 0.00 | ORB-short ORB[397.55,401.60] vol=6.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-15 11:50:00 | 398.20 | 398.51 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:40:00 | 395.65 | 396.85 | 0.00 | ORB-short ORB[396.50,399.00] vol=2.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-12-16 09:50:00 | 396.52 | 396.74 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 382.45 | 379.33 | 0.00 | ORB-long ORB[377.45,381.00] vol=3.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-12-19 09:50:00 | 381.19 | 380.02 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:15:00 | 390.10 | 389.67 | 0.00 | ORB-long ORB[386.50,389.40] vol=2.1x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-12-26 15:20:00 | 390.00 | 390.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:40:00 | 385.55 | 386.60 | 0.00 | ORB-short ORB[386.00,389.00] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:55:00 | 384.60 | 386.16 | 0.00 | T1 1.5R @ 384.60 |
| Stop hit — per-position SL triggered | 2025-12-29 11:40:00 | 385.55 | 385.22 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 389.15 | 389.36 | 0.00 | ORB-short ORB[390.20,392.90] vol=18.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 389.73 | 389.67 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2026-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:40:00 | 388.05 | 388.97 | 0.00 | ORB-short ORB[388.50,392.85] vol=1.9x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-01-02 10:35:00 | 389.42 | 389.64 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:25:00 | 382.10 | 384.61 | 0.00 | ORB-short ORB[384.60,389.90] vol=1.8x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-01-05 10:35:00 | 382.90 | 384.42 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:15:00 | 380.10 | 382.15 | 0.00 | ORB-short ORB[382.10,384.95] vol=1.8x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-01-06 11:35:00 | 380.63 | 381.94 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:55:00 | 378.20 | 375.92 | 0.00 | ORB-long ORB[374.75,377.40] vol=1.8x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 10:10:00 | 379.58 | 376.52 | 0.00 | T1 1.5R @ 379.58 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 378.20 | 376.56 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:05:00 | 375.85 | 378.21 | 0.00 | ORB-short ORB[377.05,380.10] vol=1.7x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 374.83 | 377.71 | 0.00 | T1 1.5R @ 374.83 |
| Target hit | 2026-01-08 14:05:00 | 373.30 | 372.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 56 — SELL (started 2026-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:35:00 | 348.95 | 350.55 | 0.00 | ORB-short ORB[349.30,353.70] vol=1.7x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-01-19 09:50:00 | 349.95 | 350.23 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-02-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:30:00 | 340.10 | 336.20 | 0.00 | ORB-long ORB[331.80,334.90] vol=2.3x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-02-01 10:35:00 | 339.03 | 336.74 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:15:00 | 332.00 | 333.57 | 0.00 | ORB-short ORB[333.65,336.85] vol=2.0x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 12:10:00 | 330.75 | 332.88 | 0.00 | T1 1.5R @ 330.75 |
| Stop hit — per-position SL triggered | 2026-02-06 14:00:00 | 332.00 | 332.61 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:05:00 | 338.25 | 335.47 | 0.00 | ORB-long ORB[332.20,335.95] vol=4.3x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-02-09 11:35:00 | 337.54 | 335.81 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 329.75 | 331.82 | 0.00 | ORB-short ORB[332.10,333.80] vol=4.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 330.93 | 331.74 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 323.15 | 323.98 | 0.00 | ORB-short ORB[323.80,325.75] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2026-02-18 10:30:00 | 323.73 | 323.96 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 321.35 | 322.87 | 0.00 | ORB-short ORB[322.60,325.20] vol=2.2x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:20:00 | 320.57 | 322.41 | 0.00 | T1 1.5R @ 320.57 |
| Target hit | 2026-02-19 15:20:00 | 320.05 | 321.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 316.40 | 317.58 | 0.00 | ORB-short ORB[317.00,320.50] vol=3.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-02-20 09:35:00 | 317.15 | 317.50 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:00:00 | 315.95 | 317.82 | 0.00 | ORB-short ORB[317.30,321.85] vol=1.8x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-23 10:05:00 | 316.94 | 317.72 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 289.45 | 292.00 | 0.00 | ORB-short ORB[292.50,296.85] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-27 10:55:00 | 290.27 | 291.94 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-03-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 10:00:00 | 272.00 | 274.20 | 0.00 | ORB-short ORB[273.15,276.95] vol=1.5x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-09 11:10:00 | 273.45 | 273.50 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 278.55 | 279.77 | 0.00 | ORB-short ORB[279.15,282.00] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-03-10 09:50:00 | 279.51 | 279.15 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:30:00 | 279.55 | 276.92 | 0.00 | ORB-long ORB[274.50,278.40] vol=1.5x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 281.44 | 279.66 | 0.00 | T1 1.5R @ 281.44 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 279.55 | 279.61 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 279.65 | 278.21 | 0.00 | ORB-long ORB[276.00,279.15] vol=5.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 281.13 | 278.54 | 0.00 | T1 1.5R @ 281.13 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 279.65 | 279.34 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 314.90 | 314.35 | 0.00 | ORB-long ORB[310.85,314.75] vol=4.6x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:50:00 | 316.58 | 314.59 | 0.00 | T1 1.5R @ 316.58 |
| Target hit | 2026-04-10 15:20:00 | 324.90 | 322.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 315.00 | 313.23 | 0.00 | ORB-long ORB[310.00,314.40] vol=2.1x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:50:00 | 317.43 | 313.82 | 0.00 | T1 1.5R @ 317.43 |
| Target hit | 2026-04-15 12:25:00 | 342.90 | 348.75 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-06-16 09:30:00 | 432.65 | 2025-06-16 09:35:00 | 434.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-01 10:55:00 | 431.55 | 2025-07-01 11:45:00 | 432.59 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-02 09:30:00 | 429.35 | 2025-07-02 09:35:00 | 430.28 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-03 10:55:00 | 434.30 | 2025-07-03 11:25:00 | 433.33 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-09 11:10:00 | 427.95 | 2025-07-09 11:20:00 | 426.96 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-11 10:15:00 | 422.45 | 2025-07-11 10:45:00 | 420.67 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-11 10:15:00 | 422.45 | 2025-07-11 12:55:00 | 422.40 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2025-07-16 10:55:00 | 422.85 | 2025-07-16 12:05:00 | 421.64 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-07-16 10:55:00 | 422.85 | 2025-07-16 15:20:00 | 419.55 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-07-17 11:00:00 | 419.30 | 2025-07-17 11:10:00 | 419.82 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-07-23 11:15:00 | 423.40 | 2025-07-23 11:30:00 | 424.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-01 09:35:00 | 401.70 | 2025-08-01 10:15:00 | 403.34 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-08-01 09:35:00 | 401.70 | 2025-08-01 14:00:00 | 403.55 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2025-08-06 09:40:00 | 403.90 | 2025-08-06 10:20:00 | 404.82 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-12 10:55:00 | 419.70 | 2025-08-12 11:00:00 | 417.82 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-12 10:55:00 | 419.70 | 2025-08-12 13:55:00 | 419.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 09:55:00 | 418.60 | 2025-08-14 10:10:00 | 416.32 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-08-14 09:55:00 | 418.60 | 2025-08-14 10:15:00 | 418.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 10:30:00 | 430.80 | 2025-08-19 10:35:00 | 429.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-21 09:50:00 | 433.85 | 2025-08-21 10:05:00 | 432.38 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-22 09:30:00 | 431.65 | 2025-08-22 09:50:00 | 430.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-25 10:05:00 | 434.00 | 2025-08-25 10:10:00 | 432.79 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-01 11:00:00 | 425.10 | 2025-09-01 11:20:00 | 426.44 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-09-01 11:00:00 | 425.10 | 2025-09-01 11:35:00 | 425.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 09:40:00 | 431.60 | 2025-09-03 09:45:00 | 433.95 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-09-03 09:40:00 | 431.60 | 2025-09-03 15:00:00 | 434.30 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-09-10 09:30:00 | 444.60 | 2025-09-10 09:35:00 | 443.18 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-11 09:30:00 | 452.00 | 2025-09-11 09:35:00 | 453.82 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-09-11 09:30:00 | 452.00 | 2025-09-11 09:45:00 | 452.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 10:20:00 | 450.35 | 2025-09-12 11:35:00 | 448.04 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-09-12 10:20:00 | 450.35 | 2025-09-12 15:05:00 | 449.65 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-09-15 10:50:00 | 452.00 | 2025-09-15 11:45:00 | 450.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-17 10:25:00 | 458.30 | 2025-09-17 10:35:00 | 457.12 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-19 11:00:00 | 450.15 | 2025-09-19 11:25:00 | 451.24 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-23 11:10:00 | 453.05 | 2025-09-23 11:55:00 | 453.91 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-25 10:35:00 | 449.85 | 2025-09-25 11:55:00 | 448.04 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-25 10:35:00 | 449.85 | 2025-09-25 15:20:00 | 443.80 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2025-10-03 11:10:00 | 450.50 | 2025-10-03 11:45:00 | 448.72 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-03 11:10:00 | 450.50 | 2025-10-03 13:05:00 | 450.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 09:50:00 | 444.70 | 2025-10-06 09:55:00 | 445.99 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-14 09:45:00 | 459.55 | 2025-10-14 10:15:00 | 457.58 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-14 09:45:00 | 459.55 | 2025-10-14 15:20:00 | 453.80 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-10-15 10:50:00 | 458.30 | 2025-10-15 11:05:00 | 456.67 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-17 11:05:00 | 452.95 | 2025-10-17 12:10:00 | 451.28 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-17 11:05:00 | 452.95 | 2025-10-17 15:20:00 | 444.80 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2025-10-30 09:55:00 | 453.75 | 2025-10-30 12:25:00 | 451.81 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-30 09:55:00 | 453.75 | 2025-10-30 14:30:00 | 453.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:05:00 | 454.00 | 2025-11-03 10:20:00 | 452.71 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-04 09:45:00 | 448.45 | 2025-11-04 09:50:00 | 446.87 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-04 09:45:00 | 448.45 | 2025-11-04 15:20:00 | 444.85 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2025-11-10 09:50:00 | 440.10 | 2025-11-10 10:00:00 | 438.66 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-10 09:50:00 | 440.10 | 2025-11-10 10:45:00 | 440.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 09:30:00 | 430.50 | 2025-11-11 09:35:00 | 428.50 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-11-11 09:30:00 | 430.50 | 2025-11-11 15:20:00 | 417.20 | TARGET_HIT | 0.50 | 3.09% |
| SELL | retest1 | 2025-11-13 11:05:00 | 410.10 | 2025-11-13 12:55:00 | 411.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-18 09:55:00 | 399.60 | 2025-11-18 10:00:00 | 400.47 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-26 10:45:00 | 405.00 | 2025-11-26 10:55:00 | 404.13 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-27 09:40:00 | 405.35 | 2025-11-27 09:45:00 | 406.13 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-08 10:25:00 | 396.65 | 2025-12-08 10:35:00 | 395.30 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-08 10:25:00 | 396.65 | 2025-12-08 13:40:00 | 394.85 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-09 11:05:00 | 389.25 | 2025-12-09 11:25:00 | 391.26 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-12-09 11:05:00 | 389.25 | 2025-12-09 15:20:00 | 399.10 | TARGET_HIT | 0.50 | 2.53% |
| BUY | retest1 | 2025-12-12 09:50:00 | 404.00 | 2025-12-12 10:00:00 | 402.71 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-15 11:05:00 | 397.40 | 2025-12-15 11:50:00 | 398.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-16 09:40:00 | 395.65 | 2025-12-16 09:50:00 | 396.52 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-19 09:30:00 | 382.45 | 2025-12-19 09:50:00 | 381.19 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-26 10:15:00 | 390.10 | 2025-12-26 15:20:00 | 390.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest1 | 2025-12-29 10:40:00 | 385.55 | 2025-12-29 10:55:00 | 384.60 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-29 10:40:00 | 385.55 | 2025-12-29 11:40:00 | 385.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-31 11:00:00 | 389.15 | 2025-12-31 11:20:00 | 389.73 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-01-02 09:40:00 | 388.05 | 2026-01-02 10:35:00 | 389.42 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-05 10:25:00 | 382.10 | 2026-01-05 10:35:00 | 382.90 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-06 11:15:00 | 380.10 | 2026-01-06 11:35:00 | 380.63 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2026-01-07 09:55:00 | 378.20 | 2026-01-07 10:10:00 | 379.58 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-07 09:55:00 | 378.20 | 2026-01-07 10:15:00 | 378.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 11:05:00 | 375.85 | 2026-01-08 11:20:00 | 374.83 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-01-08 11:05:00 | 375.85 | 2026-01-08 14:05:00 | 373.30 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2026-01-19 09:35:00 | 348.95 | 2026-01-19 09:50:00 | 349.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-01 10:30:00 | 340.10 | 2026-02-01 10:35:00 | 339.03 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-06 11:15:00 | 332.00 | 2026-02-06 12:10:00 | 330.75 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-06 11:15:00 | 332.00 | 2026-02-06 14:00:00 | 332.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 11:05:00 | 338.25 | 2026-02-09 11:35:00 | 337.54 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-13 09:35:00 | 329.75 | 2026-02-13 09:40:00 | 330.93 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 10:10:00 | 323.15 | 2026-02-18 10:30:00 | 323.73 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-19 10:40:00 | 321.35 | 2026-02-19 11:20:00 | 320.57 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-19 10:40:00 | 321.35 | 2026-02-19 15:20:00 | 320.05 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-20 09:30:00 | 316.40 | 2026-02-20 09:35:00 | 317.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-23 10:00:00 | 315.95 | 2026-02-23 10:05:00 | 316.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 10:50:00 | 289.45 | 2026-02-27 10:55:00 | 290.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-09 10:00:00 | 272.00 | 2026-03-09 11:10:00 | 273.45 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-03-10 09:30:00 | 278.55 | 2026-03-10 09:50:00 | 279.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-13 09:30:00 | 279.55 | 2026-03-13 10:15:00 | 281.44 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-13 09:30:00 | 279.55 | 2026-03-13 11:20:00 | 279.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:15:00 | 279.65 | 2026-03-17 10:20:00 | 281.13 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-17 10:15:00 | 279.65 | 2026-03-17 11:25:00 | 279.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 314.90 | 2026-04-10 09:50:00 | 316.58 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-10 09:30:00 | 314.90 | 2026-04-10 15:20:00 | 324.90 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2026-04-15 09:40:00 | 315.00 | 2026-04-15 09:50:00 | 317.43 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-04-15 09:40:00 | 315.00 | 2026-04-15 12:25:00 | 342.90 | TARGET_HIT | 0.50 | 8.86% |
