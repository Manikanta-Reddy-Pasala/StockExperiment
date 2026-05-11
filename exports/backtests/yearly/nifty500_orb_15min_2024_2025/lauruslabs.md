# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35596 bars)
- **Last close:** 1225.20
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 16 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 76
- **Target hits / Stop hits / Partials:** 16 / 75 / 29
- **Avg / median % per leg:** 0.01% / -0.24%
- **Sum % (uncompounded):** 1.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 23 | 33.8% | 8 | 45 | 15 | -0.03% | -2.2% |
| BUY @ 2nd Alert (retest1) | 68 | 23 | 33.8% | 8 | 45 | 15 | -0.03% | -2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 21 | 40.4% | 8 | 30 | 14 | 0.07% | 3.6% |
| SELL @ 2nd Alert (retest1) | 52 | 21 | 40.4% | 8 | 30 | 14 | 0.07% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 44 | 36.7% | 16 | 75 | 29 | 0.01% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:50:00 | 430.85 | 433.86 | 0.00 | ORB-short ORB[432.55,437.30] vol=1.6x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-05-14 11:10:00 | 432.44 | 432.91 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:15:00 | 432.60 | 433.47 | 0.00 | ORB-short ORB[433.00,437.40] vol=1.9x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:30:00 | 430.34 | 432.65 | 0.00 | T1 1.5R @ 430.34 |
| Target hit | 2024-05-15 10:30:00 | 433.35 | 432.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — BUY (started 2024-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:35:00 | 442.00 | 438.84 | 0.00 | ORB-long ORB[436.00,440.10] vol=2.2x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-05-16 10:45:00 | 440.56 | 439.36 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 440.95 | 443.99 | 0.00 | ORB-short ORB[443.45,448.00] vol=2.0x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 13:00:00 | 439.34 | 443.07 | 0.00 | T1 1.5R @ 439.34 |
| Target hit | 2024-05-17 15:20:00 | 440.45 | 441.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-05-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:40:00 | 443.35 | 441.93 | 0.00 | ORB-long ORB[440.15,442.55] vol=2.4x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-05-18 09:45:00 | 442.07 | 442.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:10:00 | 445.20 | 443.43 | 0.00 | ORB-long ORB[441.05,444.20] vol=3.0x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:25:00 | 447.64 | 444.65 | 0.00 | T1 1.5R @ 447.64 |
| Target hit | 2024-05-21 11:10:00 | 445.80 | 446.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2024-05-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:55:00 | 459.25 | 455.90 | 0.00 | ORB-long ORB[453.55,458.50] vol=3.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-05-22 11:00:00 | 457.72 | 455.96 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 446.25 | 450.54 | 0.00 | ORB-short ORB[452.00,457.95] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-05-23 10:50:00 | 447.65 | 449.77 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 448.05 | 445.45 | 0.00 | ORB-long ORB[442.00,446.00] vol=2.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-05-28 09:55:00 | 446.48 | 446.08 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 11:00:00 | 441.50 | 444.08 | 0.00 | ORB-short ORB[442.55,448.00] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-05-29 11:20:00 | 442.56 | 443.96 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-05-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:10:00 | 435.70 | 438.25 | 0.00 | ORB-short ORB[437.50,441.10] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 436.85 | 438.23 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:20:00 | 436.60 | 430.70 | 0.00 | ORB-long ORB[426.00,432.40] vol=2.1x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-06-06 10:25:00 | 434.29 | 430.98 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:10:00 | 434.20 | 433.25 | 0.00 | ORB-long ORB[427.80,431.55] vol=2.1x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 11:35:00 | 436.58 | 433.60 | 0.00 | T1 1.5R @ 436.58 |
| Target hit | 2024-06-07 12:50:00 | 436.60 | 436.67 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 10:15:00 | 437.30 | 439.52 | 0.00 | ORB-short ORB[439.10,443.65] vol=1.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-06-12 10:55:00 | 438.85 | 438.84 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 433.90 | 435.78 | 0.00 | ORB-short ORB[434.10,439.90] vol=1.9x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-06-13 10:00:00 | 435.51 | 435.55 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 437.55 | 438.92 | 0.00 | ORB-short ORB[438.50,441.75] vol=1.6x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:15:00 | 435.23 | 437.34 | 0.00 | T1 1.5R @ 435.23 |
| Stop hit — per-position SL triggered | 2024-06-14 11:45:00 | 437.55 | 437.30 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:50:00 | 434.10 | 430.46 | 0.00 | ORB-long ORB[427.75,432.95] vol=2.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-06-19 10:00:00 | 432.59 | 431.37 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-06-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:10:00 | 429.35 | 427.66 | 0.00 | ORB-long ORB[423.90,427.40] vol=9.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:25:00 | 430.95 | 427.98 | 0.00 | T1 1.5R @ 430.95 |
| Stop hit — per-position SL triggered | 2024-06-26 11:00:00 | 429.35 | 428.27 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:50:00 | 425.65 | 427.63 | 0.00 | ORB-short ORB[426.20,428.85] vol=1.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-06-27 10:55:00 | 426.78 | 427.61 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:30:00 | 429.00 | 427.07 | 0.00 | ORB-long ORB[423.60,428.90] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-06-28 09:45:00 | 427.58 | 427.24 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 11:05:00 | 436.75 | 432.94 | 0.00 | ORB-long ORB[430.15,435.50] vol=7.0x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:10:00 | 438.73 | 433.92 | 0.00 | T1 1.5R @ 438.73 |
| Stop hit — per-position SL triggered | 2024-07-02 12:10:00 | 436.75 | 437.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 441.00 | 438.03 | 0.00 | ORB-long ORB[435.30,439.25] vol=2.1x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-07-03 09:45:00 | 439.34 | 438.38 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:05:00 | 441.40 | 438.52 | 0.00 | ORB-long ORB[436.05,439.60] vol=2.3x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 440.20 | 438.80 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:45:00 | 460.75 | 458.07 | 0.00 | ORB-long ORB[454.45,459.00] vol=3.0x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-07-05 09:50:00 | 458.73 | 458.15 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:20:00 | 470.15 | 472.81 | 0.00 | ORB-short ORB[472.65,478.00] vol=3.0x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-07-09 10:25:00 | 472.10 | 472.78 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:15:00 | 472.45 | 477.04 | 0.00 | ORB-short ORB[478.45,484.30] vol=1.7x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:25:00 | 469.34 | 476.29 | 0.00 | T1 1.5R @ 469.34 |
| Stop hit — per-position SL triggered | 2024-07-10 11:45:00 | 472.45 | 473.58 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:00:00 | 470.50 | 471.05 | 0.00 | ORB-short ORB[470.80,474.80] vol=1.6x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-07-11 12:50:00 | 471.73 | 471.10 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-07-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:10:00 | 467.55 | 470.60 | 0.00 | ORB-short ORB[470.95,474.00] vol=1.7x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-07-12 10:30:00 | 468.91 | 470.17 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-07-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:20:00 | 469.15 | 465.95 | 0.00 | ORB-long ORB[462.85,466.30] vol=2.3x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-07-15 10:35:00 | 467.51 | 466.35 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-07-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:45:00 | 467.90 | 466.23 | 0.00 | ORB-long ORB[463.00,467.70] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-07-16 09:55:00 | 466.74 | 466.38 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:35:00 | 459.90 | 461.66 | 0.00 | ORB-short ORB[460.00,465.00] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 461.35 | 461.60 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 439.20 | 441.37 | 0.00 | ORB-short ORB[439.50,444.80] vol=1.5x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 09:45:00 | 437.07 | 439.91 | 0.00 | T1 1.5R @ 437.07 |
| Target hit | 2024-07-23 14:50:00 | 432.95 | 432.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2024-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:50:00 | 438.00 | 437.21 | 0.00 | ORB-long ORB[430.00,435.05] vol=4.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-07-24 13:25:00 | 436.33 | 437.55 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-07-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:55:00 | 425.65 | 428.04 | 0.00 | ORB-short ORB[428.05,433.95] vol=1.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-07-25 10:05:00 | 427.53 | 427.98 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 447.90 | 450.21 | 0.00 | ORB-short ORB[449.05,454.85] vol=1.8x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-07-29 09:50:00 | 450.07 | 449.86 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-07-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:45:00 | 460.75 | 455.09 | 0.00 | ORB-long ORB[448.05,453.90] vol=3.9x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:55:00 | 463.18 | 457.72 | 0.00 | T1 1.5R @ 463.18 |
| Stop hit — per-position SL triggered | 2024-07-30 11:20:00 | 460.75 | 458.58 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 463.70 | 460.98 | 0.00 | ORB-long ORB[457.35,461.10] vol=3.0x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 462.11 | 461.30 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-08-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:35:00 | 459.10 | 459.39 | 0.00 | ORB-short ORB[461.25,466.10] vol=10.8x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:50:00 | 457.12 | 458.99 | 0.00 | T1 1.5R @ 457.12 |
| Target hit | 2024-08-01 13:10:00 | 457.00 | 456.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 39 — SELL (started 2024-08-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:05:00 | 432.50 | 435.22 | 0.00 | ORB-short ORB[434.15,437.05] vol=4.1x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:35:00 | 430.01 | 432.91 | 0.00 | T1 1.5R @ 430.01 |
| Target hit | 2024-08-09 12:10:00 | 430.50 | 429.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — SELL (started 2024-08-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:40:00 | 418.60 | 420.82 | 0.00 | ORB-short ORB[420.60,425.90] vol=5.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 420.43 | 420.85 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 436.00 | 432.72 | 0.00 | ORB-long ORB[428.00,432.35] vol=1.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-08-19 09:50:00 | 434.42 | 433.77 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 436.75 | 438.77 | 0.00 | ORB-short ORB[436.80,441.80] vol=1.8x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:10:00 | 434.50 | 437.65 | 0.00 | T1 1.5R @ 434.50 |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 436.75 | 437.62 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:10:00 | 442.25 | 440.12 | 0.00 | ORB-long ORB[437.00,441.25] vol=1.8x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:40:00 | 444.08 | 440.97 | 0.00 | T1 1.5R @ 444.08 |
| Target hit | 2024-08-21 15:20:00 | 445.95 | 443.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2024-08-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:55:00 | 452.35 | 448.26 | 0.00 | ORB-long ORB[447.15,450.80] vol=5.7x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-08-23 11:05:00 | 450.91 | 448.60 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 452.55 | 450.71 | 0.00 | ORB-long ORB[446.30,450.65] vol=2.6x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:45:00 | 454.57 | 451.83 | 0.00 | T1 1.5R @ 454.57 |
| Target hit | 2024-08-27 11:30:00 | 454.00 | 454.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2024-08-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:05:00 | 463.90 | 460.76 | 0.00 | ORB-long ORB[455.65,461.20] vol=1.6x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 462.05 | 461.22 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:00:00 | 462.35 | 459.61 | 0.00 | ORB-long ORB[455.35,461.20] vol=1.9x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-08-30 10:10:00 | 460.40 | 459.93 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 467.60 | 465.19 | 0.00 | ORB-long ORB[461.75,466.75] vol=2.4x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 466.01 | 465.47 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:45:00 | 482.50 | 477.41 | 0.00 | ORB-long ORB[472.70,479.65] vol=2.9x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-09-05 09:55:00 | 480.47 | 478.94 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 512.45 | 509.46 | 0.00 | ORB-long ORB[506.30,510.40] vol=1.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-09-11 10:00:00 | 510.36 | 510.75 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-09-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:45:00 | 504.45 | 507.09 | 0.00 | ORB-short ORB[506.80,510.00] vol=2.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-09-13 11:20:00 | 505.86 | 506.58 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 499.25 | 501.99 | 0.00 | ORB-short ORB[501.00,506.40] vol=1.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-09-17 10:20:00 | 500.68 | 501.79 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:40:00 | 495.50 | 496.84 | 0.00 | ORB-short ORB[496.20,502.00] vol=1.6x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:55:00 | 493.51 | 496.28 | 0.00 | T1 1.5R @ 493.51 |
| Target hit | 2024-09-18 10:40:00 | 494.10 | 493.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — SELL (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 11:15:00 | 466.65 | 468.69 | 0.00 | ORB-short ORB[468.15,473.65] vol=2.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-09-20 12:25:00 | 468.28 | 468.38 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-09-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:55:00 | 468.55 | 464.74 | 0.00 | ORB-long ORB[462.05,465.95] vol=2.8x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-09-27 11:05:00 | 466.76 | 464.96 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-10-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:00:00 | 440.00 | 443.49 | 0.00 | ORB-short ORB[445.45,448.90] vol=1.7x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:30:00 | 437.12 | 442.30 | 0.00 | T1 1.5R @ 437.12 |
| Target hit | 2024-10-07 11:20:00 | 438.60 | 437.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2024-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:30:00 | 446.65 | 443.40 | 0.00 | ORB-long ORB[441.40,444.30] vol=2.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 09:50:00 | 449.29 | 446.36 | 0.00 | T1 1.5R @ 449.29 |
| Target hit | 2024-10-09 15:20:00 | 453.20 | 452.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2024-10-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 11:05:00 | 457.30 | 455.38 | 0.00 | ORB-long ORB[453.00,456.70] vol=2.5x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:15:00 | 459.44 | 457.04 | 0.00 | T1 1.5R @ 459.44 |
| Stop hit — per-position SL triggered | 2024-10-10 11:30:00 | 457.30 | 457.14 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 472.05 | 469.72 | 0.00 | ORB-long ORB[466.10,471.90] vol=1.5x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:40:00 | 475.40 | 472.61 | 0.00 | T1 1.5R @ 475.40 |
| Target hit | 2024-10-11 11:35:00 | 473.30 | 473.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 483.80 | 482.07 | 0.00 | ORB-long ORB[477.00,482.75] vol=3.7x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:40:00 | 486.33 | 483.07 | 0.00 | T1 1.5R @ 486.33 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 483.80 | 483.34 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-10-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:45:00 | 474.30 | 477.32 | 0.00 | ORB-short ORB[478.10,482.45] vol=2.0x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:10:00 | 471.58 | 476.99 | 0.00 | T1 1.5R @ 471.58 |
| Target hit | 2024-10-17 15:20:00 | 466.20 | 472.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2024-10-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:35:00 | 469.70 | 465.30 | 0.00 | ORB-long ORB[461.00,465.60] vol=2.9x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-10-18 09:45:00 | 467.81 | 465.90 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-11-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:55:00 | 493.95 | 494.15 | 0.00 | ORB-short ORB[495.15,502.45] vol=10.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-11-07 12:45:00 | 495.73 | 493.87 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-11-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 10:00:00 | 486.60 | 481.90 | 0.00 | ORB-long ORB[477.00,483.45] vol=2.9x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-11-14 10:30:00 | 484.06 | 482.95 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:15:00 | 491.05 | 488.47 | 0.00 | ORB-long ORB[483.65,488.10] vol=1.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-11-19 10:45:00 | 489.32 | 488.82 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-11-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 11:05:00 | 489.00 | 487.16 | 0.00 | ORB-long ORB[484.00,488.95] vol=2.2x ATR=1.82 |
| Target hit | 2024-11-21 15:20:00 | 489.30 | 488.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — BUY (started 2024-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:30:00 | 494.25 | 492.34 | 0.00 | ORB-long ORB[486.85,494.05] vol=3.2x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-11-22 09:40:00 | 492.57 | 492.84 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 11:00:00 | 536.40 | 532.86 | 0.00 | ORB-long ORB[531.15,536.20] vol=2.1x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-11-26 11:05:00 | 534.45 | 532.91 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-11-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:45:00 | 553.35 | 548.55 | 0.00 | ORB-long ORB[543.05,550.00] vol=1.8x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-11-27 11:45:00 | 550.98 | 549.62 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-12-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 09:50:00 | 578.95 | 582.08 | 0.00 | ORB-short ORB[579.00,586.10] vol=3.3x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-12-04 10:00:00 | 581.27 | 581.45 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:40:00 | 589.85 | 586.78 | 0.00 | ORB-long ORB[582.50,588.45] vol=3.3x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-12-05 09:45:00 | 587.64 | 586.96 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:20:00 | 566.60 | 570.06 | 0.00 | ORB-short ORB[571.60,575.35] vol=4.7x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:35:00 | 564.54 | 569.24 | 0.00 | T1 1.5R @ 564.54 |
| Stop hit — per-position SL triggered | 2024-12-12 10:40:00 | 566.60 | 569.09 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 558.75 | 561.56 | 0.00 | ORB-short ORB[560.00,567.60] vol=2.0x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-12-13 09:40:00 | 560.53 | 561.39 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-12-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:10:00 | 569.80 | 572.49 | 0.00 | ORB-short ORB[570.30,574.70] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-12-16 10:25:00 | 571.51 | 572.16 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:35:00 | 576.00 | 573.90 | 0.00 | ORB-long ORB[570.60,573.45] vol=1.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-12-18 09:50:00 | 574.40 | 574.30 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-12-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:55:00 | 570.60 | 565.99 | 0.00 | ORB-long ORB[560.40,566.40] vol=1.6x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 14:15:00 | 573.32 | 569.38 | 0.00 | T1 1.5R @ 573.32 |
| Target hit | 2024-12-19 15:20:00 | 572.45 | 570.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 567.90 | 562.33 | 0.00 | ORB-long ORB[560.55,566.10] vol=2.0x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:35:00 | 570.62 | 563.32 | 0.00 | T1 1.5R @ 570.62 |
| Stop hit — per-position SL triggered | 2024-12-23 11:40:00 | 567.90 | 563.40 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-12-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:05:00 | 572.90 | 567.93 | 0.00 | ORB-long ORB[562.40,566.95] vol=2.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-12-24 11:05:00 | 571.18 | 569.79 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-12-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:00:00 | 560.40 | 562.24 | 0.00 | ORB-short ORB[563.50,568.40] vol=3.4x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-12-26 10:05:00 | 562.21 | 562.61 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 575.00 | 573.44 | 0.00 | ORB-long ORB[570.50,574.90] vol=3.7x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 09:40:00 | 577.52 | 574.60 | 0.00 | T1 1.5R @ 577.52 |
| Stop hit — per-position SL triggered | 2024-12-27 09:45:00 | 575.00 | 574.74 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-12-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:55:00 | 593.05 | 589.58 | 0.00 | ORB-long ORB[585.00,592.80] vol=2.7x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 590.91 | 590.67 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-01-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:10:00 | 606.80 | 603.27 | 0.00 | ORB-long ORB[600.00,606.15] vol=2.0x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-01-01 11:00:00 | 604.49 | 604.63 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:05:00 | 600.35 | 608.42 | 0.00 | ORB-short ORB[609.40,615.90] vol=1.6x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 597.29 | 607.18 | 0.00 | T1 1.5R @ 597.29 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 600.35 | 605.83 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-01-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 09:30:00 | 607.40 | 605.26 | 0.00 | ORB-long ORB[601.30,606.00] vol=1.8x ATR=2.37 |
| Stop hit — per-position SL triggered | 2025-01-07 09:45:00 | 605.03 | 605.73 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-01-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:45:00 | 608.70 | 612.66 | 0.00 | ORB-short ORB[612.10,617.90] vol=1.9x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-01-08 09:50:00 | 610.98 | 612.55 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2025-01-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:00:00 | 583.80 | 588.26 | 0.00 | ORB-short ORB[584.75,593.20] vol=2.4x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:20:00 | 580.51 | 586.90 | 0.00 | T1 1.5R @ 580.51 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 583.80 | 585.19 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:50:00 | 585.85 | 584.13 | 0.00 | ORB-long ORB[581.00,585.45] vol=2.4x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 583.60 | 584.11 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-02-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:55:00 | 644.05 | 638.03 | 0.00 | ORB-long ORB[633.90,639.40] vol=2.4x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-02-07 11:50:00 | 641.43 | 639.89 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-03-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 10:35:00 | 571.55 | 576.42 | 0.00 | ORB-short ORB[574.50,582.05] vol=1.8x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-03-07 10:40:00 | 573.40 | 575.98 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 587.85 | 584.33 | 0.00 | ORB-long ORB[581.05,586.75] vol=3.2x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 09:45:00 | 590.34 | 586.51 | 0.00 | T1 1.5R @ 590.34 |
| Stop hit — per-position SL triggered | 2025-03-18 10:30:00 | 587.85 | 588.33 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 611.75 | 608.71 | 0.00 | ORB-long ORB[602.70,611.20] vol=1.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 609.98 | 609.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:50:00 | 430.85 | 2024-05-14 11:10:00 | 432.44 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-15 10:15:00 | 432.60 | 2024-05-15 10:30:00 | 430.34 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-05-15 10:15:00 | 432.60 | 2024-05-15 10:30:00 | 433.35 | TARGET_HIT | 0.50 | -0.17% |
| BUY | retest1 | 2024-05-16 10:35:00 | 442.00 | 2024-05-16 10:45:00 | 440.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-17 10:55:00 | 440.95 | 2024-05-17 13:00:00 | 439.34 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-17 10:55:00 | 440.95 | 2024-05-17 15:20:00 | 440.45 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-05-18 09:40:00 | 443.35 | 2024-05-18 09:45:00 | 442.07 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-21 10:10:00 | 445.20 | 2024-05-21 10:25:00 | 447.64 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-21 10:10:00 | 445.20 | 2024-05-21 11:10:00 | 445.80 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-05-22 10:55:00 | 459.25 | 2024-05-22 11:00:00 | 457.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-23 10:35:00 | 446.25 | 2024-05-23 10:50:00 | 447.65 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-28 09:45:00 | 448.05 | 2024-05-28 09:55:00 | 446.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-29 11:00:00 | 441.50 | 2024-05-29 11:20:00 | 442.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-30 11:10:00 | 435.70 | 2024-05-30 11:15:00 | 436.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-06 10:20:00 | 436.60 | 2024-06-06 10:25:00 | 434.29 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-06-07 11:10:00 | 434.20 | 2024-06-07 11:35:00 | 436.58 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-07 11:10:00 | 434.20 | 2024-06-07 12:50:00 | 436.60 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-06-12 10:15:00 | 437.30 | 2024-06-12 10:55:00 | 438.85 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-13 09:35:00 | 433.90 | 2024-06-13 10:00:00 | 435.51 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-06-14 09:55:00 | 437.55 | 2024-06-14 11:15:00 | 435.23 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-14 09:55:00 | 437.55 | 2024-06-14 11:45:00 | 437.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-19 09:50:00 | 434.10 | 2024-06-19 10:00:00 | 432.59 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-26 10:10:00 | 429.35 | 2024-06-26 10:25:00 | 430.95 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-26 10:10:00 | 429.35 | 2024-06-26 11:00:00 | 429.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:50:00 | 425.65 | 2024-06-27 10:55:00 | 426.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-28 09:30:00 | 429.00 | 2024-06-28 09:45:00 | 427.58 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-02 11:05:00 | 436.75 | 2024-07-02 11:10:00 | 438.73 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-02 11:05:00 | 436.75 | 2024-07-02 12:10:00 | 436.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 09:40:00 | 441.00 | 2024-07-03 09:45:00 | 439.34 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-04 10:05:00 | 441.40 | 2024-07-04 10:10:00 | 440.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-05 09:45:00 | 460.75 | 2024-07-05 09:50:00 | 458.73 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-09 10:20:00 | 470.15 | 2024-07-09 10:25:00 | 472.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-10 10:15:00 | 472.45 | 2024-07-10 10:25:00 | 469.34 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-10 10:15:00 | 472.45 | 2024-07-10 11:45:00 | 472.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 11:00:00 | 470.50 | 2024-07-11 12:50:00 | 471.73 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-12 10:10:00 | 467.55 | 2024-07-12 10:30:00 | 468.91 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-15 10:20:00 | 469.15 | 2024-07-15 10:35:00 | 467.51 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-16 09:45:00 | 467.90 | 2024-07-16 09:55:00 | 466.74 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-18 09:35:00 | 459.90 | 2024-07-18 09:40:00 | 461.35 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-23 09:30:00 | 439.20 | 2024-07-23 09:45:00 | 437.07 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-23 09:30:00 | 439.20 | 2024-07-23 14:50:00 | 432.95 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2024-07-24 10:50:00 | 438.00 | 2024-07-24 13:25:00 | 436.33 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-25 09:55:00 | 425.65 | 2024-07-25 10:05:00 | 427.53 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-07-29 09:30:00 | 447.90 | 2024-07-29 09:50:00 | 450.07 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-07-30 10:45:00 | 460.75 | 2024-07-30 10:55:00 | 463.18 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-30 10:45:00 | 460.75 | 2024-07-30 11:20:00 | 460.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 09:40:00 | 463.70 | 2024-07-31 09:45:00 | 462.11 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-01 10:35:00 | 459.10 | 2024-08-01 11:50:00 | 457.12 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-01 10:35:00 | 459.10 | 2024-08-01 13:10:00 | 457.00 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-09 10:05:00 | 432.50 | 2024-08-09 10:35:00 | 430.01 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-08-09 10:05:00 | 432.50 | 2024-08-09 12:10:00 | 430.50 | TARGET_HIT | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-14 09:40:00 | 418.60 | 2024-08-14 09:45:00 | 420.43 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-19 09:35:00 | 436.00 | 2024-08-19 09:50:00 | 434.42 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-20 09:30:00 | 436.75 | 2024-08-20 10:10:00 | 434.50 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-20 09:30:00 | 436.75 | 2024-08-20 10:15:00 | 436.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 10:10:00 | 442.25 | 2024-08-21 10:40:00 | 444.08 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-08-21 10:10:00 | 442.25 | 2024-08-21 15:20:00 | 445.95 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-08-23 10:55:00 | 452.35 | 2024-08-23 11:05:00 | 450.91 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-27 09:30:00 | 452.55 | 2024-08-27 09:45:00 | 454.57 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-27 09:30:00 | 452.55 | 2024-08-27 11:30:00 | 454.00 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-29 10:05:00 | 463.90 | 2024-08-29 10:15:00 | 462.05 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-08-30 10:00:00 | 462.35 | 2024-08-30 10:10:00 | 460.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-09-03 09:30:00 | 467.60 | 2024-09-03 09:40:00 | 466.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-05 09:45:00 | 482.50 | 2024-09-05 09:55:00 | 480.47 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-09-11 09:35:00 | 512.45 | 2024-09-11 10:00:00 | 510.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-09-13 10:45:00 | 504.45 | 2024-09-13 11:20:00 | 505.86 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-17 10:10:00 | 499.25 | 2024-09-17 10:20:00 | 500.68 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-18 09:40:00 | 495.50 | 2024-09-18 09:55:00 | 493.51 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-18 09:40:00 | 495.50 | 2024-09-18 10:40:00 | 494.10 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-20 11:15:00 | 466.65 | 2024-09-20 12:25:00 | 468.28 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-27 10:55:00 | 468.55 | 2024-09-27 11:05:00 | 466.76 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-07 10:00:00 | 440.00 | 2024-10-07 10:30:00 | 437.12 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-10-07 10:00:00 | 440.00 | 2024-10-07 11:20:00 | 438.60 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-10-09 09:30:00 | 446.65 | 2024-10-09 09:50:00 | 449.29 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-10-09 09:30:00 | 446.65 | 2024-10-09 15:20:00 | 453.20 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2024-10-10 11:05:00 | 457.30 | 2024-10-10 11:15:00 | 459.44 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-10-10 11:05:00 | 457.30 | 2024-10-10 11:30:00 | 457.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:35:00 | 472.05 | 2024-10-11 10:40:00 | 475.40 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-10-11 09:35:00 | 472.05 | 2024-10-11 11:35:00 | 473.30 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-10-15 09:30:00 | 483.80 | 2024-10-15 09:40:00 | 486.33 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-10-15 09:30:00 | 483.80 | 2024-10-15 09:50:00 | 483.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:45:00 | 474.30 | 2024-10-17 11:10:00 | 471.58 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-10-17 10:45:00 | 474.30 | 2024-10-17 15:20:00 | 466.20 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2024-10-18 09:35:00 | 469.70 | 2024-10-18 09:45:00 | 467.81 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-11-07 10:55:00 | 493.95 | 2024-11-07 12:45:00 | 495.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-14 10:00:00 | 486.60 | 2024-11-14 10:30:00 | 484.06 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-11-19 10:15:00 | 491.05 | 2024-11-19 10:45:00 | 489.32 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-11-21 11:05:00 | 489.00 | 2024-11-21 15:20:00 | 489.30 | TARGET_HIT | 1.00 | 0.06% |
| BUY | retest1 | 2024-11-22 09:30:00 | 494.25 | 2024-11-22 09:40:00 | 492.57 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-26 11:00:00 | 536.40 | 2024-11-26 11:05:00 | 534.45 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-11-27 10:45:00 | 553.35 | 2024-11-27 11:45:00 | 550.98 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-04 09:50:00 | 578.95 | 2024-12-04 10:00:00 | 581.27 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-05 09:40:00 | 589.85 | 2024-12-05 09:45:00 | 587.64 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-12 10:20:00 | 566.60 | 2024-12-12 10:35:00 | 564.54 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-12 10:20:00 | 566.60 | 2024-12-12 10:40:00 | 566.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 09:35:00 | 558.75 | 2024-12-13 09:40:00 | 560.53 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-16 10:10:00 | 569.80 | 2024-12-16 10:25:00 | 571.51 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-18 09:35:00 | 576.00 | 2024-12-18 09:50:00 | 574.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-19 10:55:00 | 570.60 | 2024-12-19 14:15:00 | 573.32 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-12-19 10:55:00 | 570.60 | 2024-12-19 15:20:00 | 572.45 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-12-23 11:15:00 | 567.90 | 2024-12-23 11:35:00 | 570.62 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-12-23 11:15:00 | 567.90 | 2024-12-23 11:40:00 | 567.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:05:00 | 572.90 | 2024-12-24 11:05:00 | 571.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-26 10:00:00 | 560.40 | 2024-12-26 10:05:00 | 562.21 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-27 09:30:00 | 575.00 | 2024-12-27 09:40:00 | 577.52 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-27 09:30:00 | 575.00 | 2024-12-27 09:45:00 | 575.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 09:55:00 | 593.05 | 2024-12-30 10:05:00 | 590.91 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-01 10:10:00 | 606.80 | 2025-01-01 11:00:00 | 604.49 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-06 11:05:00 | 600.35 | 2025-01-06 11:10:00 | 597.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-06 11:05:00 | 600.35 | 2025-01-06 11:30:00 | 600.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-07 09:30:00 | 607.40 | 2025-01-07 09:45:00 | 605.03 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-08 09:45:00 | 608.70 | 2025-01-08 09:50:00 | 610.98 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-21 10:00:00 | 583.80 | 2025-01-21 10:20:00 | 580.51 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-21 10:00:00 | 583.80 | 2025-01-21 11:45:00 | 583.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 10:50:00 | 585.85 | 2025-02-01 11:00:00 | 583.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-07 10:55:00 | 644.05 | 2025-02-07 11:50:00 | 641.43 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-03-07 10:35:00 | 571.55 | 2025-03-07 10:40:00 | 573.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-18 09:35:00 | 587.85 | 2025-03-18 09:45:00 | 590.34 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-03-18 09:35:00 | 587.85 | 2025-03-18 10:30:00 | 587.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:30:00 | 611.75 | 2025-03-21 09:50:00 | 609.98 | STOP_HIT | 1.00 | -0.29% |
