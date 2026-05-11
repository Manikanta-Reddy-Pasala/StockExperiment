# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-04-02 15:25:00 (16738 bars)
- **Last close:** 300.40
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
| TARGET_HIT | 14 |
| STOP_HIT | 81 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 81
- **Target hits / Stop hits / Partials:** 14 / 81 / 36
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 11.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 19 | 32.8% | 7 | 39 | 12 | 0.01% | 0.3% |
| BUY @ 2nd Alert (retest1) | 58 | 19 | 32.8% | 7 | 39 | 12 | 0.01% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 31 | 42.5% | 7 | 42 | 24 | 0.15% | 11.2% |
| SELL @ 2nd Alert (retest1) | 73 | 31 | 42.5% | 7 | 42 | 24 | 0.15% | 11.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 131 | 50 | 38.2% | 14 | 81 | 36 | 0.09% | 11.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:35:00 | 379.15 | 377.34 | 0.00 | ORB-long ORB[374.50,377.70] vol=4.4x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-05-14 10:55:00 | 378.10 | 377.44 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 385.75 | 383.92 | 0.00 | ORB-long ORB[381.15,384.90] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 384.63 | 384.19 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 11:10:00 | 382.90 | 385.18 | 0.00 | ORB-short ORB[384.65,388.50] vol=8.7x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 11:15:00 | 380.99 | 383.39 | 0.00 | T1 1.5R @ 380.99 |
| Stop hit — per-position SL triggered | 2025-05-27 12:10:00 | 382.90 | 382.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 11:05:00 | 386.00 | 386.08 | 0.00 | ORB-short ORB[386.50,390.20] vol=4.7x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-05-30 11:25:00 | 386.92 | 386.10 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:45:00 | 391.30 | 387.99 | 0.00 | ORB-long ORB[383.10,388.65] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-06-02 10:20:00 | 390.21 | 389.75 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:30:00 | 382.90 | 384.43 | 0.00 | ORB-short ORB[384.50,387.60] vol=1.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-06-04 10:45:00 | 383.85 | 384.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:10:00 | 393.70 | 390.55 | 0.00 | ORB-long ORB[388.80,392.50] vol=2.5x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-06-06 11:25:00 | 392.53 | 390.83 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 11:05:00 | 392.45 | 395.58 | 0.00 | ORB-short ORB[394.25,397.95] vol=1.6x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-06-09 11:10:00 | 393.43 | 395.55 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 380.20 | 382.10 | 0.00 | ORB-short ORB[381.30,385.90] vol=2.4x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:40:00 | 378.56 | 381.34 | 0.00 | T1 1.5R @ 378.56 |
| Stop hit — per-position SL triggered | 2025-06-16 10:10:00 | 380.20 | 380.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 11:05:00 | 376.00 | 377.78 | 0.00 | ORB-short ORB[376.65,379.95] vol=2.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:35:00 | 374.61 | 377.07 | 0.00 | T1 1.5R @ 374.61 |
| Target hit | 2025-06-19 14:00:00 | 374.95 | 374.34 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2025-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:25:00 | 379.70 | 378.20 | 0.00 | ORB-long ORB[373.15,377.40] vol=1.8x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:55:00 | 381.37 | 379.31 | 0.00 | T1 1.5R @ 381.37 |
| Target hit | 2025-06-20 15:20:00 | 380.05 | 379.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:35:00 | 384.65 | 383.50 | 0.00 | ORB-long ORB[382.05,383.90] vol=2.0x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 09:40:00 | 386.10 | 383.83 | 0.00 | T1 1.5R @ 386.10 |
| Stop hit — per-position SL triggered | 2025-06-24 09:45:00 | 384.65 | 383.89 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:30:00 | 389.60 | 387.56 | 0.00 | ORB-long ORB[385.00,388.80] vol=2.0x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-06-25 09:45:00 | 388.69 | 388.29 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:00:00 | 384.60 | 385.54 | 0.00 | ORB-short ORB[386.00,387.90] vol=1.9x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-06-26 10:10:00 | 385.40 | 385.49 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 11:00:00 | 390.20 | 387.89 | 0.00 | ORB-long ORB[387.05,389.65] vol=3.7x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 11:40:00 | 391.74 | 388.89 | 0.00 | T1 1.5R @ 391.74 |
| Stop hit — per-position SL triggered | 2025-06-30 11:45:00 | 390.20 | 389.01 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:05:00 | 384.60 | 386.55 | 0.00 | ORB-short ORB[385.60,389.15] vol=1.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:15:00 | 383.26 | 385.95 | 0.00 | T1 1.5R @ 383.26 |
| Stop hit — per-position SL triggered | 2025-07-02 11:40:00 | 384.60 | 384.59 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 10:15:00 | 382.60 | 383.25 | 0.00 | ORB-short ORB[382.70,385.45] vol=1.5x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-07-03 10:25:00 | 383.46 | 383.24 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 383.25 | 385.64 | 0.00 | ORB-short ORB[385.30,387.90] vol=1.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-07-08 11:25:00 | 383.92 | 385.40 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:15:00 | 389.05 | 388.34 | 0.00 | ORB-long ORB[387.00,388.70] vol=5.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-07-09 12:00:00 | 388.13 | 388.45 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:35:00 | 389.70 | 387.85 | 0.00 | ORB-long ORB[386.00,388.70] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-07-10 10:40:00 | 388.82 | 387.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:10:00 | 385.90 | 387.06 | 0.00 | ORB-short ORB[386.50,388.75] vol=2.1x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:30:00 | 384.67 | 386.56 | 0.00 | T1 1.5R @ 384.67 |
| Target hit | 2025-07-11 15:20:00 | 380.00 | 383.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-07-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:00:00 | 381.50 | 383.01 | 0.00 | ORB-short ORB[382.05,385.15] vol=1.5x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 380.16 | 382.11 | 0.00 | T1 1.5R @ 380.16 |
| Stop hit — per-position SL triggered | 2025-07-18 10:35:00 | 381.50 | 381.58 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:10:00 | 388.20 | 386.39 | 0.00 | ORB-long ORB[383.55,387.30] vol=2.2x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-07-21 10:45:00 | 387.13 | 386.94 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:45:00 | 392.80 | 391.72 | 0.00 | ORB-long ORB[388.40,391.35] vol=3.2x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:00:00 | 394.39 | 392.29 | 0.00 | T1 1.5R @ 394.39 |
| Stop hit — per-position SL triggered | 2025-07-22 10:20:00 | 392.80 | 392.87 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 395.10 | 394.17 | 0.00 | ORB-long ORB[390.40,395.00] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-07-24 09:50:00 | 394.31 | 394.30 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 388.45 | 390.50 | 0.00 | ORB-short ORB[388.75,392.00] vol=1.5x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:45:00 | 386.87 | 390.05 | 0.00 | T1 1.5R @ 386.87 |
| Target hit | 2025-07-25 15:20:00 | 380.80 | 384.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-07-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:30:00 | 384.20 | 381.63 | 0.00 | ORB-long ORB[377.30,380.80] vol=2.4x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-07-28 10:45:00 | 383.19 | 382.00 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:35:00 | 385.30 | 384.29 | 0.00 | ORB-long ORB[381.40,385.00] vol=2.8x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:45:00 | 386.88 | 385.74 | 0.00 | T1 1.5R @ 386.88 |
| Target hit | 2025-07-29 15:20:00 | 391.55 | 389.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-08-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:45:00 | 380.00 | 381.14 | 0.00 | ORB-short ORB[380.85,384.70] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-08-01 10:30:00 | 381.27 | 380.63 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 10:35:00 | 376.20 | 377.16 | 0.00 | ORB-short ORB[376.55,378.30] vol=4.4x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-08-13 11:05:00 | 376.85 | 377.10 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 374.55 | 375.57 | 0.00 | ORB-short ORB[375.35,376.80] vol=1.5x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-08-14 09:45:00 | 375.17 | 375.21 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:50:00 | 380.35 | 379.13 | 0.00 | ORB-long ORB[376.00,378.30] vol=11.2x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-08-19 10:35:00 | 379.49 | 380.13 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:55:00 | 398.05 | 395.80 | 0.00 | ORB-long ORB[392.40,397.00] vol=4.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:25:00 | 399.51 | 396.72 | 0.00 | T1 1.5R @ 399.51 |
| Target hit | 2025-08-25 15:20:00 | 400.50 | 399.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-09-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:40:00 | 404.70 | 401.07 | 0.00 | ORB-long ORB[396.30,401.70] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-09-01 09:55:00 | 403.40 | 402.46 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:50:00 | 420.90 | 415.30 | 0.00 | ORB-long ORB[410.70,414.90] vol=2.3x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-09-02 11:25:00 | 419.51 | 416.27 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:05:00 | 413.10 | 411.56 | 0.00 | ORB-long ORB[406.50,410.90] vol=1.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 412.00 | 411.58 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 417.40 | 414.89 | 0.00 | ORB-long ORB[413.05,416.25] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-09-08 09:35:00 | 416.30 | 415.10 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:50:00 | 430.85 | 427.92 | 0.00 | ORB-long ORB[425.20,429.80] vol=2.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-09-10 10:55:00 | 429.86 | 428.04 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:35:00 | 426.20 | 424.33 | 0.00 | ORB-long ORB[421.70,424.45] vol=3.5x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-09-12 09:40:00 | 425.26 | 424.45 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:35:00 | 420.30 | 419.01 | 0.00 | ORB-long ORB[415.75,419.80] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-09-16 10:25:00 | 419.40 | 419.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:50:00 | 421.20 | 420.04 | 0.00 | ORB-long ORB[417.90,420.80] vol=1.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-09-18 10:10:00 | 420.31 | 420.19 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 421.70 | 423.52 | 0.00 | ORB-short ORB[423.00,425.10] vol=1.8x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-09-19 10:05:00 | 422.59 | 423.32 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 411.65 | 414.05 | 0.00 | ORB-short ORB[412.80,416.15] vol=2.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-09-22 11:10:00 | 412.70 | 413.98 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:45:00 | 399.80 | 401.63 | 0.00 | ORB-short ORB[400.50,404.10] vol=1.6x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:00:00 | 398.44 | 401.28 | 0.00 | T1 1.5R @ 398.44 |
| Stop hit — per-position SL triggered | 2025-09-24 11:35:00 | 399.80 | 400.58 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:45:00 | 399.85 | 396.16 | 0.00 | ORB-long ORB[393.20,395.70] vol=1.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-09-26 09:50:00 | 398.56 | 396.55 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:15:00 | 388.70 | 390.35 | 0.00 | ORB-short ORB[388.80,391.00] vol=1.6x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:30:00 | 387.44 | 390.15 | 0.00 | T1 1.5R @ 387.44 |
| Stop hit — per-position SL triggered | 2025-09-29 12:25:00 | 388.70 | 389.41 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-09-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 11:10:00 | 390.00 | 388.55 | 0.00 | ORB-long ORB[387.40,389.80] vol=5.9x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 389.09 | 388.57 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:35:00 | 394.45 | 392.86 | 0.00 | ORB-long ORB[391.05,393.70] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-10-01 10:40:00 | 393.47 | 393.00 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:45:00 | 403.00 | 400.85 | 0.00 | ORB-long ORB[398.10,400.90] vol=3.4x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:05:00 | 404.69 | 402.13 | 0.00 | T1 1.5R @ 404.69 |
| Target hit | 2025-10-07 12:10:00 | 405.40 | 405.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:15:00 | 400.00 | 397.73 | 0.00 | ORB-long ORB[393.00,397.70] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-10-10 11:30:00 | 399.03 | 397.85 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:05:00 | 394.05 | 396.47 | 0.00 | ORB-short ORB[397.50,401.30] vol=2.8x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:25:00 | 392.98 | 396.08 | 0.00 | T1 1.5R @ 392.98 |
| Stop hit — per-position SL triggered | 2025-10-14 14:40:00 | 394.05 | 394.01 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 393.00 | 394.04 | 0.00 | ORB-short ORB[393.05,394.90] vol=3.4x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-10-15 11:00:00 | 393.79 | 393.98 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 396.70 | 394.71 | 0.00 | ORB-long ORB[390.60,395.70] vol=2.0x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-10-16 09:40:00 | 395.69 | 395.29 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 10:10:00 | 386.15 | 388.47 | 0.00 | ORB-short ORB[387.55,390.00] vol=2.2x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:20:00 | 384.56 | 387.36 | 0.00 | T1 1.5R @ 384.56 |
| Stop hit — per-position SL triggered | 2025-10-27 10:50:00 | 386.15 | 386.21 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 11:00:00 | 383.30 | 381.15 | 0.00 | ORB-long ORB[379.15,382.65] vol=3.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-10-28 11:10:00 | 382.42 | 381.50 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:00:00 | 381.25 | 382.55 | 0.00 | ORB-short ORB[382.50,385.85] vol=1.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-10-30 11:10:00 | 382.18 | 382.52 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:10:00 | 378.20 | 380.20 | 0.00 | ORB-short ORB[379.50,382.65] vol=2.0x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:25:00 | 377.06 | 379.72 | 0.00 | T1 1.5R @ 377.06 |
| Stop hit — per-position SL triggered | 2025-11-06 12:20:00 | 378.20 | 378.38 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:40:00 | 375.55 | 376.39 | 0.00 | ORB-short ORB[375.60,379.25] vol=1.9x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-11-07 09:45:00 | 376.44 | 376.36 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:50:00 | 381.50 | 380.36 | 0.00 | ORB-long ORB[378.40,381.15] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-11-13 10:45:00 | 380.71 | 380.65 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-11-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:00:00 | 376.45 | 377.46 | 0.00 | ORB-short ORB[378.00,380.40] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-11-21 11:20:00 | 377.16 | 377.40 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 11:00:00 | 365.25 | 364.80 | 0.00 | ORB-long ORB[360.10,365.15] vol=3.0x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-11-26 11:25:00 | 364.58 | 364.81 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:45:00 | 365.50 | 365.97 | 0.00 | ORB-short ORB[366.25,368.00] vol=2.4x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:15:00 | 364.30 | 365.84 | 0.00 | T1 1.5R @ 364.30 |
| Stop hit — per-position SL triggered | 2025-11-27 11:00:00 | 365.50 | 365.64 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 11:15:00 | 377.50 | 376.53 | 0.00 | ORB-long ORB[374.80,376.95] vol=2.8x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:25:00 | 378.48 | 376.74 | 0.00 | T1 1.5R @ 378.48 |
| Stop hit — per-position SL triggered | 2025-12-01 11:35:00 | 377.50 | 377.00 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:00:00 | 374.85 | 376.25 | 0.00 | ORB-short ORB[375.85,378.60] vol=1.5x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:15:00 | 373.72 | 375.94 | 0.00 | T1 1.5R @ 373.72 |
| Stop hit — per-position SL triggered | 2025-12-03 10:25:00 | 374.85 | 375.78 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:05:00 | 377.50 | 378.44 | 0.00 | ORB-short ORB[377.75,380.40] vol=1.7x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:50:00 | 376.21 | 378.00 | 0.00 | T1 1.5R @ 376.21 |
| Target hit | 2025-12-08 15:00:00 | 373.10 | 373.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:15:00 | 372.85 | 370.96 | 0.00 | ORB-long ORB[368.80,372.65] vol=7.2x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-12-09 11:20:00 | 371.86 | 370.99 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 370.50 | 372.86 | 0.00 | ORB-short ORB[372.80,374.80] vol=6.2x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-12-10 11:10:00 | 371.35 | 372.72 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:30:00 | 373.30 | 370.49 | 0.00 | ORB-long ORB[369.00,371.45] vol=1.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 372.37 | 370.75 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:50:00 | 368.90 | 369.60 | 0.00 | ORB-short ORB[369.00,371.90] vol=2.2x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 10:00:00 | 367.83 | 369.30 | 0.00 | T1 1.5R @ 367.83 |
| Target hit | 2025-12-16 11:40:00 | 368.50 | 368.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — SELL (started 2025-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:40:00 | 360.10 | 361.14 | 0.00 | ORB-short ORB[360.60,363.70] vol=2.1x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:50:00 | 358.99 | 360.63 | 0.00 | T1 1.5R @ 358.99 |
| Stop hit — per-position SL triggered | 2025-12-18 10:30:00 | 360.10 | 360.09 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-12-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:10:00 | 359.60 | 360.53 | 0.00 | ORB-short ORB[359.80,361.85] vol=3.0x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 360.16 | 360.51 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 366.00 | 365.30 | 0.00 | ORB-long ORB[362.50,365.95] vol=2.4x ATR=0.64 |
| Stop hit — per-position SL triggered | 2026-01-02 10:00:00 | 365.36 | 365.53 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:10:00 | 363.10 | 365.55 | 0.00 | ORB-short ORB[366.00,368.25] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 361.88 | 365.08 | 0.00 | T1 1.5R @ 361.88 |
| Stop hit — per-position SL triggered | 2026-01-06 11:20:00 | 363.10 | 365.01 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:45:00 | 360.95 | 362.18 | 0.00 | ORB-short ORB[362.00,364.25] vol=3.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-01-07 09:55:00 | 361.89 | 362.11 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:35:00 | 357.00 | 358.80 | 0.00 | ORB-short ORB[358.40,361.85] vol=1.6x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:55:00 | 355.68 | 358.36 | 0.00 | T1 1.5R @ 355.68 |
| Target hit | 2026-01-08 15:20:00 | 352.40 | 355.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2026-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:30:00 | 349.25 | 351.22 | 0.00 | ORB-short ORB[350.25,353.75] vol=2.4x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:35:00 | 347.57 | 350.69 | 0.00 | T1 1.5R @ 347.57 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 349.25 | 349.90 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-01-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 11:05:00 | 344.35 | 342.74 | 0.00 | ORB-long ORB[341.20,343.95] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-01-19 11:10:00 | 343.49 | 342.78 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:55:00 | 338.60 | 339.54 | 0.00 | ORB-short ORB[339.30,342.80] vol=2.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2026-01-20 10:00:00 | 339.49 | 339.54 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 11:00:00 | 324.25 | 328.71 | 0.00 | ORB-short ORB[329.30,333.00] vol=2.2x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-01-21 11:10:00 | 325.65 | 328.40 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 09:40:00 | 336.65 | 334.77 | 0.00 | ORB-long ORB[331.50,335.20] vol=2.9x ATR=1.00 |
| Stop hit — per-position SL triggered | 2026-01-23 10:00:00 | 335.65 | 335.52 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:10:00 | 321.85 | 322.93 | 0.00 | ORB-short ORB[322.95,326.00] vol=8.2x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:35:00 | 320.63 | 322.72 | 0.00 | T1 1.5R @ 320.63 |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 321.85 | 322.31 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 325.00 | 323.68 | 0.00 | ORB-long ORB[322.10,324.45] vol=2.6x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-01 11:30:00 | 324.12 | 323.99 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:45:00 | 336.10 | 339.76 | 0.00 | ORB-short ORB[339.10,343.00] vol=1.9x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-02-05 11:05:00 | 337.12 | 339.46 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 338.35 | 336.58 | 0.00 | ORB-long ORB[332.20,336.70] vol=1.9x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 13:10:00 | 340.23 | 337.71 | 0.00 | T1 1.5R @ 340.23 |
| Target hit | 2026-02-16 15:20:00 | 340.35 | 338.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — SELL (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 340.15 | 340.96 | 0.00 | ORB-short ORB[340.30,343.50] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:30:00 | 339.06 | 340.68 | 0.00 | T1 1.5R @ 339.06 |
| Target hit | 2026-02-19 15:20:00 | 333.00 | 336.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 335.30 | 334.27 | 0.00 | ORB-long ORB[332.40,334.85] vol=1.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:20:00 | 336.63 | 334.95 | 0.00 | T1 1.5R @ 336.63 |
| Target hit | 2026-02-20 15:15:00 | 336.20 | 336.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 87 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 335.15 | 336.13 | 0.00 | ORB-short ORB[335.30,338.40] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 335.94 | 335.93 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 339.50 | 338.67 | 0.00 | ORB-long ORB[336.60,339.35] vol=1.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:35:00 | 340.75 | 338.97 | 0.00 | T1 1.5R @ 340.75 |
| Stop hit — per-position SL triggered | 2026-02-25 12:45:00 | 339.50 | 340.48 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 336.30 | 336.42 | 0.00 | ORB-short ORB[336.50,339.60] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-02-27 13:45:00 | 337.01 | 336.36 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 314.00 | 317.94 | 0.00 | ORB-short ORB[316.40,320.65] vol=2.3x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:55:00 | 312.43 | 315.48 | 0.00 | T1 1.5R @ 312.43 |
| Stop hit — per-position SL triggered | 2026-03-11 14:25:00 | 314.00 | 315.24 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 303.80 | 305.59 | 0.00 | ORB-short ORB[305.30,308.80] vol=7.4x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 302.03 | 304.80 | 0.00 | T1 1.5R @ 302.03 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 303.80 | 303.69 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:30:00 | 299.00 | 297.39 | 0.00 | ORB-long ORB[295.25,297.85] vol=3.0x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:35:00 | 300.89 | 298.40 | 0.00 | T1 1.5R @ 300.89 |
| Target hit | 2026-03-17 11:25:00 | 299.45 | 300.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 93 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 307.15 | 305.69 | 0.00 | ORB-long ORB[302.95,306.20] vol=2.5x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 306.13 | 305.98 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 291.80 | 294.66 | 0.00 | ORB-short ORB[294.55,297.65] vol=1.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 292.90 | 294.54 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 292.05 | 295.12 | 0.00 | ORB-short ORB[294.60,298.10] vol=1.7x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-03-30 11:35:00 | 293.18 | 294.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:35:00 | 379.15 | 2025-05-14 10:55:00 | 378.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-15 09:30:00 | 385.75 | 2025-05-15 09:35:00 | 384.63 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-27 11:10:00 | 382.90 | 2025-05-27 11:15:00 | 380.99 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-05-27 11:10:00 | 382.90 | 2025-05-27 12:10:00 | 382.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 11:05:00 | 386.00 | 2025-05-30 11:25:00 | 386.92 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-02 09:45:00 | 391.30 | 2025-06-02 10:20:00 | 390.21 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-04 10:30:00 | 382.90 | 2025-06-04 10:45:00 | 383.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-06 11:10:00 | 393.70 | 2025-06-06 11:25:00 | 392.53 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-09 11:05:00 | 392.45 | 2025-06-09 11:10:00 | 393.43 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-16 09:30:00 | 380.20 | 2025-06-16 09:40:00 | 378.56 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-06-16 09:30:00 | 380.20 | 2025-06-16 10:10:00 | 380.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-19 11:05:00 | 376.00 | 2025-06-19 11:35:00 | 374.61 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-06-19 11:05:00 | 376.00 | 2025-06-19 14:00:00 | 374.95 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-06-20 10:25:00 | 379.70 | 2025-06-20 14:55:00 | 381.37 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-20 10:25:00 | 379.70 | 2025-06-20 15:20:00 | 380.05 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-06-24 09:35:00 | 384.65 | 2025-06-24 09:40:00 | 386.10 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-24 09:35:00 | 384.65 | 2025-06-24 09:45:00 | 384.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 09:30:00 | 389.60 | 2025-06-25 09:45:00 | 388.69 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-26 10:00:00 | 384.60 | 2025-06-26 10:10:00 | 385.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-30 11:00:00 | 390.20 | 2025-06-30 11:40:00 | 391.74 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-30 11:00:00 | 390.20 | 2025-06-30 11:45:00 | 390.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 10:05:00 | 384.60 | 2025-07-02 10:15:00 | 383.26 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-02 10:05:00 | 384.60 | 2025-07-02 11:40:00 | 384.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-03 10:15:00 | 382.60 | 2025-07-03 10:25:00 | 383.46 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-08 11:05:00 | 383.25 | 2025-07-08 11:25:00 | 383.92 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-09 11:15:00 | 389.05 | 2025-07-09 12:00:00 | 388.13 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-10 10:35:00 | 389.70 | 2025-07-10 10:40:00 | 388.82 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-11 10:10:00 | 385.90 | 2025-07-11 10:30:00 | 384.67 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-11 10:10:00 | 385.90 | 2025-07-11 15:20:00 | 380.00 | TARGET_HIT | 0.50 | 1.53% |
| SELL | retest1 | 2025-07-18 10:00:00 | 381.50 | 2025-07-18 10:15:00 | 380.16 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-18 10:00:00 | 381.50 | 2025-07-18 10:35:00 | 381.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 10:10:00 | 388.20 | 2025-07-21 10:45:00 | 387.13 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-22 09:45:00 | 392.80 | 2025-07-22 10:00:00 | 394.39 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-07-22 09:45:00 | 392.80 | 2025-07-22 10:20:00 | 392.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 09:35:00 | 395.10 | 2025-07-24 09:50:00 | 394.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-25 09:40:00 | 388.45 | 2025-07-25 09:45:00 | 386.87 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-25 09:40:00 | 388.45 | 2025-07-25 15:20:00 | 380.80 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2025-07-28 10:30:00 | 384.20 | 2025-07-28 10:45:00 | 383.19 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-29 09:35:00 | 385.30 | 2025-07-29 09:45:00 | 386.88 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-07-29 09:35:00 | 385.30 | 2025-07-29 15:20:00 | 391.55 | TARGET_HIT | 0.50 | 1.62% |
| SELL | retest1 | 2025-08-01 09:45:00 | 380.00 | 2025-08-01 10:30:00 | 381.27 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-13 10:35:00 | 376.20 | 2025-08-13 11:05:00 | 376.85 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-08-14 09:30:00 | 374.55 | 2025-08-14 09:45:00 | 375.17 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-08-19 09:50:00 | 380.35 | 2025-08-19 10:35:00 | 379.49 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-25 10:55:00 | 398.05 | 2025-08-25 11:25:00 | 399.51 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-08-25 10:55:00 | 398.05 | 2025-08-25 15:20:00 | 400.50 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2025-09-01 09:40:00 | 404.70 | 2025-09-01 09:55:00 | 403.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-02 10:50:00 | 420.90 | 2025-09-02 11:25:00 | 419.51 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-05 10:05:00 | 413.10 | 2025-09-05 10:10:00 | 412.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-08 09:30:00 | 417.40 | 2025-09-08 09:35:00 | 416.30 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-10 10:50:00 | 430.85 | 2025-09-10 10:55:00 | 429.86 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-12 09:35:00 | 426.20 | 2025-09-12 09:40:00 | 425.26 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-16 09:35:00 | 420.30 | 2025-09-16 10:25:00 | 419.40 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-18 09:50:00 | 421.20 | 2025-09-18 10:10:00 | 420.31 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-19 09:55:00 | 421.70 | 2025-09-19 10:05:00 | 422.59 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-22 11:00:00 | 411.65 | 2025-09-22 11:10:00 | 412.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-24 10:45:00 | 399.80 | 2025-09-24 11:00:00 | 398.44 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-24 10:45:00 | 399.80 | 2025-09-24 11:35:00 | 399.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-26 09:45:00 | 399.85 | 2025-09-26 09:50:00 | 398.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-29 11:15:00 | 388.70 | 2025-09-29 11:30:00 | 387.44 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-29 11:15:00 | 388.70 | 2025-09-29 12:25:00 | 388.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-30 11:10:00 | 390.00 | 2025-09-30 11:15:00 | 389.09 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-01 10:35:00 | 394.45 | 2025-10-01 10:40:00 | 393.47 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-07 09:45:00 | 403.00 | 2025-10-07 10:05:00 | 404.69 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-07 09:45:00 | 403.00 | 2025-10-07 12:10:00 | 405.40 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-10-10 11:15:00 | 400.00 | 2025-10-10 11:30:00 | 399.03 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-14 11:05:00 | 394.05 | 2025-10-14 11:25:00 | 392.98 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-14 11:05:00 | 394.05 | 2025-10-14 14:40:00 | 394.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-15 10:55:00 | 393.00 | 2025-10-15 11:00:00 | 393.79 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-16 09:30:00 | 396.70 | 2025-10-16 09:40:00 | 395.69 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-27 10:10:00 | 386.15 | 2025-10-27 10:20:00 | 384.56 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-27 10:10:00 | 386.15 | 2025-10-27 10:50:00 | 386.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 11:00:00 | 383.30 | 2025-10-28 11:10:00 | 382.42 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-30 11:00:00 | 381.25 | 2025-10-30 11:10:00 | 382.18 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-06 10:10:00 | 378.20 | 2025-11-06 10:25:00 | 377.06 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-06 10:10:00 | 378.20 | 2025-11-06 12:20:00 | 378.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:40:00 | 375.55 | 2025-11-07 09:45:00 | 376.44 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-13 09:50:00 | 381.50 | 2025-11-13 10:45:00 | 380.71 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-21 11:00:00 | 376.45 | 2025-11-21 11:20:00 | 377.16 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-26 11:00:00 | 365.25 | 2025-11-26 11:25:00 | 364.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-27 09:45:00 | 365.50 | 2025-11-27 10:15:00 | 364.30 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-27 09:45:00 | 365.50 | 2025-11-27 11:00:00 | 365.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 11:15:00 | 377.50 | 2025-12-01 11:25:00 | 378.48 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-12-01 11:15:00 | 377.50 | 2025-12-01 11:35:00 | 377.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:00:00 | 374.85 | 2025-12-03 10:15:00 | 373.72 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-03 10:00:00 | 374.85 | 2025-12-03 10:25:00 | 374.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:05:00 | 377.50 | 2025-12-08 10:50:00 | 376.21 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-08 10:05:00 | 377.50 | 2025-12-08 15:00:00 | 373.10 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2025-12-09 11:15:00 | 372.85 | 2025-12-09 11:20:00 | 371.86 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-10 10:55:00 | 370.50 | 2025-12-10 11:10:00 | 371.35 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-11 10:30:00 | 373.30 | 2025-12-11 10:55:00 | 372.37 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-16 09:50:00 | 368.90 | 2025-12-16 10:00:00 | 367.83 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-16 09:50:00 | 368.90 | 2025-12-16 11:40:00 | 368.50 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-12-18 09:40:00 | 360.10 | 2025-12-18 09:50:00 | 358.99 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-18 09:40:00 | 360.10 | 2025-12-18 10:30:00 | 360.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 11:10:00 | 359.60 | 2025-12-30 11:15:00 | 360.16 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-01-02 09:30:00 | 366.00 | 2026-01-02 10:00:00 | 365.36 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-01-06 11:10:00 | 363.10 | 2026-01-06 11:15:00 | 361.88 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-06 11:10:00 | 363.10 | 2026-01-06 11:20:00 | 363.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:45:00 | 360.95 | 2026-01-07 09:55:00 | 361.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-08 10:35:00 | 357.00 | 2026-01-08 10:55:00 | 355.68 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-08 10:35:00 | 357.00 | 2026-01-08 15:20:00 | 352.40 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2026-01-09 09:30:00 | 349.25 | 2026-01-09 09:35:00 | 347.57 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-01-09 09:30:00 | 349.25 | 2026-01-09 09:45:00 | 349.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-19 11:05:00 | 344.35 | 2026-01-19 11:10:00 | 343.49 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-20 09:55:00 | 338.60 | 2026-01-20 10:00:00 | 339.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-21 11:00:00 | 324.25 | 2026-01-21 11:10:00 | 325.65 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-01-23 09:40:00 | 336.65 | 2026-01-23 10:00:00 | 335.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-28 11:10:00 | 321.85 | 2026-01-28 11:35:00 | 320.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-28 11:10:00 | 321.85 | 2026-01-28 13:15:00 | 321.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:10:00 | 325.00 | 2026-02-01 11:30:00 | 324.12 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-05 10:45:00 | 336.10 | 2026-02-05 11:05:00 | 337.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-16 10:15:00 | 338.35 | 2026-02-16 13:10:00 | 340.23 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-16 10:15:00 | 338.35 | 2026-02-16 15:20:00 | 340.35 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-19 11:00:00 | 340.15 | 2026-02-19 11:30:00 | 339.06 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-19 11:00:00 | 340.15 | 2026-02-19 15:20:00 | 333.00 | TARGET_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2026-02-20 10:50:00 | 335.30 | 2026-02-20 12:20:00 | 336.63 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-20 10:50:00 | 335.30 | 2026-02-20 15:15:00 | 336.20 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-24 09:30:00 | 335.15 | 2026-02-24 09:45:00 | 335.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:15:00 | 339.50 | 2026-02-25 10:35:00 | 340.75 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-25 10:15:00 | 339.50 | 2026-02-25 12:45:00 | 339.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 11:10:00 | 336.30 | 2026-02-27 13:45:00 | 337.01 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-11 10:40:00 | 314.00 | 2026-03-11 13:55:00 | 312.43 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-11 10:40:00 | 314.00 | 2026-03-11 14:25:00 | 314.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:50:00 | 303.80 | 2026-03-13 10:15:00 | 302.03 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 09:50:00 | 303.80 | 2026-03-13 10:50:00 | 303.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 09:30:00 | 299.00 | 2026-03-17 09:35:00 | 300.89 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-03-17 09:30:00 | 299.00 | 2026-03-17 11:25:00 | 299.45 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-03-18 09:30:00 | 307.15 | 2026-03-18 09:45:00 | 306.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-24 10:55:00 | 291.80 | 2026-03-24 11:15:00 | 292.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-30 10:55:00 | 292.05 | 2026-03-30 11:35:00 | 293.18 | STOP_HIT | 1.00 | -0.39% |
