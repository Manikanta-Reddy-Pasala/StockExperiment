# Piramal Pharma Ltd. (PPLPHARMA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 179.58
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 16 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 41
- **Target hits / Stop hits / Partials:** 16 / 41 / 29
- **Avg / median % per leg:** 0.32% / 0.10%
- **Sum % (uncompounded):** 27.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 25 | 52.1% | 9 | 23 | 16 | 0.36% | 17.3% |
| BUY @ 2nd Alert (retest1) | 48 | 25 | 52.1% | 9 | 23 | 16 | 0.36% | 17.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 20 | 52.6% | 7 | 18 | 13 | 0.26% | 9.9% |
| SELL @ 2nd Alert (retest1) | 38 | 20 | 52.6% | 7 | 18 | 13 | 0.26% | 9.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 86 | 45 | 52.3% | 16 | 41 | 29 | 0.32% | 27.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 11:00:00 | 149.50 | 150.24 | 0.00 | ORB-short ORB[149.60,151.50] vol=2.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-05-17 11:05:00 | 149.99 | 150.22 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 148.70 | 149.78 | 0.00 | ORB-short ORB[149.30,151.50] vol=3.9x ATR=0.51 |
| Stop hit — per-position SL triggered | 2024-05-21 10:10:00 | 149.21 | 149.62 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-22 10:10:00 | 149.50 | 148.61 | 0.00 | ORB-long ORB[147.65,149.45] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-05-22 10:20:00 | 148.86 | 148.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:05:00 | 150.50 | 149.19 | 0.00 | ORB-long ORB[147.90,149.65] vol=2.1x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 149.78 | 149.27 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 11:10:00 | 150.45 | 149.39 | 0.00 | ORB-long ORB[148.15,150.00] vol=3.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 150.00 | 149.41 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:25:00 | 148.60 | 149.05 | 0.00 | ORB-short ORB[148.90,150.40] vol=2.3x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:50:00 | 148.02 | 148.88 | 0.00 | T1 1.5R @ 148.02 |
| Target hit | 2024-05-30 15:20:00 | 145.70 | 147.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 150.40 | 149.70 | 0.00 | ORB-long ORB[148.45,149.90] vol=3.7x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 09:40:00 | 151.13 | 150.04 | 0.00 | T1 1.5R @ 151.13 |
| Target hit | 2024-06-07 10:20:00 | 151.05 | 151.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 152.11 | 151.52 | 0.00 | ORB-long ORB[150.05,151.95] vol=3.6x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:40:00 | 153.01 | 151.93 | 0.00 | T1 1.5R @ 153.01 |
| Target hit | 2024-06-10 15:20:00 | 156.93 | 155.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 11:10:00 | 154.03 | 154.74 | 0.00 | ORB-short ORB[154.10,155.90] vol=1.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-06-12 12:15:00 | 154.43 | 154.60 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 153.20 | 153.76 | 0.00 | ORB-short ORB[153.60,154.54] vol=1.8x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:50:00 | 152.58 | 153.41 | 0.00 | T1 1.5R @ 152.58 |
| Stop hit — per-position SL triggered | 2024-06-13 10:00:00 | 153.20 | 153.34 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 154.55 | 153.78 | 0.00 | ORB-long ORB[152.90,153.82] vol=1.8x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:10:00 | 155.32 | 154.17 | 0.00 | T1 1.5R @ 155.32 |
| Target hit | 2024-06-14 13:50:00 | 154.64 | 154.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 155.49 | 154.51 | 0.00 | ORB-long ORB[153.31,155.10] vol=2.2x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 09:40:00 | 156.27 | 155.13 | 0.00 | T1 1.5R @ 156.27 |
| Stop hit — per-position SL triggered | 2024-06-20 10:10:00 | 155.49 | 155.71 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 154.30 | 154.99 | 0.00 | ORB-short ORB[154.31,155.90] vol=2.9x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:20:00 | 153.42 | 154.40 | 0.00 | T1 1.5R @ 153.42 |
| Stop hit — per-position SL triggered | 2024-06-21 10:30:00 | 154.30 | 154.38 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:05:00 | 159.28 | 157.11 | 0.00 | ORB-long ORB[154.50,156.59] vol=4.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-06-24 10:20:00 | 158.37 | 157.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-06-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:45:00 | 159.01 | 157.87 | 0.00 | ORB-long ORB[156.20,157.45] vol=2.9x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:50:00 | 159.82 | 158.78 | 0.00 | T1 1.5R @ 159.82 |
| Stop hit — per-position SL triggered | 2024-06-25 11:25:00 | 159.01 | 159.23 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:15:00 | 159.22 | 158.19 | 0.00 | ORB-long ORB[156.80,158.22] vol=2.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-06-28 10:35:00 | 158.64 | 158.30 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:35:00 | 157.81 | 157.27 | 0.00 | ORB-long ORB[156.27,157.79] vol=2.3x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:40:00 | 158.56 | 157.39 | 0.00 | T1 1.5R @ 158.56 |
| Stop hit — per-position SL triggered | 2024-07-01 12:05:00 | 157.81 | 158.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 156.29 | 157.59 | 0.00 | ORB-short ORB[157.52,159.50] vol=1.7x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 12:55:00 | 155.48 | 157.07 | 0.00 | T1 1.5R @ 155.48 |
| Target hit | 2024-07-08 15:20:00 | 153.91 | 156.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:40:00 | 154.84 | 154.16 | 0.00 | ORB-long ORB[153.45,154.79] vol=2.5x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-07-09 10:20:00 | 154.29 | 154.41 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:50:00 | 154.10 | 155.04 | 0.00 | ORB-short ORB[155.01,156.93] vol=2.2x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:05:00 | 153.34 | 154.60 | 0.00 | T1 1.5R @ 153.34 |
| Target hit | 2024-07-10 11:35:00 | 152.60 | 152.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — BUY (started 2024-07-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 09:35:00 | 153.82 | 153.41 | 0.00 | ORB-long ORB[152.84,153.65] vol=1.9x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-07-11 09:45:00 | 153.37 | 153.44 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:15:00 | 151.28 | 151.64 | 0.00 | ORB-short ORB[151.30,152.34] vol=1.5x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:50:00 | 150.64 | 151.50 | 0.00 | T1 1.5R @ 150.64 |
| Target hit | 2024-07-12 13:30:00 | 151.13 | 151.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:15:00 | 153.80 | 152.95 | 0.00 | ORB-long ORB[152.30,153.42] vol=1.5x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:20:00 | 154.53 | 153.44 | 0.00 | T1 1.5R @ 154.53 |
| Target hit | 2024-07-16 12:45:00 | 154.37 | 154.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 150.58 | 152.11 | 0.00 | ORB-short ORB[151.57,153.70] vol=2.2x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 151.13 | 151.82 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:50:00 | 147.89 | 148.52 | 0.00 | ORB-short ORB[148.10,149.94] vol=1.7x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:00:00 | 147.16 | 148.26 | 0.00 | T1 1.5R @ 147.16 |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 147.89 | 147.95 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-07-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:55:00 | 152.48 | 153.32 | 0.00 | ORB-short ORB[153.00,154.50] vol=1.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-07-23 10:15:00 | 153.37 | 153.53 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-07-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:00:00 | 170.50 | 168.77 | 0.00 | ORB-long ORB[168.00,169.85] vol=3.9x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:05:00 | 171.64 | 169.22 | 0.00 | T1 1.5R @ 171.64 |
| Stop hit — per-position SL triggered | 2024-07-26 11:10:00 | 170.50 | 169.30 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:45:00 | 171.54 | 170.74 | 0.00 | ORB-long ORB[169.05,171.20] vol=3.5x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:50:00 | 172.72 | 172.24 | 0.00 | T1 1.5R @ 172.72 |
| Target hit | 2024-07-30 13:45:00 | 174.47 | 174.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2024-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 11:00:00 | 172.11 | 170.78 | 0.00 | ORB-long ORB[169.01,171.19] vol=4.1x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 11:20:00 | 173.22 | 171.29 | 0.00 | T1 1.5R @ 173.22 |
| Target hit | 2024-08-07 15:20:00 | 179.85 | 176.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2024-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:55:00 | 183.90 | 185.14 | 0.00 | ORB-short ORB[184.12,186.70] vol=3.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-08-09 11:05:00 | 184.72 | 185.13 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 09:30:00 | 185.34 | 183.92 | 0.00 | ORB-long ORB[181.95,184.59] vol=3.4x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:35:00 | 186.75 | 185.25 | 0.00 | T1 1.5R @ 186.75 |
| Stop hit — per-position SL triggered | 2024-08-12 09:55:00 | 185.34 | 185.47 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:00:00 | 188.65 | 187.40 | 0.00 | ORB-long ORB[185.60,188.18] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 187.86 | 187.49 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 186.76 | 186.08 | 0.00 | ORB-long ORB[184.79,186.67] vol=3.4x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-08-21 10:00:00 | 186.15 | 186.45 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:35:00 | 189.49 | 188.49 | 0.00 | ORB-long ORB[187.34,188.85] vol=2.9x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-08-22 09:55:00 | 188.76 | 188.84 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-08-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:35:00 | 187.53 | 188.13 | 0.00 | ORB-short ORB[187.65,189.69] vol=1.7x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 188.21 | 188.08 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:45:00 | 186.84 | 187.96 | 0.00 | ORB-short ORB[187.00,189.79] vol=1.9x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:15:00 | 185.46 | 187.07 | 0.00 | T1 1.5R @ 185.46 |
| Stop hit — per-position SL triggered | 2024-08-26 14:55:00 | 186.84 | 185.92 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-08-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:45:00 | 184.80 | 186.01 | 0.00 | ORB-short ORB[185.47,187.44] vol=1.8x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-08-28 10:15:00 | 185.60 | 185.80 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-08-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:55:00 | 183.58 | 185.32 | 0.00 | ORB-short ORB[184.00,186.66] vol=1.7x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 14:15:00 | 182.61 | 184.70 | 0.00 | T1 1.5R @ 182.61 |
| Target hit | 2024-08-29 15:20:00 | 182.93 | 184.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:40:00 | 184.68 | 183.86 | 0.00 | ORB-long ORB[182.30,184.25] vol=2.5x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 09:45:00 | 185.73 | 184.14 | 0.00 | T1 1.5R @ 185.73 |
| Target hit | 2024-08-30 10:30:00 | 187.75 | 187.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2024-09-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:30:00 | 188.18 | 188.63 | 0.00 | ORB-short ORB[188.59,190.70] vol=5.1x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:20:00 | 187.00 | 188.54 | 0.00 | T1 1.5R @ 187.00 |
| Stop hit — per-position SL triggered | 2024-09-02 12:50:00 | 188.18 | 188.34 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 224.39 | 223.09 | 0.00 | ORB-long ORB[221.35,223.30] vol=2.2x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-10-14 09:35:00 | 223.50 | 223.18 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:55:00 | 255.00 | 255.75 | 0.00 | ORB-short ORB[256.15,259.70] vol=5.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-12-11 12:00:00 | 256.07 | 255.65 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:00:00 | 254.90 | 256.64 | 0.00 | ORB-short ORB[255.55,258.35] vol=3.6x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:05:00 | 253.34 | 255.94 | 0.00 | T1 1.5R @ 253.34 |
| Target hit | 2024-12-12 15:20:00 | 251.75 | 254.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:00:00 | 267.15 | 265.04 | 0.00 | ORB-long ORB[263.00,266.90] vol=3.4x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-01-01 11:05:00 | 265.87 | 265.15 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:20:00 | 259.20 | 257.99 | 0.00 | ORB-long ORB[256.30,258.80] vol=3.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-01-03 10:25:00 | 258.28 | 258.03 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 249.35 | 252.52 | 0.00 | ORB-short ORB[251.95,255.65] vol=2.5x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:15:00 | 248.09 | 252.19 | 0.00 | T1 1.5R @ 248.09 |
| Stop hit — per-position SL triggered | 2025-01-06 11:25:00 | 249.35 | 251.93 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 248.00 | 246.80 | 0.00 | ORB-long ORB[244.95,247.80] vol=2.7x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 09:40:00 | 249.61 | 247.48 | 0.00 | T1 1.5R @ 249.61 |
| Stop hit — per-position SL triggered | 2025-01-09 09:50:00 | 248.00 | 247.68 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-01-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:05:00 | 236.40 | 234.47 | 0.00 | ORB-long ORB[232.75,235.95] vol=2.1x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-01-16 10:10:00 | 235.27 | 234.59 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 234.34 | 235.94 | 0.00 | ORB-short ORB[235.00,238.23] vol=1.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-02-01 11:10:00 | 235.28 | 235.91 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-02-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:50:00 | 233.10 | 231.78 | 0.00 | ORB-long ORB[229.11,231.77] vol=1.9x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 10:15:00 | 234.77 | 232.63 | 0.00 | T1 1.5R @ 234.77 |
| Target hit | 2025-02-05 11:10:00 | 233.33 | 233.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — BUY (started 2025-03-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:45:00 | 203.07 | 202.63 | 0.00 | ORB-long ORB[200.00,202.80] vol=3.4x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-03-07 10:50:00 | 202.17 | 202.65 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 202.88 | 204.08 | 0.00 | ORB-short ORB[203.33,205.39] vol=2.4x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 09:55:00 | 201.28 | 203.65 | 0.00 | T1 1.5R @ 201.28 |
| Target hit | 2025-03-12 13:00:00 | 199.98 | 199.64 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — BUY (started 2025-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:25:00 | 205.72 | 204.48 | 0.00 | ORB-long ORB[202.93,205.40] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-03-17 11:00:00 | 204.70 | 204.58 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 208.00 | 206.84 | 0.00 | ORB-long ORB[205.35,207.75] vol=1.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:10:00 | 209.31 | 207.52 | 0.00 | T1 1.5R @ 209.31 |
| Target hit | 2025-03-18 15:20:00 | 213.17 | 209.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:50:00 | 222.45 | 221.11 | 0.00 | ORB-long ORB[219.70,221.98] vol=2.6x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:25:00 | 223.54 | 221.83 | 0.00 | T1 1.5R @ 223.54 |
| Stop hit — per-position SL triggered | 2025-04-22 11:25:00 | 222.45 | 222.22 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 11:00:00 | 222.63 | 224.83 | 0.00 | ORB-short ORB[226.06,228.45] vol=1.9x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-04-23 11:10:00 | 223.62 | 224.75 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 215.15 | 216.26 | 0.00 | ORB-short ORB[215.30,218.24] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 215.95 | 215.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-17 11:00:00 | 149.50 | 2024-05-17 11:05:00 | 149.99 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-21 10:00:00 | 148.70 | 2024-05-21 10:10:00 | 149.21 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-22 10:10:00 | 149.50 | 2024-05-22 10:20:00 | 148.86 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-27 10:05:00 | 150.50 | 2024-05-27 10:15:00 | 149.78 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-05-29 11:10:00 | 150.45 | 2024-05-29 11:15:00 | 150.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-30 10:25:00 | 148.60 | 2024-05-30 10:50:00 | 148.02 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-30 10:25:00 | 148.60 | 2024-05-30 15:20:00 | 145.70 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2024-06-07 09:30:00 | 150.40 | 2024-06-07 09:40:00 | 151.13 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-07 09:30:00 | 150.40 | 2024-06-07 10:20:00 | 151.05 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-10 09:30:00 | 152.11 | 2024-06-10 09:40:00 | 153.01 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-10 09:30:00 | 152.11 | 2024-06-10 15:20:00 | 156.93 | TARGET_HIT | 0.50 | 3.17% |
| SELL | retest1 | 2024-06-12 11:10:00 | 154.03 | 2024-06-12 12:15:00 | 154.43 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-13 09:35:00 | 153.20 | 2024-06-13 09:50:00 | 152.58 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-06-13 09:35:00 | 153.20 | 2024-06-13 10:00:00 | 153.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-14 09:55:00 | 154.55 | 2024-06-14 11:10:00 | 155.32 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-14 09:55:00 | 154.55 | 2024-06-14 13:50:00 | 154.64 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2024-06-20 09:30:00 | 155.49 | 2024-06-20 09:40:00 | 156.27 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-20 09:30:00 | 155.49 | 2024-06-20 10:10:00 | 155.49 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 09:30:00 | 154.30 | 2024-06-21 10:20:00 | 153.42 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-21 09:30:00 | 154.30 | 2024-06-21 10:30:00 | 154.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 10:05:00 | 159.28 | 2024-06-24 10:20:00 | 158.37 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-06-25 10:45:00 | 159.01 | 2024-06-25 10:50:00 | 159.82 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-06-25 10:45:00 | 159.01 | 2024-06-25 11:25:00 | 159.01 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 10:15:00 | 159.22 | 2024-06-28 10:35:00 | 158.64 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-01 09:35:00 | 157.81 | 2024-07-01 09:40:00 | 158.56 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-01 09:35:00 | 157.81 | 2024-07-01 12:05:00 | 157.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 11:10:00 | 156.29 | 2024-07-08 12:55:00 | 155.48 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-08 11:10:00 | 156.29 | 2024-07-08 15:20:00 | 153.91 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-07-09 09:40:00 | 154.84 | 2024-07-09 10:20:00 | 154.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-10 09:50:00 | 154.10 | 2024-07-10 10:05:00 | 153.34 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-07-10 09:50:00 | 154.10 | 2024-07-10 11:35:00 | 152.60 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-07-11 09:35:00 | 153.82 | 2024-07-11 09:45:00 | 153.37 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-12 10:15:00 | 151.28 | 2024-07-12 10:50:00 | 150.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-07-12 10:15:00 | 151.28 | 2024-07-12 13:30:00 | 151.13 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-07-16 10:15:00 | 153.80 | 2024-07-16 10:20:00 | 154.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-16 10:15:00 | 153.80 | 2024-07-16 12:45:00 | 154.37 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-07-18 09:30:00 | 150.58 | 2024-07-18 09:40:00 | 151.13 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-19 09:50:00 | 147.89 | 2024-07-19 10:00:00 | 147.16 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-19 09:50:00 | 147.89 | 2024-07-19 10:15:00 | 147.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 09:55:00 | 152.48 | 2024-07-23 10:15:00 | 153.37 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-07-26 11:00:00 | 170.50 | 2024-07-26 11:05:00 | 171.64 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-07-26 11:00:00 | 170.50 | 2024-07-26 11:10:00 | 170.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 09:45:00 | 171.54 | 2024-07-30 09:50:00 | 172.72 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-07-30 09:45:00 | 171.54 | 2024-07-30 13:45:00 | 174.47 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2024-08-07 11:00:00 | 172.11 | 2024-08-07 11:20:00 | 173.22 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-08-07 11:00:00 | 172.11 | 2024-08-07 15:20:00 | 179.85 | TARGET_HIT | 0.50 | 4.50% |
| SELL | retest1 | 2024-08-09 10:55:00 | 183.90 | 2024-08-09 11:05:00 | 184.72 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-12 09:30:00 | 185.34 | 2024-08-12 09:35:00 | 186.75 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-08-12 09:30:00 | 185.34 | 2024-08-12 09:55:00 | 185.34 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 10:00:00 | 188.65 | 2024-08-13 10:15:00 | 187.86 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-21 09:30:00 | 186.76 | 2024-08-21 10:00:00 | 186.15 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-22 09:35:00 | 189.49 | 2024-08-22 09:55:00 | 188.76 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-23 09:35:00 | 187.53 | 2024-08-23 10:15:00 | 188.21 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-26 09:45:00 | 186.84 | 2024-08-26 11:15:00 | 185.46 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-08-26 09:45:00 | 186.84 | 2024-08-26 14:55:00 | 186.84 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:45:00 | 184.80 | 2024-08-28 10:15:00 | 185.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-08-29 10:55:00 | 183.58 | 2024-08-29 14:15:00 | 182.61 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-08-29 10:55:00 | 183.58 | 2024-08-29 15:20:00 | 182.93 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-30 09:40:00 | 184.68 | 2024-08-30 09:45:00 | 185.73 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-30 09:40:00 | 184.68 | 2024-08-30 10:30:00 | 187.75 | TARGET_HIT | 0.50 | 1.66% |
| SELL | retest1 | 2024-09-02 10:30:00 | 188.18 | 2024-09-02 11:20:00 | 187.00 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-09-02 10:30:00 | 188.18 | 2024-09-02 12:50:00 | 188.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 09:30:00 | 224.39 | 2024-10-14 09:35:00 | 223.50 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-11 10:55:00 | 255.00 | 2024-12-11 12:00:00 | 256.07 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-12-12 11:00:00 | 254.90 | 2024-12-12 12:05:00 | 253.34 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-12-12 11:00:00 | 254.90 | 2024-12-12 15:20:00 | 251.75 | TARGET_HIT | 0.50 | 1.24% |
| BUY | retest1 | 2025-01-01 11:00:00 | 267.15 | 2025-01-01 11:05:00 | 265.87 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-01-03 10:20:00 | 259.20 | 2025-01-03 10:25:00 | 258.28 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-06 11:10:00 | 249.35 | 2025-01-06 11:15:00 | 248.09 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-01-06 11:10:00 | 249.35 | 2025-01-06 11:25:00 | 249.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 09:30:00 | 248.00 | 2025-01-09 09:40:00 | 249.61 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-01-09 09:30:00 | 248.00 | 2025-01-09 09:50:00 | 248.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-16 10:05:00 | 236.40 | 2025-01-16 10:10:00 | 235.27 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-01 11:00:00 | 234.34 | 2025-02-01 11:10:00 | 235.28 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-05 09:50:00 | 233.10 | 2025-02-05 10:15:00 | 234.77 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-02-05 09:50:00 | 233.10 | 2025-02-05 11:10:00 | 233.33 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-03-07 10:45:00 | 203.07 | 2025-03-07 10:50:00 | 202.17 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-03-12 09:30:00 | 202.88 | 2025-03-12 09:55:00 | 201.28 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2025-03-12 09:30:00 | 202.88 | 2025-03-12 13:00:00 | 199.98 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2025-03-17 10:25:00 | 205.72 | 2025-03-17 11:00:00 | 204.70 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-18 09:35:00 | 208.00 | 2025-03-18 11:10:00 | 209.31 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-03-18 09:35:00 | 208.00 | 2025-03-18 15:20:00 | 213.17 | TARGET_HIT | 0.50 | 2.49% |
| BUY | retest1 | 2025-04-22 09:50:00 | 222.45 | 2025-04-22 10:25:00 | 223.54 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-22 09:50:00 | 222.45 | 2025-04-22 11:25:00 | 222.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 11:00:00 | 222.63 | 2025-04-23 11:10:00 | 223.62 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-04-29 09:35:00 | 215.15 | 2025-04-29 09:55:00 | 215.95 | STOP_HIT | 1.00 | -0.37% |
