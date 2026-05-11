# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35146 bars)
- **Last close:** 214.60
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 10 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 55
- **Target hits / Stop hits / Partials:** 10 / 55 / 25
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 10.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 15 | 30.6% | 4 | 34 | 11 | 0.09% | 4.3% |
| BUY @ 2nd Alert (retest1) | 49 | 15 | 30.6% | 4 | 34 | 11 | 0.09% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 20 | 48.8% | 6 | 21 | 14 | 0.15% | 6.3% |
| SELL @ 2nd Alert (retest1) | 41 | 20 | 48.8% | 6 | 21 | 14 | 0.15% | 6.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 35 | 38.9% | 10 | 55 | 25 | 0.12% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:35:00 | 165.65 | 164.62 | 0.00 | ORB-long ORB[163.50,164.95] vol=2.3x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-05-14 09:45:00 | 165.09 | 164.76 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 164.10 | 166.44 | 0.00 | ORB-short ORB[166.70,168.00] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-05-16 11:40:00 | 164.53 | 166.16 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:35:00 | 170.35 | 169.58 | 0.00 | ORB-long ORB[167.95,170.20] vol=3.1x ATR=0.45 |
| Stop hit — per-position SL triggered | 2024-05-21 09:45:00 | 169.90 | 169.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:25:00 | 182.66 | 181.61 | 0.00 | ORB-long ORB[180.50,182.04] vol=1.7x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:35:00 | 183.48 | 181.82 | 0.00 | T1 1.5R @ 183.48 |
| Stop hit — per-position SL triggered | 2024-06-11 11:00:00 | 182.66 | 182.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:05:00 | 181.86 | 182.97 | 0.00 | ORB-short ORB[182.84,184.10] vol=2.1x ATR=0.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:20:00 | 181.33 | 182.77 | 0.00 | T1 1.5R @ 181.33 |
| Stop hit — per-position SL triggered | 2024-06-13 11:50:00 | 181.86 | 182.44 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:40:00 | 182.92 | 183.72 | 0.00 | ORB-short ORB[183.50,184.60] vol=1.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-06-18 09:45:00 | 183.35 | 183.69 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 11:00:00 | 182.59 | 181.10 | 0.00 | ORB-long ORB[179.37,181.65] vol=2.0x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-06-20 11:10:00 | 182.12 | 181.20 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:50:00 | 180.22 | 179.68 | 0.00 | ORB-long ORB[178.85,180.00] vol=2.5x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-06-21 10:45:00 | 179.57 | 179.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:55:00 | 172.80 | 173.83 | 0.00 | ORB-short ORB[173.60,176.00] vol=1.6x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:50:00 | 172.09 | 173.20 | 0.00 | T1 1.5R @ 172.09 |
| Stop hit — per-position SL triggered | 2024-06-26 11:45:00 | 172.80 | 172.96 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:55:00 | 175.68 | 174.51 | 0.00 | ORB-long ORB[173.60,174.80] vol=2.0x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-07-02 11:05:00 | 175.28 | 174.59 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 169.92 | 171.35 | 0.00 | ORB-short ORB[171.61,172.89] vol=2.8x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 169.26 | 170.92 | 0.00 | T1 1.5R @ 169.26 |
| Target hit | 2024-07-10 15:20:00 | 167.88 | 168.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 169.50 | 168.44 | 0.00 | ORB-long ORB[167.10,168.47] vol=1.5x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 169.10 | 168.82 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 160.80 | 159.96 | 0.00 | ORB-long ORB[159.19,160.36] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-07-24 09:40:00 | 160.26 | 160.17 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:40:00 | 161.86 | 160.44 | 0.00 | ORB-long ORB[158.21,160.00] vol=1.8x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:00:00 | 162.48 | 160.76 | 0.00 | T1 1.5R @ 162.48 |
| Stop hit — per-position SL triggered | 2024-07-26 12:05:00 | 161.86 | 161.21 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:15:00 | 164.06 | 162.83 | 0.00 | ORB-long ORB[161.31,162.90] vol=1.5x ATR=0.38 |
| Stop hit — per-position SL triggered | 2024-07-30 11:40:00 | 163.68 | 163.11 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 11:10:00 | 163.82 | 164.63 | 0.00 | ORB-short ORB[164.50,166.50] vol=1.8x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-07-31 11:45:00 | 164.22 | 164.54 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:35:00 | 163.97 | 167.13 | 0.00 | ORB-short ORB[166.56,168.80] vol=1.6x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:00:00 | 162.98 | 166.48 | 0.00 | T1 1.5R @ 162.98 |
| Target hit | 2024-08-01 15:20:00 | 163.31 | 164.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 148.10 | 148.56 | 0.00 | ORB-short ORB[148.20,149.90] vol=2.2x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:35:00 | 147.41 | 148.37 | 0.00 | T1 1.5R @ 147.41 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 148.10 | 148.18 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:10:00 | 146.88 | 147.57 | 0.00 | ORB-short ORB[147.65,148.99] vol=1.7x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-08-16 10:20:00 | 147.37 | 147.54 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:05:00 | 153.20 | 152.85 | 0.00 | ORB-long ORB[152.51,153.15] vol=2.0x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 11:15:00 | 153.57 | 152.98 | 0.00 | T1 1.5R @ 153.57 |
| Target hit | 2024-08-22 15:20:00 | 154.10 | 153.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 153.18 | 153.91 | 0.00 | ORB-short ORB[153.75,155.00] vol=2.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 153.47 | 153.77 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:05:00 | 148.97 | 149.84 | 0.00 | ORB-short ORB[149.91,150.95] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-09-10 12:00:00 | 149.27 | 149.66 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 154.37 | 153.57 | 0.00 | ORB-long ORB[152.50,153.88] vol=1.7x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-09-13 09:40:00 | 153.88 | 153.68 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 152.58 | 152.33 | 0.00 | ORB-long ORB[151.94,152.49] vol=2.1x ATR=0.26 |
| Stop hit — per-position SL triggered | 2024-09-18 09:40:00 | 152.32 | 152.36 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:15:00 | 153.20 | 152.24 | 0.00 | ORB-long ORB[150.65,152.67] vol=1.5x ATR=0.39 |
| Stop hit — per-position SL triggered | 2024-09-20 11:30:00 | 152.81 | 152.28 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 157.50 | 156.60 | 0.00 | ORB-long ORB[154.85,157.19] vol=2.6x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 09:45:00 | 158.16 | 156.97 | 0.00 | T1 1.5R @ 158.16 |
| Target hit | 2024-09-24 15:20:00 | 160.60 | 159.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-09-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:45:00 | 162.39 | 161.27 | 0.00 | ORB-long ORB[160.65,162.05] vol=2.0x ATR=0.44 |
| Stop hit — per-position SL triggered | 2024-09-26 10:35:00 | 161.95 | 162.04 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 11:15:00 | 169.93 | 169.12 | 0.00 | ORB-long ORB[167.26,169.55] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-09-30 11:30:00 | 169.38 | 169.15 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:45:00 | 166.35 | 167.22 | 0.00 | ORB-short ORB[166.67,167.95] vol=1.6x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:00:00 | 165.48 | 166.88 | 0.00 | T1 1.5R @ 165.48 |
| Target hit | 2024-10-07 11:25:00 | 165.47 | 165.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 162.72 | 161.32 | 0.00 | ORB-long ORB[159.20,161.39] vol=4.1x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:40:00 | 163.52 | 161.77 | 0.00 | T1 1.5R @ 163.52 |
| Stop hit — per-position SL triggered | 2024-10-11 09:50:00 | 162.72 | 161.94 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:30:00 | 152.88 | 151.81 | 0.00 | ORB-long ORB[151.10,152.50] vol=2.1x ATR=0.49 |
| Stop hit — per-position SL triggered | 2024-10-18 09:45:00 | 152.39 | 152.17 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 146.93 | 147.88 | 0.00 | ORB-short ORB[147.54,149.50] vol=2.2x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:45:00 | 146.29 | 147.35 | 0.00 | T1 1.5R @ 146.29 |
| Target hit | 2024-10-25 11:15:00 | 146.41 | 146.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2024-10-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:25:00 | 148.73 | 146.71 | 0.00 | ORB-long ORB[144.73,146.70] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-10-28 11:50:00 | 148.10 | 147.44 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 11:00:00 | 149.80 | 149.27 | 0.00 | ORB-long ORB[148.20,149.47] vol=1.5x ATR=0.29 |
| Stop hit — per-position SL triggered | 2024-10-31 11:10:00 | 149.51 | 149.29 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 11:05:00 | 146.99 | 146.14 | 0.00 | ORB-long ORB[145.60,146.71] vol=1.6x ATR=0.42 |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 146.57 | 146.16 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:40:00 | 140.71 | 142.13 | 0.00 | ORB-short ORB[141.71,143.60] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 141.25 | 141.85 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 10:45:00 | 140.92 | 141.78 | 0.00 | ORB-short ORB[141.40,142.31] vol=1.9x ATR=0.40 |
| Stop hit — per-position SL triggered | 2024-11-19 10:55:00 | 141.32 | 141.72 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 11:05:00 | 146.13 | 145.00 | 0.00 | ORB-long ORB[143.85,144.94] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2024-11-25 11:20:00 | 145.76 | 145.05 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 143.12 | 143.57 | 0.00 | ORB-short ORB[143.19,144.43] vol=1.8x ATR=0.34 |
| Stop hit — per-position SL triggered | 2024-11-27 09:50:00 | 143.46 | 143.46 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 145.14 | 144.69 | 0.00 | ORB-long ORB[144.00,145.09] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2024-11-28 09:45:00 | 144.84 | 144.78 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:55:00 | 145.32 | 146.11 | 0.00 | ORB-short ORB[146.10,147.05] vol=2.0x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:55:00 | 144.95 | 145.79 | 0.00 | T1 1.5R @ 144.95 |
| Stop hit — per-position SL triggered | 2024-12-04 13:30:00 | 145.32 | 145.45 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:45:00 | 145.45 | 145.77 | 0.00 | ORB-short ORB[145.52,146.74] vol=2.0x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-12-05 10:50:00 | 145.88 | 145.65 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 147.06 | 147.59 | 0.00 | ORB-short ORB[147.10,148.50] vol=1.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-12-09 11:35:00 | 147.49 | 147.27 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:45:00 | 151.08 | 150.35 | 0.00 | ORB-long ORB[149.31,150.80] vol=1.9x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:25:00 | 151.87 | 150.83 | 0.00 | T1 1.5R @ 151.87 |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 151.08 | 151.24 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:35:00 | 149.71 | 149.89 | 0.00 | ORB-short ORB[149.85,150.87] vol=1.6x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:05:00 | 149.17 | 149.80 | 0.00 | T1 1.5R @ 149.17 |
| Stop hit — per-position SL triggered | 2024-12-12 12:45:00 | 149.71 | 149.74 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:55:00 | 147.88 | 148.58 | 0.00 | ORB-short ORB[148.24,149.80] vol=3.2x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:15:00 | 147.20 | 148.17 | 0.00 | T1 1.5R @ 147.20 |
| Stop hit — per-position SL triggered | 2024-12-16 12:45:00 | 147.88 | 148.12 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 143.50 | 142.62 | 0.00 | ORB-long ORB[142.01,142.89] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 143.09 | 142.78 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:10:00 | 139.90 | 140.78 | 0.00 | ORB-short ORB[140.90,141.50] vol=2.1x ATR=0.32 |
| Stop hit — per-position SL triggered | 2024-12-24 11:25:00 | 140.22 | 140.72 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:45:00 | 139.14 | 138.72 | 0.00 | ORB-long ORB[137.84,138.99] vol=1.8x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-01-03 09:55:00 | 138.78 | 138.75 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-02-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:50:00 | 134.90 | 134.26 | 0.00 | ORB-long ORB[133.08,134.46] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-02-05 13:05:00 | 134.34 | 134.65 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:00:00 | 136.69 | 135.93 | 0.00 | ORB-long ORB[135.00,136.50] vol=2.0x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:15:00 | 137.17 | 136.11 | 0.00 | T1 1.5R @ 137.17 |
| Target hit | 2025-02-20 15:20:00 | 138.25 | 137.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:35:00 | 138.87 | 138.32 | 0.00 | ORB-long ORB[137.40,138.80] vol=1.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-02-25 09:40:00 | 138.36 | 138.34 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:30:00 | 141.17 | 140.54 | 0.00 | ORB-long ORB[139.00,141.05] vol=2.1x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 09:35:00 | 141.89 | 140.87 | 0.00 | T1 1.5R @ 141.89 |
| Target hit | 2025-03-05 15:20:00 | 146.37 | 144.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-03-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:45:00 | 153.78 | 152.90 | 0.00 | ORB-long ORB[151.56,153.46] vol=2.1x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-03-10 09:55:00 | 153.20 | 152.97 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:10:00 | 148.89 | 150.43 | 0.00 | ORB-short ORB[150.60,152.28] vol=2.1x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:40:00 | 148.25 | 150.15 | 0.00 | T1 1.5R @ 148.25 |
| Stop hit — per-position SL triggered | 2025-03-12 13:10:00 | 148.89 | 149.64 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:50:00 | 153.64 | 152.93 | 0.00 | ORB-long ORB[152.22,153.19] vol=1.5x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 153.33 | 152.94 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-03-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:05:00 | 154.71 | 155.00 | 0.00 | ORB-short ORB[155.16,156.91] vol=1.7x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 11:15:00 | 154.14 | 154.96 | 0.00 | T1 1.5R @ 154.14 |
| Stop hit — per-position SL triggered | 2025-03-28 11:55:00 | 154.71 | 154.90 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-04-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:35:00 | 154.94 | 153.48 | 0.00 | ORB-long ORB[152.43,154.29] vol=2.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 10:50:00 | 155.68 | 153.92 | 0.00 | T1 1.5R @ 155.68 |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 154.94 | 154.18 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 135.93 | 135.48 | 0.00 | ORB-long ORB[134.84,135.84] vol=1.6x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:00:00 | 136.66 | 135.70 | 0.00 | T1 1.5R @ 136.66 |
| Stop hit — per-position SL triggered | 2025-04-15 10:05:00 | 135.93 | 135.71 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-17 09:30:00 | 133.90 | 134.50 | 0.00 | ORB-short ORB[134.10,135.18] vol=2.1x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-04-17 09:35:00 | 134.45 | 134.47 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-04-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:40:00 | 140.76 | 142.09 | 0.00 | ORB-short ORB[141.61,143.45] vol=1.7x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 140.03 | 141.95 | 0.00 | T1 1.5R @ 140.03 |
| Target hit | 2025-04-25 12:55:00 | 140.20 | 140.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — BUY (started 2025-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 11:05:00 | 141.32 | 140.21 | 0.00 | ORB-long ORB[138.38,140.39] vol=2.5x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 11:30:00 | 141.89 | 140.42 | 0.00 | T1 1.5R @ 141.89 |
| Stop hit — per-position SL triggered | 2025-04-28 14:15:00 | 141.32 | 141.09 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-05-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:45:00 | 143.00 | 141.29 | 0.00 | ORB-long ORB[139.45,140.99] vol=2.2x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-05-02 09:50:00 | 142.37 | 141.53 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:30:00 | 143.13 | 142.38 | 0.00 | ORB-long ORB[141.54,142.59] vol=2.9x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-05-06 09:40:00 | 142.74 | 142.63 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 11:00:00 | 145.01 | 145.54 | 0.00 | ORB-short ORB[145.43,146.66] vol=1.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 12:00:00 | 144.53 | 145.39 | 0.00 | T1 1.5R @ 144.53 |
| Target hit | 2025-05-08 15:20:00 | 143.26 | 144.28 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:35:00 | 165.65 | 2024-05-14 09:45:00 | 165.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-16 11:15:00 | 164.10 | 2024-05-16 11:40:00 | 164.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-21 09:35:00 | 170.35 | 2024-05-21 09:45:00 | 169.90 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-11 10:25:00 | 182.66 | 2024-06-11 10:35:00 | 183.48 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-11 10:25:00 | 182.66 | 2024-06-11 11:00:00 | 182.66 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 11:05:00 | 181.86 | 2024-06-13 11:20:00 | 181.33 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-06-13 11:05:00 | 181.86 | 2024-06-13 11:50:00 | 181.86 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-18 09:40:00 | 182.92 | 2024-06-18 09:45:00 | 183.35 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-20 11:00:00 | 182.59 | 2024-06-20 11:10:00 | 182.12 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-21 09:50:00 | 180.22 | 2024-06-21 10:45:00 | 179.57 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-26 09:55:00 | 172.80 | 2024-06-26 10:50:00 | 172.09 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-26 09:55:00 | 172.80 | 2024-06-26 11:45:00 | 172.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 10:55:00 | 175.68 | 2024-07-02 11:05:00 | 175.28 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-10 10:05:00 | 169.92 | 2024-07-10 10:20:00 | 169.26 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 10:05:00 | 169.92 | 2024-07-10 15:20:00 | 167.88 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2024-07-16 10:30:00 | 169.50 | 2024-07-16 11:15:00 | 169.10 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-24 09:30:00 | 160.80 | 2024-07-24 09:40:00 | 160.26 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-26 10:40:00 | 161.86 | 2024-07-26 11:00:00 | 162.48 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-26 10:40:00 | 161.86 | 2024-07-26 12:05:00 | 161.86 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 11:15:00 | 164.06 | 2024-07-30 11:40:00 | 163.68 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-31 11:10:00 | 163.82 | 2024-07-31 11:45:00 | 164.22 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-01 10:35:00 | 163.97 | 2024-08-01 11:00:00 | 162.98 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-08-01 10:35:00 | 163.97 | 2024-08-01 15:20:00 | 163.31 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-14 09:30:00 | 148.10 | 2024-08-14 09:35:00 | 147.41 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-08-14 09:30:00 | 148.10 | 2024-08-14 09:45:00 | 148.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-16 10:10:00 | 146.88 | 2024-08-16 10:20:00 | 147.37 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-22 11:05:00 | 153.20 | 2024-08-22 11:15:00 | 153.57 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-08-22 11:05:00 | 153.20 | 2024-08-22 15:20:00 | 154.10 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2024-08-28 09:30:00 | 153.18 | 2024-08-28 09:40:00 | 153.47 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-09-10 11:05:00 | 148.97 | 2024-09-10 12:00:00 | 149.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-13 09:30:00 | 154.37 | 2024-09-13 09:40:00 | 153.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-18 09:30:00 | 152.58 | 2024-09-18 09:40:00 | 152.32 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-09-20 11:15:00 | 153.20 | 2024-09-20 11:30:00 | 152.81 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-24 09:30:00 | 157.50 | 2024-09-24 09:45:00 | 158.16 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-09-24 09:30:00 | 157.50 | 2024-09-24 15:20:00 | 160.60 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2024-09-26 09:45:00 | 162.39 | 2024-09-26 10:35:00 | 161.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-30 11:15:00 | 169.93 | 2024-09-30 11:30:00 | 169.38 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-07 09:45:00 | 166.35 | 2024-10-07 10:00:00 | 165.48 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-07 09:45:00 | 166.35 | 2024-10-07 11:25:00 | 165.47 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-11 09:35:00 | 162.72 | 2024-10-11 09:40:00 | 163.52 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-10-11 09:35:00 | 162.72 | 2024-10-11 09:50:00 | 162.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 09:30:00 | 152.88 | 2024-10-18 09:45:00 | 152.39 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-25 09:30:00 | 146.93 | 2024-10-25 09:45:00 | 146.29 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-25 09:30:00 | 146.93 | 2024-10-25 11:15:00 | 146.41 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-10-28 10:25:00 | 148.73 | 2024-10-28 11:50:00 | 148.10 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-31 11:00:00 | 149.80 | 2024-10-31 11:10:00 | 149.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-11-12 11:05:00 | 146.99 | 2024-11-12 11:15:00 | 146.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-13 09:40:00 | 140.71 | 2024-11-13 09:50:00 | 141.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-11-19 10:45:00 | 140.92 | 2024-11-19 10:55:00 | 141.32 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-25 11:05:00 | 146.13 | 2024-11-25 11:20:00 | 145.76 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-11-27 09:35:00 | 143.12 | 2024-11-27 09:50:00 | 143.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-28 09:30:00 | 145.14 | 2024-11-28 09:45:00 | 144.84 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-04 10:55:00 | 145.32 | 2024-12-04 11:55:00 | 144.95 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-12-04 10:55:00 | 145.32 | 2024-12-04 13:30:00 | 145.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 09:45:00 | 145.45 | 2024-12-05 10:50:00 | 145.88 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-09 09:30:00 | 147.06 | 2024-12-09 11:35:00 | 147.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-10 09:45:00 | 151.08 | 2024-12-10 10:25:00 | 151.87 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-12-10 09:45:00 | 151.08 | 2024-12-10 12:15:00 | 151.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:35:00 | 149.71 | 2024-12-12 12:05:00 | 149.17 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-12 10:35:00 | 149.71 | 2024-12-12 12:45:00 | 149.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 09:55:00 | 147.88 | 2024-12-16 12:15:00 | 147.20 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-16 09:55:00 | 147.88 | 2024-12-16 12:45:00 | 147.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 09:35:00 | 143.50 | 2024-12-20 09:45:00 | 143.09 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-24 11:10:00 | 139.90 | 2024-12-24 11:25:00 | 140.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-03 09:45:00 | 139.14 | 2025-01-03 09:55:00 | 138.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-05 09:50:00 | 134.90 | 2025-02-05 13:05:00 | 134.34 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-20 11:00:00 | 136.69 | 2025-02-20 11:15:00 | 137.17 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-02-20 11:00:00 | 136.69 | 2025-02-20 15:20:00 | 138.25 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2025-02-25 09:35:00 | 138.87 | 2025-02-25 09:40:00 | 138.36 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-05 09:30:00 | 141.17 | 2025-03-05 09:35:00 | 141.89 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-03-05 09:30:00 | 141.17 | 2025-03-05 15:20:00 | 146.37 | TARGET_HIT | 0.50 | 3.68% |
| BUY | retest1 | 2025-03-10 09:45:00 | 153.78 | 2025-03-10 09:55:00 | 153.20 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-03-12 11:10:00 | 148.89 | 2025-03-12 11:40:00 | 148.25 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-03-12 11:10:00 | 148.89 | 2025-03-12 13:10:00 | 148.89 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:50:00 | 153.64 | 2025-03-18 10:55:00 | 153.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-03-28 11:05:00 | 154.71 | 2025-03-28 11:15:00 | 154.14 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-03-28 11:05:00 | 154.71 | 2025-03-28 11:55:00 | 154.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 10:35:00 | 154.94 | 2025-04-02 10:50:00 | 155.68 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-04-02 10:35:00 | 154.94 | 2025-04-02 11:15:00 | 154.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 09:30:00 | 135.93 | 2025-04-15 10:00:00 | 136.66 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-15 09:30:00 | 135.93 | 2025-04-15 10:05:00 | 135.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-17 09:30:00 | 133.90 | 2025-04-17 09:35:00 | 134.45 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-25 09:40:00 | 140.76 | 2025-04-25 09:45:00 | 140.03 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-25 09:40:00 | 140.76 | 2025-04-25 12:55:00 | 140.20 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-28 11:05:00 | 141.32 | 2025-04-28 11:30:00 | 141.89 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-28 11:05:00 | 141.32 | 2025-04-28 14:15:00 | 141.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-02 09:45:00 | 143.00 | 2025-05-02 09:50:00 | 142.37 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-06 09:30:00 | 143.13 | 2025-05-06 09:40:00 | 142.74 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-08 11:00:00 | 145.01 | 2025-05-08 12:00:00 | 144.53 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-05-08 11:00:00 | 145.01 | 2025-05-08 15:20:00 | 143.26 | TARGET_HIT | 0.50 | 1.21% |
