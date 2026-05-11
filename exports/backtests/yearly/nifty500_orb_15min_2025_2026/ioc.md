# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2025-09-08 09:15:00 → 2026-05-08 15:25:00 (10738 bars)
- **Last close:** 144.88
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
| ENTRY1 | 39 |
| ENTRY2 | 0 |
| PARTIAL | 15 |
| TARGET_HIT | 6 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 33
- **Target hits / Stop hits / Partials:** 6 / 33 / 15
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 2.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 12 | 35.3% | 4 | 22 | 8 | -0.03% | -0.9% |
| BUY @ 2nd Alert (retest1) | 34 | 12 | 35.3% | 4 | 22 | 8 | -0.03% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 9 | 45.0% | 2 | 11 | 7 | 0.18% | 3.7% |
| SELL @ 2nd Alert (retest1) | 20 | 9 | 45.0% | 2 | 11 | 7 | 0.18% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 21 | 38.9% | 6 | 33 | 15 | 0.05% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:05:00 | 141.99 | 141.37 | 0.00 | ORB-long ORB[140.70,141.50] vol=2.4x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-09-09 10:10:00 | 141.71 | 141.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-09-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:25:00 | 143.70 | 144.26 | 0.00 | ORB-short ORB[144.01,145.20] vol=1.8x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-09-12 10:45:00 | 143.98 | 144.12 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-09-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:50:00 | 143.18 | 143.08 | 0.00 | ORB-long ORB[142.31,143.00] vol=1.7x ATR=0.21 |
| Stop hit — per-position SL triggered | 2025-09-15 10:55:00 | 142.97 | 143.08 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:15:00 | 145.04 | 144.55 | 0.00 | ORB-long ORB[143.82,144.55] vol=2.2x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-09-16 10:20:00 | 144.78 | 144.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:15:00 | 146.44 | 145.83 | 0.00 | ORB-long ORB[144.92,146.04] vol=2.1x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:25:00 | 146.81 | 146.16 | 0.00 | T1 1.5R @ 146.81 |
| Stop hit — per-position SL triggered | 2025-09-17 11:05:00 | 146.44 | 146.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:30:00 | 148.62 | 147.78 | 0.00 | ORB-long ORB[146.73,147.89] vol=2.9x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 09:40:00 | 149.07 | 148.51 | 0.00 | T1 1.5R @ 149.07 |
| Target hit | 2025-09-19 10:30:00 | 148.90 | 148.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:15:00 | 144.05 | 144.87 | 0.00 | ORB-short ORB[144.39,146.22] vol=3.1x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-09-26 12:45:00 | 144.36 | 144.57 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-09-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:35:00 | 147.99 | 146.69 | 0.00 | ORB-long ORB[145.04,146.69] vol=1.6x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:40:00 | 148.57 | 147.15 | 0.00 | T1 1.5R @ 148.57 |
| Target hit | 2025-09-29 11:35:00 | 148.40 | 148.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-10-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:45:00 | 151.51 | 150.72 | 0.00 | ORB-long ORB[149.89,150.85] vol=2.3x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-10-01 10:00:00 | 151.15 | 150.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-10-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:25:00 | 156.06 | 154.86 | 0.00 | ORB-long ORB[153.84,154.86] vol=3.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-10-08 10:30:00 | 155.62 | 154.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:10:00 | 153.94 | 153.31 | 0.00 | ORB-long ORB[153.00,153.90] vol=2.4x ATR=0.28 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 153.66 | 153.33 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:30:00 | 153.14 | 152.30 | 0.00 | ORB-long ORB[151.13,152.55] vol=2.5x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 09:45:00 | 153.71 | 152.75 | 0.00 | T1 1.5R @ 153.71 |
| Target hit | 2025-10-27 11:25:00 | 153.26 | 153.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2025-10-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 10:00:00 | 164.78 | 163.82 | 0.00 | ORB-long ORB[162.15,163.74] vol=1.8x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:10:00 | 165.61 | 164.24 | 0.00 | T1 1.5R @ 165.61 |
| Stop hit — per-position SL triggered | 2025-10-30 12:15:00 | 164.78 | 165.06 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-10-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 11:10:00 | 165.99 | 165.08 | 0.00 | ORB-long ORB[163.09,165.06] vol=2.0x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 12:00:00 | 166.58 | 165.37 | 0.00 | T1 1.5R @ 166.58 |
| Stop hit — per-position SL triggered | 2025-10-31 13:40:00 | 165.99 | 165.67 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-11-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:05:00 | 168.39 | 167.15 | 0.00 | ORB-long ORB[165.90,167.70] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 167.83 | 167.26 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 169.28 | 168.56 | 0.00 | ORB-long ORB[167.13,168.94] vol=2.2x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:50:00 | 169.90 | 168.99 | 0.00 | T1 1.5R @ 169.90 |
| Stop hit — per-position SL triggered | 2025-11-04 09:55:00 | 169.28 | 169.04 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 09:40:00 | 167.97 | 167.12 | 0.00 | ORB-long ORB[166.10,167.93] vol=2.0x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-11-07 09:50:00 | 167.51 | 167.18 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-11-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:55:00 | 169.44 | 168.57 | 0.00 | ORB-long ORB[167.32,169.00] vol=2.1x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-11-11 10:30:00 | 169.00 | 168.80 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 174.10 | 173.17 | 0.00 | ORB-long ORB[172.11,173.29] vol=1.5x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-11-12 09:50:00 | 173.66 | 173.55 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 173.35 | 172.78 | 0.00 | ORB-long ORB[171.52,173.21] vol=1.8x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-11-14 09:35:00 | 172.96 | 172.82 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 173.03 | 172.20 | 0.00 | ORB-long ORB[171.16,172.60] vol=1.6x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:50:00 | 173.59 | 172.60 | 0.00 | T1 1.5R @ 173.59 |
| Target hit | 2025-11-17 11:15:00 | 173.24 | 173.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2025-11-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 11:10:00 | 170.00 | 171.33 | 0.00 | ORB-short ORB[170.76,172.23] vol=1.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-11-19 11:25:00 | 170.37 | 171.23 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:45:00 | 166.53 | 167.25 | 0.00 | ORB-short ORB[167.42,168.61] vol=5.6x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-11-21 11:00:00 | 166.89 | 167.19 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:45:00 | 162.00 | 162.49 | 0.00 | ORB-short ORB[162.30,163.70] vol=2.3x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-11-28 10:05:00 | 162.33 | 162.39 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 162.90 | 161.75 | 0.00 | ORB-long ORB[161.00,162.43] vol=2.3x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-12-03 11:10:00 | 162.55 | 161.83 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 162.20 | 163.36 | 0.00 | ORB-short ORB[163.08,164.89] vol=1.6x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:20:00 | 161.53 | 163.15 | 0.00 | T1 1.5R @ 161.53 |
| Stop hit — per-position SL triggered | 2026-01-06 10:30:00 | 162.20 | 163.03 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 159.10 | 160.56 | 0.00 | ORB-short ORB[161.11,162.70] vol=1.6x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 158.53 | 160.43 | 0.00 | T1 1.5R @ 158.53 |
| Target hit | 2026-01-08 15:20:00 | 156.10 | 158.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 11:15:00 | 164.35 | 163.59 | 0.00 | ORB-long ORB[161.77,163.45] vol=1.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-01-29 11:25:00 | 163.92 | 163.62 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 161.27 | 161.57 | 0.00 | ORB-short ORB[161.51,162.94] vol=1.8x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 11:25:00 | 160.79 | 161.48 | 0.00 | T1 1.5R @ 160.79 |
| Target hit | 2026-02-01 13:55:00 | 160.55 | 160.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 30 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 177.52 | 176.46 | 0.00 | ORB-long ORB[175.20,177.16] vol=2.8x ATR=0.49 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 177.03 | 176.68 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 179.64 | 179.32 | 0.00 | ORB-long ORB[176.82,179.30] vol=1.8x ATR=0.43 |
| Stop hit — per-position SL triggered | 2026-02-11 12:50:00 | 179.21 | 179.49 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 177.87 | 179.80 | 0.00 | ORB-short ORB[179.90,182.25] vol=2.1x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 177.04 | 179.42 | 0.00 | T1 1.5R @ 177.04 |
| Stop hit — per-position SL triggered | 2026-02-12 13:25:00 | 177.87 | 178.39 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 172.64 | 173.28 | 0.00 | ORB-short ORB[173.75,174.95] vol=2.0x ATR=0.35 |
| Stop hit — per-position SL triggered | 2026-02-17 11:35:00 | 172.99 | 173.23 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 177.46 | 176.77 | 0.00 | ORB-long ORB[175.86,177.38] vol=2.1x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 177.00 | 177.00 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:00:00 | 176.38 | 177.18 | 0.00 | ORB-short ORB[177.00,177.99] vol=1.8x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:10:00 | 175.75 | 176.99 | 0.00 | T1 1.5R @ 175.75 |
| Stop hit — per-position SL triggered | 2026-02-19 10:40:00 | 176.38 | 176.62 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 11:00:00 | 176.90 | 175.73 | 0.00 | ORB-long ORB[173.67,175.97] vol=3.0x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-02-23 11:10:00 | 176.38 | 175.81 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 179.20 | 178.26 | 0.00 | ORB-long ORB[176.51,178.67] vol=2.1x ATR=0.51 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 178.69 | 178.55 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 145.00 | 145.47 | 0.00 | ORB-short ORB[145.04,146.43] vol=1.6x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:55:00 | 144.56 | 145.31 | 0.00 | T1 1.5R @ 144.56 |
| Stop hit — per-position SL triggered | 2026-04-16 10:10:00 | 145.00 | 145.25 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 143.73 | 144.64 | 0.00 | ORB-short ORB[144.50,145.60] vol=1.8x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:45:00 | 143.18 | 144.44 | 0.00 | T1 1.5R @ 143.18 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 143.73 | 143.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-09 10:05:00 | 141.99 | 2025-09-09 10:10:00 | 141.71 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-12 10:25:00 | 143.70 | 2025-09-12 10:45:00 | 143.98 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-15 10:50:00 | 143.18 | 2025-09-15 10:55:00 | 142.97 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-16 10:15:00 | 145.04 | 2025-09-16 10:20:00 | 144.78 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-17 10:15:00 | 146.44 | 2025-09-17 10:25:00 | 146.81 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-09-17 10:15:00 | 146.44 | 2025-09-17 11:05:00 | 146.44 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 09:30:00 | 148.62 | 2025-09-19 09:40:00 | 149.07 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-19 09:30:00 | 148.62 | 2025-09-19 10:30:00 | 148.90 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2025-09-26 11:15:00 | 144.05 | 2025-09-26 12:45:00 | 144.36 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-29 09:35:00 | 147.99 | 2025-09-29 09:40:00 | 148.57 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-29 09:35:00 | 147.99 | 2025-09-29 11:35:00 | 148.40 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-01 09:45:00 | 151.51 | 2025-10-01 10:00:00 | 151.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-08 10:25:00 | 156.06 | 2025-10-08 10:30:00 | 155.62 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-15 11:10:00 | 153.94 | 2025-10-15 11:15:00 | 153.66 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-27 09:30:00 | 153.14 | 2025-10-27 09:45:00 | 153.71 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-27 09:30:00 | 153.14 | 2025-10-27 11:25:00 | 153.26 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2025-10-30 10:00:00 | 164.78 | 2025-10-30 10:10:00 | 165.61 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-30 10:00:00 | 164.78 | 2025-10-30 12:15:00 | 164.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 11:10:00 | 165.99 | 2025-10-31 12:00:00 | 166.58 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-31 11:10:00 | 165.99 | 2025-10-31 13:40:00 | 165.99 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:05:00 | 168.39 | 2025-11-03 10:15:00 | 167.83 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-04 09:30:00 | 169.28 | 2025-11-04 09:50:00 | 169.90 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-04 09:30:00 | 169.28 | 2025-11-04 09:55:00 | 169.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 09:40:00 | 167.97 | 2025-11-07 09:50:00 | 167.51 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-11 09:55:00 | 169.44 | 2025-11-11 10:30:00 | 169.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-12 09:30:00 | 174.10 | 2025-11-12 09:50:00 | 173.66 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-14 09:30:00 | 173.35 | 2025-11-14 09:35:00 | 172.96 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-17 09:30:00 | 173.03 | 2025-11-17 09:50:00 | 173.59 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-11-17 09:30:00 | 173.03 | 2025-11-17 11:15:00 | 173.24 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2025-11-19 11:10:00 | 170.00 | 2025-11-19 11:25:00 | 170.37 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-21 10:45:00 | 166.53 | 2025-11-21 11:00:00 | 166.89 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-28 09:45:00 | 162.00 | 2025-11-28 10:05:00 | 162.33 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-03 11:05:00 | 162.90 | 2025-12-03 11:10:00 | 162.55 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-06 10:15:00 | 162.20 | 2026-01-06 10:20:00 | 161.53 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-06 10:15:00 | 162.20 | 2026-01-06 10:30:00 | 162.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 11:10:00 | 159.10 | 2026-01-08 11:20:00 | 158.53 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-08 11:10:00 | 159.10 | 2026-01-08 15:20:00 | 156.10 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2026-01-29 11:15:00 | 164.35 | 2026-01-29 11:25:00 | 163.92 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-01 11:05:00 | 161.27 | 2026-02-01 11:25:00 | 160.79 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-01 11:05:00 | 161.27 | 2026-02-01 13:55:00 | 160.55 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-10 09:30:00 | 177.52 | 2026-02-10 09:45:00 | 177.03 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-11 10:30:00 | 179.64 | 2026-02-11 12:50:00 | 179.21 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-12 11:15:00 | 177.87 | 2026-02-12 11:30:00 | 177.04 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-12 11:15:00 | 177.87 | 2026-02-12 13:25:00 | 177.87 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 11:05:00 | 172.64 | 2026-02-17 11:35:00 | 172.99 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-18 09:30:00 | 177.46 | 2026-02-18 09:45:00 | 177.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 10:00:00 | 176.38 | 2026-02-19 10:10:00 | 175.75 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-19 10:00:00 | 176.38 | 2026-02-19 10:40:00 | 176.38 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-23 11:00:00 | 176.90 | 2026-02-23 11:10:00 | 176.38 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-24 09:35:00 | 179.20 | 2026-02-24 09:55:00 | 178.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-16 09:35:00 | 145.00 | 2026-04-16 09:55:00 | 144.56 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-16 09:35:00 | 145.00 | 2026-04-16 10:10:00 | 145.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:35:00 | 143.73 | 2026-04-24 09:45:00 | 143.18 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 09:35:00 | 143.73 | 2026-04-24 11:20:00 | 143.73 | STOP_HIT | 0.50 | 0.00% |
