# Vedanta Ltd. (VEDL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18454 bars)
- **Last close:** 297.00
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
| ENTRY1 | 80 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 7 |
| STOP_HIT | 73 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 73
- **Target hits / Stop hits / Partials:** 7 / 73 / 24
- **Avg / median % per leg:** -0.01% / -0.20%
- **Sum % (uncompounded):** -0.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 14 | 23.3% | 2 | 46 | 12 | -0.07% | -4.3% |
| BUY @ 2nd Alert (retest1) | 60 | 14 | 23.3% | 2 | 46 | 12 | -0.07% | -4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 44 | 17 | 38.6% | 5 | 27 | 12 | 0.08% | 3.7% |
| SELL @ 2nd Alert (retest1) | 44 | 17 | 38.6% | 5 | 27 | 12 | 0.08% | 3.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 31 | 29.8% | 7 | 73 | 24 | -0.01% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:35:00 | 164.49 | 163.30 | 0.00 | ORB-long ORB[162.21,163.78] vol=2.1x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-05-13 09:40:00 | 163.77 | 163.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:35:00 | 167.60 | 166.40 | 0.00 | ORB-long ORB[164.93,166.55] vol=2.2x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-05-15 10:45:00 | 167.07 | 166.53 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 11:05:00 | 165.06 | 165.78 | 0.00 | ORB-short ORB[165.30,166.55] vol=1.7x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-05-16 11:30:00 | 165.37 | 165.73 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 11:15:00 | 164.55 | 163.42 | 0.00 | ORB-long ORB[162.42,164.42] vol=2.3x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-05-22 11:20:00 | 164.18 | 163.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 167.19 | 166.40 | 0.00 | ORB-long ORB[165.64,166.67] vol=3.2x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-05-26 09:40:00 | 166.72 | 166.49 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:20:00 | 167.28 | 166.89 | 0.00 | ORB-long ORB[166.33,167.27] vol=1.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-05-27 10:25:00 | 166.94 | 166.91 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:35:00 | 166.80 | 167.45 | 0.00 | ORB-short ORB[167.06,168.54] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-05-29 11:25:00 | 167.16 | 167.37 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:20:00 | 163.31 | 162.28 | 0.00 | ORB-long ORB[161.72,162.87] vol=1.7x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-06-04 10:25:00 | 162.78 | 162.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:35:00 | 170.94 | 169.91 | 0.00 | ORB-long ORB[168.95,170.39] vol=2.6x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 11:50:00 | 171.73 | 170.51 | 0.00 | T1 1.5R @ 171.73 |
| Target hit | 2025-06-09 15:20:00 | 171.52 | 171.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-06-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 11:10:00 | 174.25 | 173.15 | 0.00 | ORB-long ORB[171.84,173.78] vol=3.2x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-06-17 11:50:00 | 173.81 | 173.31 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 172.51 | 173.53 | 0.00 | ORB-short ORB[172.87,175.19] vol=1.9x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:55:00 | 171.59 | 173.13 | 0.00 | T1 1.5R @ 171.59 |
| Target hit | 2025-06-18 13:15:00 | 172.10 | 171.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2025-06-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:30:00 | 168.61 | 167.88 | 0.00 | ORB-long ORB[166.59,168.37] vol=1.7x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-06-24 11:25:00 | 168.19 | 168.05 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 171.35 | 171.86 | 0.00 | ORB-short ORB[172.21,173.91] vol=6.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-07-01 12:05:00 | 171.78 | 171.73 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:00:00 | 169.64 | 170.25 | 0.00 | ORB-short ORB[169.93,171.10] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-07-08 11:10:00 | 170.00 | 170.15 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 166.40 | 165.00 | 0.00 | ORB-long ORB[163.50,165.09] vol=2.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-07-11 09:45:00 | 165.83 | 165.10 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:30:00 | 168.48 | 166.94 | 0.00 | ORB-long ORB[164.53,166.24] vol=1.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-07-14 10:35:00 | 168.00 | 166.99 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:35:00 | 167.32 | 167.91 | 0.00 | ORB-short ORB[167.49,169.01] vol=2.3x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:45:00 | 166.71 | 167.78 | 0.00 | T1 1.5R @ 166.71 |
| Stop hit — per-position SL triggered | 2025-07-15 09:50:00 | 167.32 | 167.73 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:10:00 | 166.89 | 167.29 | 0.00 | ORB-short ORB[167.15,168.71] vol=1.5x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-07-16 11:30:00 | 167.22 | 167.25 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:40:00 | 167.34 | 167.81 | 0.00 | ORB-short ORB[167.57,168.46] vol=1.6x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-07-17 10:25:00 | 167.65 | 167.59 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:30:00 | 169.55 | 168.87 | 0.00 | ORB-long ORB[168.03,169.48] vol=1.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-07-21 10:40:00 | 169.10 | 168.90 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:35:00 | 170.07 | 170.53 | 0.00 | ORB-short ORB[170.32,171.14] vol=3.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:00:00 | 169.42 | 170.15 | 0.00 | T1 1.5R @ 169.42 |
| Target hit | 2025-07-24 15:20:00 | 169.12 | 169.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-07-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:45:00 | 164.38 | 165.19 | 0.00 | ORB-short ORB[164.64,166.18] vol=1.6x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-07-28 10:00:00 | 164.86 | 165.02 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:45:00 | 163.05 | 163.82 | 0.00 | ORB-short ORB[163.56,165.06] vol=1.6x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-07-30 09:55:00 | 163.48 | 163.76 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:00:00 | 161.16 | 159.70 | 0.00 | ORB-long ORB[158.18,160.37] vol=2.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-08-01 11:25:00 | 160.64 | 159.86 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 09:30:00 | 162.87 | 162.35 | 0.00 | ORB-long ORB[161.31,162.66] vol=2.3x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-08-05 09:35:00 | 162.48 | 162.43 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 161.24 | 161.84 | 0.00 | ORB-short ORB[161.25,162.83] vol=4.1x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:35:00 | 160.68 | 161.66 | 0.00 | T1 1.5R @ 160.68 |
| Stop hit — per-position SL triggered | 2025-08-11 14:45:00 | 161.24 | 161.10 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 165.36 | 164.36 | 0.00 | ORB-long ORB[162.96,165.02] vol=1.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-13 09:35:00 | 164.90 | 164.44 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 168.31 | 167.39 | 0.00 | ORB-long ORB[166.31,167.98] vol=2.2x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-08-25 09:35:00 | 167.97 | 167.46 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:30:00 | 157.92 | 158.61 | 0.00 | ORB-short ORB[158.05,159.68] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-08-29 10:30:00 | 158.29 | 158.21 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:10:00 | 163.30 | 163.07 | 0.00 | ORB-long ORB[162.30,163.26] vol=2.9x ATR=0.22 |
| Stop hit — per-position SL triggered | 2025-09-10 11:20:00 | 163.08 | 163.07 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:15:00 | 164.04 | 163.23 | 0.00 | ORB-long ORB[162.15,163.52] vol=2.8x ATR=0.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 10:20:00 | 164.44 | 163.37 | 0.00 | T1 1.5R @ 164.44 |
| Stop hit — per-position SL triggered | 2025-09-11 10:25:00 | 164.04 | 163.42 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:35:00 | 172.98 | 171.46 | 0.00 | ORB-long ORB[170.43,172.62] vol=3.1x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-09-15 10:45:00 | 172.38 | 171.68 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 173.01 | 173.83 | 0.00 | ORB-short ORB[173.33,175.21] vol=1.6x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-09-17 10:10:00 | 173.47 | 173.67 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:35:00 | 171.82 | 170.70 | 0.00 | ORB-long ORB[170.41,171.46] vol=2.0x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-09-18 10:55:00 | 171.18 | 170.86 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:00:00 | 169.85 | 170.85 | 0.00 | ORB-short ORB[170.15,171.54] vol=3.0x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:15:00 | 169.34 | 170.68 | 0.00 | T1 1.5R @ 169.34 |
| Stop hit — per-position SL triggered | 2025-09-19 11:20:00 | 169.85 | 170.65 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:55:00 | 172.66 | 172.12 | 0.00 | ORB-long ORB[170.28,172.25] vol=1.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-09-22 11:20:00 | 172.25 | 172.16 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 170.28 | 170.98 | 0.00 | ORB-short ORB[170.51,172.25] vol=1.5x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 09:45:00 | 169.67 | 170.59 | 0.00 | T1 1.5R @ 169.67 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 170.28 | 170.47 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:30:00 | 171.72 | 170.53 | 0.00 | ORB-long ORB[169.70,171.35] vol=2.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-09-24 10:40:00 | 171.31 | 170.61 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:50:00 | 171.33 | 170.24 | 0.00 | ORB-long ORB[168.35,170.58] vol=1.9x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 09:55:00 | 172.08 | 170.87 | 0.00 | T1 1.5R @ 172.08 |
| Stop hit — per-position SL triggered | 2025-09-25 12:00:00 | 171.33 | 171.44 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:00:00 | 168.58 | 171.00 | 0.00 | ORB-short ORB[171.22,173.71] vol=2.7x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:20:00 | 167.69 | 170.73 | 0.00 | T1 1.5R @ 167.69 |
| Target hit | 2025-09-26 15:20:00 | 167.79 | 169.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-09-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:45:00 | 170.73 | 169.93 | 0.00 | ORB-long ORB[168.26,169.81] vol=2.1x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-09-29 10:50:00 | 170.28 | 169.95 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:00:00 | 176.99 | 175.55 | 0.00 | ORB-long ORB[173.97,176.03] vol=2.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 176.38 | 175.77 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 175.32 | 176.34 | 0.00 | ORB-short ORB[175.43,177.90] vol=2.3x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 176.03 | 176.09 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 178.45 | 178.94 | 0.00 | ORB-short ORB[178.61,179.91] vol=2.5x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-10-13 10:05:00 | 179.01 | 178.73 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:00:00 | 180.86 | 179.78 | 0.00 | ORB-long ORB[178.35,180.09] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-10-15 10:20:00 | 180.22 | 179.89 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:00:00 | 177.96 | 179.31 | 0.00 | ORB-short ORB[179.40,181.65] vol=2.1x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-10-16 10:05:00 | 178.43 | 179.23 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:20:00 | 178.37 | 179.35 | 0.00 | ORB-short ORB[178.84,180.19] vol=2.4x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:45:00 | 177.77 | 178.68 | 0.00 | T1 1.5R @ 177.77 |
| Target hit | 2025-10-17 15:20:00 | 177.55 | 177.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:35:00 | 179.96 | 179.20 | 0.00 | ORB-long ORB[177.77,179.89] vol=1.9x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:15:00 | 180.68 | 179.70 | 0.00 | T1 1.5R @ 180.68 |
| Stop hit — per-position SL triggered | 2025-10-23 10:45:00 | 179.96 | 179.85 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-10-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:55:00 | 188.84 | 187.30 | 0.00 | ORB-long ORB[186.24,188.13] vol=1.9x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-10-27 10:00:00 | 188.24 | 187.50 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:45:00 | 190.32 | 189.25 | 0.00 | ORB-long ORB[188.07,189.59] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-10-28 10:00:00 | 189.78 | 189.56 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:35:00 | 192.23 | 191.03 | 0.00 | ORB-long ORB[189.78,191.76] vol=1.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-11-06 09:45:00 | 191.69 | 191.16 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 186.70 | 187.42 | 0.00 | ORB-short ORB[186.93,188.39] vol=2.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-11-07 09:40:00 | 187.29 | 187.32 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:50:00 | 195.69 | 195.02 | 0.00 | ORB-long ORB[194.31,195.58] vol=2.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-11-11 10:10:00 | 195.18 | 195.21 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 197.75 | 196.76 | 0.00 | ORB-long ORB[195.36,197.10] vol=4.0x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-11-12 09:35:00 | 197.23 | 196.85 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:30:00 | 199.44 | 198.23 | 0.00 | ORB-long ORB[196.33,198.69] vol=4.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-11-13 09:40:00 | 198.74 | 198.38 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:15:00 | 192.13 | 193.17 | 0.00 | ORB-short ORB[192.81,194.38] vol=1.6x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-11-20 10:20:00 | 192.53 | 193.15 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:20:00 | 187.77 | 186.93 | 0.00 | ORB-long ORB[185.73,187.28] vol=1.8x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-11-24 10:30:00 | 187.33 | 186.98 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:35:00 | 192.12 | 191.39 | 0.00 | ORB-long ORB[189.55,191.72] vol=2.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-11-26 10:40:00 | 191.70 | 191.40 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:50:00 | 195.92 | 195.23 | 0.00 | ORB-long ORB[194.25,195.88] vol=1.7x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-11-28 10:00:00 | 195.45 | 195.33 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 198.22 | 200.60 | 0.00 | ORB-short ORB[200.77,202.64] vol=4.3x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-12-03 11:30:00 | 198.77 | 200.48 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:35:00 | 202.58 | 201.74 | 0.00 | ORB-long ORB[200.13,202.25] vol=1.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-12-04 11:40:00 | 201.89 | 202.28 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:55:00 | 194.36 | 195.93 | 0.00 | ORB-short ORB[195.69,198.13] vol=1.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 195.04 | 195.77 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 194.48 | 195.87 | 0.00 | ORB-short ORB[195.84,197.49] vol=1.7x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-12-08 11:20:00 | 194.94 | 195.68 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 11:05:00 | 196.74 | 195.36 | 0.00 | ORB-long ORB[193.31,194.93] vol=1.9x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 13:00:00 | 197.72 | 196.14 | 0.00 | T1 1.5R @ 197.72 |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 196.74 | 196.45 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:15:00 | 199.10 | 198.11 | 0.00 | ORB-long ORB[196.67,198.61] vol=1.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:50:00 | 200.15 | 198.46 | 0.00 | T1 1.5R @ 200.15 |
| Stop hit — per-position SL triggered | 2025-12-11 11:00:00 | 199.10 | 198.52 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:35:00 | 220.81 | 220.33 | 0.00 | ORB-long ORB[218.95,220.69] vol=2.4x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-12-23 09:40:00 | 220.25 | 220.34 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 222.43 | 221.44 | 0.00 | ORB-long ORB[219.93,222.21] vol=2.3x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 09:35:00 | 223.13 | 221.89 | 0.00 | T1 1.5R @ 223.13 |
| Stop hit — per-position SL triggered | 2025-12-24 09:45:00 | 222.43 | 222.09 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 224.49 | 225.91 | 0.00 | ORB-short ORB[224.76,227.68] vol=2.0x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:20:00 | 223.70 | 225.76 | 0.00 | T1 1.5R @ 223.70 |
| Stop hit — per-position SL triggered | 2025-12-26 11:30:00 | 224.49 | 225.72 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:35:00 | 228.86 | 227.82 | 0.00 | ORB-long ORB[226.07,228.80] vol=2.0x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-01-02 09:40:00 | 228.24 | 227.90 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:00:00 | 229.93 | 231.27 | 0.00 | ORB-short ORB[231.12,233.90] vol=2.2x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:20:00 | 229.05 | 231.11 | 0.00 | T1 1.5R @ 229.05 |
| Stop hit — per-position SL triggered | 2026-01-05 11:40:00 | 229.93 | 230.96 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:15:00 | 234.23 | 233.38 | 0.00 | ORB-long ORB[231.46,234.08] vol=1.6x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 10:35:00 | 235.12 | 233.64 | 0.00 | T1 1.5R @ 235.12 |
| Stop hit — per-position SL triggered | 2026-01-06 10:50:00 | 234.23 | 233.73 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 09:40:00 | 238.03 | 237.26 | 0.00 | ORB-long ORB[235.62,237.96] vol=1.9x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 09:50:00 | 239.12 | 237.67 | 0.00 | T1 1.5R @ 239.12 |
| Target hit | 2026-01-13 11:50:00 | 238.54 | 238.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 73 — SELL (started 2026-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:50:00 | 247.25 | 249.90 | 0.00 | ORB-short ORB[250.00,253.16] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-01-21 10:55:00 | 248.27 | 249.72 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:30:00 | 256.93 | 255.43 | 0.00 | ORB-long ORB[253.28,256.40] vol=3.0x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 09:35:00 | 258.23 | 256.35 | 0.00 | T1 1.5R @ 258.23 |
| Stop hit — per-position SL triggered | 2026-01-22 09:50:00 | 256.93 | 256.73 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 09:50:00 | 260.86 | 258.48 | 0.00 | ORB-long ORB[256.44,259.18] vol=2.4x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-01-23 10:10:00 | 259.76 | 259.21 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:15:00 | 257.10 | 255.52 | 0.00 | ORB-long ORB[253.93,256.93] vol=1.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:10:00 | 258.35 | 256.29 | 0.00 | T1 1.5R @ 258.35 |
| Stop hit — per-position SL triggered | 2026-02-10 13:30:00 | 257.10 | 256.69 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 253.20 | 255.25 | 0.00 | ORB-short ORB[255.21,257.98] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-02-19 12:20:00 | 253.81 | 254.81 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:05:00 | 270.62 | 271.84 | 0.00 | ORB-short ORB[271.54,275.28] vol=1.7x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 11:35:00 | 269.03 | 271.56 | 0.00 | T1 1.5R @ 269.03 |
| Stop hit — per-position SL triggered | 2026-04-08 12:25:00 | 270.62 | 271.40 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 293.43 | 294.84 | 0.00 | ORB-short ORB[293.63,297.75] vol=2.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:40:00 | 291.64 | 294.51 | 0.00 | T1 1.5R @ 291.64 |
| Target hit | 2026-04-21 15:20:00 | 286.85 | 290.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 275.84 | 272.26 | 0.00 | ORB-long ORB[269.87,272.58] vol=2.1x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 277.65 | 273.68 | 0.00 | T1 1.5R @ 277.65 |
| Stop hit — per-position SL triggered | 2026-04-27 10:55:00 | 275.84 | 274.90 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:35:00 | 164.49 | 2025-05-13 09:40:00 | 163.77 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-15 10:35:00 | 167.60 | 2025-05-15 10:45:00 | 167.07 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-16 11:05:00 | 165.06 | 2025-05-16 11:30:00 | 165.37 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-22 11:15:00 | 164.55 | 2025-05-22 11:20:00 | 164.18 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-26 09:30:00 | 167.19 | 2025-05-26 09:40:00 | 166.72 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-27 10:20:00 | 167.28 | 2025-05-27 10:25:00 | 166.94 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-29 10:35:00 | 166.80 | 2025-05-29 11:25:00 | 167.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-04 10:20:00 | 163.31 | 2025-06-04 10:25:00 | 162.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-09 10:35:00 | 170.94 | 2025-06-09 11:50:00 | 171.73 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-09 10:35:00 | 170.94 | 2025-06-09 15:20:00 | 171.52 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-06-17 11:10:00 | 174.25 | 2025-06-17 11:50:00 | 173.81 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-18 09:30:00 | 172.51 | 2025-06-18 09:55:00 | 171.59 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-06-18 09:30:00 | 172.51 | 2025-06-18 13:15:00 | 172.10 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-06-24 10:30:00 | 168.61 | 2025-06-24 11:25:00 | 168.19 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-01 10:50:00 | 171.35 | 2025-07-01 12:05:00 | 171.78 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-08 11:00:00 | 169.64 | 2025-07-08 11:10:00 | 170.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-11 09:40:00 | 166.40 | 2025-07-11 09:45:00 | 165.83 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-14 10:30:00 | 168.48 | 2025-07-14 10:35:00 | 168.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-15 09:35:00 | 167.32 | 2025-07-15 09:45:00 | 166.71 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-15 09:35:00 | 167.32 | 2025-07-15 09:50:00 | 167.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-16 11:10:00 | 166.89 | 2025-07-16 11:30:00 | 167.22 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-17 09:40:00 | 167.34 | 2025-07-17 10:25:00 | 167.65 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-21 10:30:00 | 169.55 | 2025-07-21 10:40:00 | 169.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-24 09:35:00 | 170.07 | 2025-07-24 10:00:00 | 169.42 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-24 09:35:00 | 170.07 | 2025-07-24 15:20:00 | 169.12 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-07-28 09:45:00 | 164.38 | 2025-07-28 10:00:00 | 164.86 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-30 09:45:00 | 163.05 | 2025-07-30 09:55:00 | 163.48 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-01 11:00:00 | 161.16 | 2025-08-01 11:25:00 | 160.64 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-05 09:30:00 | 162.87 | 2025-08-05 09:35:00 | 162.48 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-11 11:10:00 | 161.24 | 2025-08-11 11:35:00 | 160.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-11 11:10:00 | 161.24 | 2025-08-11 14:45:00 | 161.24 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 09:30:00 | 165.36 | 2025-08-13 09:35:00 | 164.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-25 09:30:00 | 168.31 | 2025-08-25 09:35:00 | 167.97 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-29 09:30:00 | 157.92 | 2025-08-29 10:30:00 | 158.29 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-10 11:10:00 | 163.30 | 2025-09-10 11:20:00 | 163.08 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-11 10:15:00 | 164.04 | 2025-09-11 10:20:00 | 164.44 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-11 10:15:00 | 164.04 | 2025-09-11 10:25:00 | 164.04 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-15 10:35:00 | 172.98 | 2025-09-15 10:45:00 | 172.38 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-17 09:45:00 | 173.01 | 2025-09-17 10:10:00 | 173.47 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-18 10:35:00 | 171.82 | 2025-09-18 10:55:00 | 171.18 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-19 11:00:00 | 169.85 | 2025-09-19 11:15:00 | 169.34 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-09-19 11:00:00 | 169.85 | 2025-09-19 11:20:00 | 169.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 10:55:00 | 172.66 | 2025-09-22 11:20:00 | 172.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-23 09:30:00 | 170.28 | 2025-09-23 09:45:00 | 169.67 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-09-23 09:30:00 | 170.28 | 2025-09-23 10:05:00 | 170.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-24 10:30:00 | 171.72 | 2025-09-24 10:40:00 | 171.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-25 09:50:00 | 171.33 | 2025-09-25 09:55:00 | 172.08 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-25 09:50:00 | 171.33 | 2025-09-25 12:00:00 | 171.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-26 11:00:00 | 168.58 | 2025-09-26 11:20:00 | 167.69 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-09-26 11:00:00 | 168.58 | 2025-09-26 15:20:00 | 167.79 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-09-29 10:45:00 | 170.73 | 2025-09-29 10:50:00 | 170.28 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-03 10:00:00 | 176.99 | 2025-10-03 10:15:00 | 176.38 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-08 11:00:00 | 175.32 | 2025-10-08 11:25:00 | 176.03 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-10-13 09:30:00 | 178.45 | 2025-10-13 10:05:00 | 179.01 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-15 10:00:00 | 180.86 | 2025-10-15 10:20:00 | 180.22 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-16 10:00:00 | 177.96 | 2025-10-16 10:05:00 | 178.43 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-17 10:20:00 | 178.37 | 2025-10-17 11:45:00 | 177.77 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-17 10:20:00 | 178.37 | 2025-10-17 15:20:00 | 177.55 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-23 09:35:00 | 179.96 | 2025-10-23 10:15:00 | 180.68 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-23 09:35:00 | 179.96 | 2025-10-23 10:45:00 | 179.96 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:55:00 | 188.84 | 2025-10-27 10:00:00 | 188.24 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-28 09:45:00 | 190.32 | 2025-10-28 10:00:00 | 189.78 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-06 09:35:00 | 192.23 | 2025-11-06 09:45:00 | 191.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-07 09:30:00 | 186.70 | 2025-11-07 09:40:00 | 187.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-11 09:50:00 | 195.69 | 2025-11-11 10:10:00 | 195.18 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-12 09:30:00 | 197.75 | 2025-11-12 09:35:00 | 197.23 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-13 09:30:00 | 199.44 | 2025-11-13 09:40:00 | 198.74 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-20 10:15:00 | 192.13 | 2025-11-20 10:20:00 | 192.53 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-24 10:20:00 | 187.77 | 2025-11-24 10:30:00 | 187.33 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-26 10:35:00 | 192.12 | 2025-11-26 10:40:00 | 191.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-28 09:50:00 | 195.92 | 2025-11-28 10:00:00 | 195.45 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-03 11:15:00 | 198.22 | 2025-12-03 11:30:00 | 198.77 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-04 09:35:00 | 202.58 | 2025-12-04 11:40:00 | 201.89 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-05 09:55:00 | 194.36 | 2025-12-05 10:00:00 | 195.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-08 11:00:00 | 194.48 | 2025-12-08 11:20:00 | 194.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-10 11:05:00 | 196.74 | 2025-12-10 13:00:00 | 197.72 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-12-10 11:05:00 | 196.74 | 2025-12-10 14:15:00 | 196.74 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-11 10:15:00 | 199.10 | 2025-12-11 10:50:00 | 200.15 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-12-11 10:15:00 | 199.10 | 2025-12-11 11:00:00 | 199.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 09:35:00 | 220.81 | 2025-12-23 09:40:00 | 220.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-24 09:30:00 | 222.43 | 2025-12-24 09:35:00 | 223.13 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-12-24 09:30:00 | 222.43 | 2025-12-24 09:45:00 | 222.43 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 11:10:00 | 224.49 | 2025-12-26 11:20:00 | 223.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-26 11:10:00 | 224.49 | 2025-12-26 11:30:00 | 224.49 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:35:00 | 228.86 | 2026-01-02 09:40:00 | 228.24 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-05 11:00:00 | 229.93 | 2026-01-05 11:20:00 | 229.05 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-05 11:00:00 | 229.93 | 2026-01-05 11:40:00 | 229.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 10:15:00 | 234.23 | 2026-01-06 10:35:00 | 235.12 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-06 10:15:00 | 234.23 | 2026-01-06 10:50:00 | 234.23 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-13 09:40:00 | 238.03 | 2026-01-13 09:50:00 | 239.12 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-13 09:40:00 | 238.03 | 2026-01-13 11:50:00 | 238.54 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-01-21 10:50:00 | 247.25 | 2026-01-21 10:55:00 | 248.27 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-01-22 09:30:00 | 256.93 | 2026-01-22 09:35:00 | 258.23 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-01-22 09:30:00 | 256.93 | 2026-01-22 09:50:00 | 256.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 09:50:00 | 260.86 | 2026-01-23 10:10:00 | 259.76 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-10 10:15:00 | 257.10 | 2026-02-10 12:10:00 | 258.35 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-10 10:15:00 | 257.10 | 2026-02-10 13:30:00 | 257.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 253.20 | 2026-02-19 12:20:00 | 253.81 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-08 11:05:00 | 270.62 | 2026-04-08 11:35:00 | 269.03 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-08 11:05:00 | 270.62 | 2026-04-08 12:25:00 | 270.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 09:30:00 | 293.43 | 2026-04-21 09:40:00 | 291.64 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-21 09:30:00 | 293.43 | 2026-04-21 15:20:00 | 286.85 | TARGET_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2026-04-27 09:45:00 | 275.84 | 2026-04-27 09:55:00 | 277.65 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-27 09:45:00 | 275.84 | 2026-04-27 10:55:00 | 275.84 | STOP_HIT | 0.50 | 0.00% |
