# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-10-07 15:25:00 (7800 bars)
- **Last close:** 171.50
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
| ENTRY1 | 29 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 25
- **Target hits / Stop hits / Partials:** 4 / 25 / 8
- **Avg / median % per leg:** -0.00% / -0.17%
- **Sum % (uncompounded):** -0.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 10 | 43.5% | 4 | 13 | 6 | 0.06% | 1.5% |
| BUY @ 2nd Alert (retest1) | 23 | 10 | 43.5% | 4 | 13 | 6 | 0.06% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.11% | -1.5% |
| SELL @ 2nd Alert (retest1) | 14 | 2 | 14.3% | 0 | 12 | 2 | -0.11% | -1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 37 | 12 | 32.4% | 4 | 25 | 8 | -0.00% | -0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:35:00 | 157.36 | 155.74 | 0.00 | ORB-long ORB[154.30,156.29] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-05-15 10:45:00 | 156.80 | 155.93 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 11:15:00 | 163.16 | 162.37 | 0.00 | ORB-long ORB[160.36,162.80] vol=1.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-05-22 11:20:00 | 162.79 | 162.38 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 165.04 | 164.33 | 0.00 | ORB-long ORB[163.10,164.68] vol=1.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-05-26 09:55:00 | 164.62 | 164.64 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 163.03 | 162.01 | 0.00 | ORB-long ORB[161.63,162.68] vol=1.9x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 11:25:00 | 163.60 | 162.27 | 0.00 | T1 1.5R @ 163.60 |
| Stop hit — per-position SL triggered | 2025-05-27 11:45:00 | 163.03 | 162.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:55:00 | 157.18 | 158.09 | 0.00 | ORB-short ORB[157.59,159.30] vol=2.0x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-06-10 10:10:00 | 157.58 | 157.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 154.94 | 155.43 | 0.00 | ORB-short ORB[155.04,156.26] vol=1.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2025-06-12 10:30:00 | 155.30 | 155.20 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:50:00 | 153.16 | 152.12 | 0.00 | ORB-long ORB[151.32,153.06] vol=3.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 12:30:00 | 153.80 | 152.55 | 0.00 | T1 1.5R @ 153.80 |
| Target hit | 2025-06-16 15:20:00 | 154.12 | 153.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 154.56 | 153.95 | 0.00 | ORB-long ORB[153.17,154.35] vol=1.7x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-06-17 09:45:00 | 154.21 | 154.13 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:55:00 | 155.00 | 154.35 | 0.00 | ORB-long ORB[153.64,154.80] vol=2.0x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:05:00 | 155.46 | 154.46 | 0.00 | T1 1.5R @ 155.46 |
| Target hit | 2025-06-24 13:20:00 | 155.26 | 155.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 158.10 | 158.77 | 0.00 | ORB-short ORB[158.65,160.44] vol=1.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-07-01 11:40:00 | 158.45 | 158.61 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:45:00 | 160.19 | 160.47 | 0.00 | ORB-short ORB[160.35,161.95] vol=5.0x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-07-09 09:55:00 | 160.53 | 160.46 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 161.40 | 160.94 | 0.00 | ORB-long ORB[159.80,161.25] vol=1.5x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-07-11 09:55:00 | 161.05 | 161.02 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:40:00 | 158.82 | 159.78 | 0.00 | ORB-short ORB[159.90,160.85] vol=2.3x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-07-15 09:45:00 | 159.19 | 159.68 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:15:00 | 161.50 | 162.12 | 0.00 | ORB-short ORB[161.90,163.51] vol=2.4x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-07-25 10:20:00 | 161.83 | 162.07 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:50:00 | 161.00 | 160.39 | 0.00 | ORB-long ORB[158.50,160.90] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-08-12 10:30:00 | 160.49 | 160.70 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:30:00 | 161.83 | 161.33 | 0.00 | ORB-long ORB[160.37,161.65] vol=1.6x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-08-13 10:45:00 | 161.45 | 161.76 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:40:00 | 160.04 | 159.19 | 0.00 | ORB-long ORB[158.13,159.41] vol=6.7x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:45:00 | 160.55 | 159.46 | 0.00 | T1 1.5R @ 160.55 |
| Target hit | 2025-08-20 15:20:00 | 161.86 | 161.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 160.17 | 160.47 | 0.00 | ORB-short ORB[160.48,161.28] vol=1.6x ATR=0.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 11:30:00 | 159.80 | 160.43 | 0.00 | T1 1.5R @ 159.80 |
| Stop hit — per-position SL triggered | 2025-08-22 11:40:00 | 160.17 | 160.40 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:40:00 | 156.77 | 157.54 | 0.00 | ORB-short ORB[157.30,159.16] vol=3.1x ATR=0.35 |
| Stop hit — per-position SL triggered | 2025-08-26 11:35:00 | 157.12 | 157.40 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:40:00 | 157.19 | 156.40 | 0.00 | ORB-long ORB[155.50,156.59] vol=2.1x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:25:00 | 157.64 | 156.75 | 0.00 | T1 1.5R @ 157.64 |
| Target hit | 2025-09-02 14:20:00 | 157.70 | 157.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2025-09-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:05:00 | 170.59 | 169.50 | 0.00 | ORB-long ORB[168.75,170.19] vol=2.5x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 170.16 | 169.61 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:15:00 | 170.20 | 169.49 | 0.00 | ORB-long ORB[169.01,169.85] vol=2.2x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 10:20:00 | 170.64 | 169.69 | 0.00 | T1 1.5R @ 170.64 |
| Stop hit — per-position SL triggered | 2025-09-11 10:55:00 | 170.20 | 169.86 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:05:00 | 169.50 | 169.90 | 0.00 | ORB-short ORB[169.60,171.00] vol=1.6x ATR=0.26 |
| Stop hit — per-position SL triggered | 2025-09-12 11:10:00 | 169.76 | 169.88 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:30:00 | 169.04 | 169.54 | 0.00 | ORB-short ORB[169.31,170.20] vol=1.8x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-09-15 09:45:00 | 169.33 | 169.44 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:00:00 | 170.59 | 171.26 | 0.00 | ORB-short ORB[171.35,172.90] vol=2.0x ATR=0.29 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 170.88 | 171.24 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:45:00 | 172.30 | 171.96 | 0.00 | ORB-long ORB[171.42,172.29] vol=2.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-09-19 09:50:00 | 171.98 | 171.97 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:30:00 | 173.90 | 173.25 | 0.00 | ORB-long ORB[172.38,173.29] vol=2.5x ATR=0.42 |
| Stop hit — per-position SL triggered | 2025-09-24 10:45:00 | 173.48 | 173.67 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:05:00 | 170.01 | 171.16 | 0.00 | ORB-short ORB[171.32,173.48] vol=2.4x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:25:00 | 169.39 | 170.90 | 0.00 | T1 1.5R @ 169.39 |
| Stop hit — per-position SL triggered | 2025-10-06 12:05:00 | 170.01 | 170.79 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 171.75 | 171.11 | 0.00 | ORB-long ORB[170.40,171.28] vol=2.1x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-10-07 09:40:00 | 171.38 | 171.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:35:00 | 157.36 | 2025-05-15 10:45:00 | 156.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-05-22 11:15:00 | 163.16 | 2025-05-22 11:20:00 | 162.79 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-05-26 09:30:00 | 165.04 | 2025-05-26 09:55:00 | 164.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-27 11:00:00 | 163.03 | 2025-05-27 11:25:00 | 163.60 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-27 11:00:00 | 163.03 | 2025-05-27 11:45:00 | 163.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-10 09:55:00 | 157.18 | 2025-06-10 10:10:00 | 157.58 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-12 09:40:00 | 154.94 | 2025-06-12 10:30:00 | 155.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-16 10:50:00 | 153.16 | 2025-06-16 12:30:00 | 153.80 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-16 10:50:00 | 153.16 | 2025-06-16 15:20:00 | 154.12 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-06-17 09:30:00 | 154.56 | 2025-06-17 09:45:00 | 154.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-24 10:55:00 | 155.00 | 2025-06-24 11:05:00 | 155.46 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-06-24 10:55:00 | 155.00 | 2025-06-24 13:20:00 | 155.26 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-07-01 10:55:00 | 158.10 | 2025-07-01 11:40:00 | 158.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-09 09:45:00 | 160.19 | 2025-07-09 09:55:00 | 160.53 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-11 09:40:00 | 161.40 | 2025-07-11 09:55:00 | 161.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-15 09:40:00 | 158.82 | 2025-07-15 09:45:00 | 159.19 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-25 10:15:00 | 161.50 | 2025-07-25 10:20:00 | 161.83 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-12 09:50:00 | 161.00 | 2025-08-12 10:30:00 | 160.49 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-13 09:30:00 | 161.83 | 2025-08-13 10:45:00 | 161.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-20 10:40:00 | 160.04 | 2025-08-20 10:45:00 | 160.55 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-08-20 10:40:00 | 160.04 | 2025-08-20 15:20:00 | 161.86 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-08-22 11:00:00 | 160.17 | 2025-08-22 11:30:00 | 159.80 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-08-22 11:00:00 | 160.17 | 2025-08-22 11:40:00 | 160.17 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 10:40:00 | 156.77 | 2025-08-26 11:35:00 | 157.12 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-02 09:40:00 | 157.19 | 2025-09-02 10:25:00 | 157.64 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-02 09:40:00 | 157.19 | 2025-09-02 14:20:00 | 157.70 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-10 10:05:00 | 170.59 | 2025-09-10 10:15:00 | 170.16 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-11 10:15:00 | 170.20 | 2025-09-11 10:20:00 | 170.64 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-09-11 10:15:00 | 170.20 | 2025-09-11 10:55:00 | 170.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 11:05:00 | 169.50 | 2025-09-12 11:10:00 | 169.76 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-15 09:30:00 | 169.04 | 2025-09-15 09:45:00 | 169.33 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-17 11:00:00 | 170.59 | 2025-09-17 11:15:00 | 170.88 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-19 09:45:00 | 172.30 | 2025-09-19 09:50:00 | 171.98 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-24 09:30:00 | 173.90 | 2025-09-24 10:45:00 | 173.48 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-06 11:05:00 | 170.01 | 2025-10-06 11:25:00 | 169.39 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-06 11:05:00 | 170.01 | 2025-10-06 12:05:00 | 170.01 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-07 09:30:00 | 171.75 | 2025-10-07 09:40:00 | 171.38 | STOP_HIT | 1.00 | -0.21% |
