# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 37365.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 10
- **Avg / median % per leg:** 0.14% / 0.21%
- **Sum % (uncompounded):** 3.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.15% | 1.1% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 0.15% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 10 | 47.6% | 2 | 11 | 8 | 0.14% | 2.8% |
| SELL @ 2nd Alert (retest1) | 21 | 10 | 47.6% | 2 | 11 | 8 | 0.14% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 14 | 50.0% | 4 | 14 | 10 | 0.14% | 3.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 35200.00 | 35265.97 | 0.00 | ORB-short ORB[35375.00,35730.00] vol=4.6x ATR=151.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 34972.45 | 35236.59 | 0.00 | T1 1.5R @ 34972.45 |
| Target hit | 2026-02-09 14:20:00 | 34910.00 | 34904.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:15:00 | 34540.00 | 34740.89 | 0.00 | ORB-short ORB[34855.00,35060.00] vol=4.4x ATR=80.99 |
| Stop hit — per-position SL triggered | 2026-02-10 12:00:00 | 34620.99 | 34722.45 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 34315.00 | 34414.44 | 0.00 | ORB-short ORB[34365.00,34660.00] vol=2.4x ATR=69.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:15:00 | 34210.44 | 34391.77 | 0.00 | T1 1.5R @ 34210.44 |
| Stop hit — per-position SL triggered | 2026-02-11 12:20:00 | 34315.00 | 34227.97 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:55:00 | 33460.00 | 33683.51 | 0.00 | ORB-short ORB[33650.00,33950.00] vol=1.8x ATR=78.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:15:00 | 33343.00 | 33588.61 | 0.00 | T1 1.5R @ 33343.00 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 33460.00 | 33565.33 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:00:00 | 33260.00 | 33304.45 | 0.00 | ORB-short ORB[33275.00,33600.00] vol=2.5x ATR=77.44 |
| Stop hit — per-position SL triggered | 2026-02-19 10:05:00 | 33337.44 | 33302.42 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 32385.00 | 32545.22 | 0.00 | ORB-short ORB[32550.00,32795.00] vol=2.9x ATR=57.21 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 32442.21 | 32520.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 32030.00 | 32146.82 | 0.00 | ORB-short ORB[32040.00,32410.00] vol=1.5x ATR=87.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 31898.97 | 32013.87 | 0.00 | T1 1.5R @ 31898.97 |
| Target hit | 2026-02-27 13:50:00 | 31950.00 | 31893.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2026-03-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 09:35:00 | 31350.00 | 31083.63 | 0.00 | ORB-long ORB[30820.00,31125.00] vol=2.0x ATR=147.57 |
| Stop hit — per-position SL triggered | 2026-03-09 09:40:00 | 31202.43 | 31110.96 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 30470.00 | 30425.84 | 0.00 | ORB-long ORB[30200.00,30450.00] vol=1.7x ATR=64.55 |
| Stop hit — per-position SL triggered | 2026-03-17 11:30:00 | 30405.45 | 30428.76 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:15:00 | 31065.00 | 31213.06 | 0.00 | ORB-short ORB[31155.00,31555.00] vol=4.6x ATR=81.53 |
| Stop hit — per-position SL triggered | 2026-03-19 11:30:00 | 31146.53 | 31205.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:05:00 | 31470.00 | 31495.62 | 0.00 | ORB-short ORB[31530.00,31845.00] vol=1.6x ATR=132.70 |
| Stop hit — per-position SL triggered | 2026-03-23 13:50:00 | 31602.70 | 31465.25 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:20:00 | 32930.00 | 32572.25 | 0.00 | ORB-long ORB[32180.00,32595.00] vol=1.5x ATR=119.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:00:00 | 33109.25 | 32709.32 | 0.00 | T1 1.5R @ 33109.25 |
| Target hit | 2026-03-25 14:15:00 | 33000.00 | 33019.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2026-04-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:40:00 | 36095.00 | 35941.70 | 0.00 | ORB-long ORB[35505.00,36020.00] vol=1.8x ATR=158.25 |
| Stop hit — per-position SL triggered | 2026-04-13 15:00:00 | 35936.75 | 36077.19 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 37610.00 | 37339.91 | 0.00 | ORB-long ORB[37080.00,37425.00] vol=2.4x ATR=129.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 37804.00 | 37477.93 | 0.00 | T1 1.5R @ 37804.00 |
| Target hit | 2026-04-17 15:05:00 | 37955.00 | 37995.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 37870.00 | 38044.07 | 0.00 | ORB-short ORB[37900.00,38250.00] vol=1.6x ATR=116.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:25:00 | 37695.25 | 38005.39 | 0.00 | T1 1.5R @ 37695.25 |
| Stop hit — per-position SL triggered | 2026-04-24 11:35:00 | 37870.00 | 37988.30 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 37490.00 | 37627.69 | 0.00 | ORB-short ORB[37540.00,37800.00] vol=1.7x ATR=78.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 12:15:00 | 37372.39 | 37576.41 | 0.00 | T1 1.5R @ 37372.39 |
| Stop hit — per-position SL triggered | 2026-04-28 13:40:00 | 37490.00 | 37479.26 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 37285.00 | 37419.92 | 0.00 | ORB-short ORB[37405.00,37700.00] vol=1.7x ATR=67.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:20:00 | 37183.69 | 37411.29 | 0.00 | T1 1.5R @ 37183.69 |
| Stop hit — per-position SL triggered | 2026-05-07 11:25:00 | 37285.00 | 37409.67 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:35:00 | 36995.00 | 37167.67 | 0.00 | ORB-short ORB[37100.00,37475.00] vol=2.3x ATR=83.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:20:00 | 36870.46 | 37080.32 | 0.00 | T1 1.5R @ 36870.46 |
| Stop hit — per-position SL triggered | 2026-05-08 12:35:00 | 36995.00 | 37073.17 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:40:00 | 35200.00 | 2026-02-09 11:00:00 | 34972.45 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-02-09 10:40:00 | 35200.00 | 2026-02-09 14:20:00 | 34910.00 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2026-02-10 11:15:00 | 34540.00 | 2026-02-10 12:00:00 | 34620.99 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-11 10:35:00 | 34315.00 | 2026-02-11 11:15:00 | 34210.44 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-11 10:35:00 | 34315.00 | 2026-02-11 12:20:00 | 34315.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 09:55:00 | 33460.00 | 2026-02-18 10:15:00 | 33343.00 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-18 09:55:00 | 33460.00 | 2026-02-18 10:35:00 | 33460.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:00:00 | 33260.00 | 2026-02-19 10:05:00 | 33337.44 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-26 10:55:00 | 32385.00 | 2026-02-26 11:25:00 | 32442.21 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-27 09:30:00 | 32030.00 | 2026-02-27 10:15:00 | 31898.97 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-27 09:30:00 | 32030.00 | 2026-02-27 13:50:00 | 31950.00 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2026-03-09 09:35:00 | 31350.00 | 2026-03-09 09:40:00 | 31202.43 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-17 11:15:00 | 30470.00 | 2026-03-17 11:30:00 | 30405.45 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-19 11:15:00 | 31065.00 | 2026-03-19 11:30:00 | 31146.53 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-23 11:05:00 | 31470.00 | 2026-03-23 13:50:00 | 31602.70 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 10:20:00 | 32930.00 | 2026-03-25 11:00:00 | 33109.25 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-03-25 10:20:00 | 32930.00 | 2026-03-25 14:15:00 | 33000.00 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2026-04-13 10:40:00 | 36095.00 | 2026-04-13 15:00:00 | 35936.75 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-17 09:35:00 | 37610.00 | 2026-04-17 10:00:00 | 37804.00 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-17 09:35:00 | 37610.00 | 2026-04-17 15:05:00 | 37955.00 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2026-04-24 10:55:00 | 37870.00 | 2026-04-24 11:25:00 | 37695.25 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-04-24 10:55:00 | 37870.00 | 2026-04-24 11:35:00 | 37870.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 11:05:00 | 37490.00 | 2026-04-28 12:15:00 | 37372.39 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-28 11:05:00 | 37490.00 | 2026-04-28 13:40:00 | 37490.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 11:10:00 | 37285.00 | 2026-05-07 11:20:00 | 37183.69 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-05-07 11:10:00 | 37285.00 | 2026-05-07 11:25:00 | 37285.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:35:00 | 36995.00 | 2026-05-08 12:20:00 | 36870.46 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-05-08 10:35:00 | 36995.00 | 2026-05-08 12:35:00 | 36995.00 | STOP_HIT | 0.50 | 0.00% |
