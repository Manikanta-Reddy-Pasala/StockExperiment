# Bajaj Holdings & Investment Ltd. (BAJAJHLDNG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 10678.00
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 15
- **Target hits / Stop hits / Partials:** 4 / 15 / 6
- **Avg / median % per leg:** 0.06% / -0.17%
- **Sum % (uncompounded):** 1.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.02% | 0.3% |
| BUY @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 2 | 8 | 3 | 0.02% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.09% | 1.1% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.09% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 10 | 40.0% | 4 | 15 | 6 | 0.06% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 11005.00 | 11061.55 | 0.00 | ORB-short ORB[11020.00,11139.00] vol=1.6x ATR=38.61 |
| Stop hit — per-position SL triggered | 2026-02-09 12:50:00 | 11043.61 | 11039.83 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 11104.00 | 11059.51 | 0.00 | ORB-long ORB[11005.00,11070.00] vol=2.4x ATR=24.99 |
| Stop hit — per-position SL triggered | 2026-02-10 10:50:00 | 11079.01 | 11061.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 11161.00 | 11086.75 | 0.00 | ORB-long ORB[11015.00,11143.00] vol=2.0x ATR=24.14 |
| Stop hit — per-position SL triggered | 2026-02-11 10:55:00 | 11136.86 | 11098.97 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 10870.00 | 10924.06 | 0.00 | ORB-short ORB[10904.00,11040.00] vol=1.7x ATR=30.64 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 10900.64 | 10909.30 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 11280.00 | 11372.56 | 0.00 | ORB-short ORB[11378.00,11460.00] vol=1.9x ATR=31.41 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 11311.41 | 11365.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:20:00 | 11279.00 | 11304.52 | 0.00 | ORB-short ORB[11310.00,11449.00] vol=3.4x ATR=32.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:25:00 | 11229.64 | 11286.73 | 0.00 | T1 1.5R @ 11229.64 |
| Target hit | 2026-02-25 15:20:00 | 11152.00 | 11230.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 10625.00 | 10659.89 | 0.00 | ORB-short ORB[10656.00,10726.00] vol=2.5x ATR=18.13 |
| Stop hit — per-position SL triggered | 2026-03-05 11:00:00 | 10643.13 | 10657.86 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 10282.00 | 10315.43 | 0.00 | ORB-short ORB[10285.00,10419.00] vol=1.7x ATR=27.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:50:00 | 10240.01 | 10292.87 | 0.00 | T1 1.5R @ 10240.01 |
| Target hit | 2026-03-11 13:10:00 | 10250.00 | 10239.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2026-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:45:00 | 9990.00 | 10002.80 | 0.00 | ORB-short ORB[10015.00,10103.00] vol=1.9x ATR=31.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:05:00 | 9942.49 | 9994.90 | 0.00 | T1 1.5R @ 9942.49 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 9990.00 | 9989.17 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 9820.00 | 9758.23 | 0.00 | ORB-long ORB[9657.00,9790.00] vol=1.6x ATR=29.02 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 9790.98 | 9767.05 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 9693.00 | 9733.99 | 0.00 | ORB-short ORB[9703.00,9790.00] vol=1.7x ATR=18.15 |
| Stop hit — per-position SL triggered | 2026-03-20 11:20:00 | 9711.15 | 9731.42 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:55:00 | 9990.00 | 9892.79 | 0.00 | ORB-long ORB[9795.00,9936.50] vol=1.6x ATR=40.61 |
| Stop hit — per-position SL triggered | 2026-04-13 10:15:00 | 9949.39 | 9906.79 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 10223.50 | 10155.53 | 0.00 | ORB-long ORB[10098.00,10199.00] vol=1.6x ATR=34.51 |
| Stop hit — per-position SL triggered | 2026-04-16 10:20:00 | 10188.99 | 10185.11 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:45:00 | 10313.50 | 10261.80 | 0.00 | ORB-long ORB[10150.50,10289.50] vol=1.8x ATR=34.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 14:10:00 | 10365.33 | 10302.78 | 0.00 | T1 1.5R @ 10365.33 |
| Target hit | 2026-04-17 15:20:00 | 10372.00 | 10312.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 10420.00 | 10372.18 | 0.00 | ORB-long ORB[10252.00,10403.50] vol=1.8x ATR=24.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:50:00 | 10456.83 | 10394.21 | 0.00 | T1 1.5R @ 10456.83 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 10420.00 | 10408.66 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 10264.00 | 10344.99 | 0.00 | ORB-short ORB[10317.00,10445.50] vol=2.1x ATR=37.88 |
| Stop hit — per-position SL triggered | 2026-04-24 09:45:00 | 10301.88 | 10319.20 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:25:00 | 10394.50 | 10358.57 | 0.00 | ORB-long ORB[10255.50,10334.00] vol=3.0x ATR=30.45 |
| Stop hit — per-position SL triggered | 2026-04-27 11:35:00 | 10364.05 | 10365.22 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 10360.50 | 10338.81 | 0.00 | ORB-long ORB[10260.00,10350.00] vol=4.3x ATR=22.62 |
| Stop hit — per-position SL triggered | 2026-04-28 11:55:00 | 10337.88 | 10344.68 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 10660.00 | 10609.97 | 0.00 | ORB-long ORB[10528.00,10650.00] vol=2.1x ATR=31.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:45:00 | 10707.90 | 10668.42 | 0.00 | T1 1.5R @ 10707.90 |
| Target hit | 2026-05-06 10:30:00 | 10700.00 | 10724.44 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 11:00:00 | 11005.00 | 2026-02-09 12:50:00 | 11043.61 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-10 10:35:00 | 11104.00 | 2026-02-10 10:50:00 | 11079.01 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-11 10:35:00 | 11161.00 | 2026-02-11 10:55:00 | 11136.86 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-13 09:30:00 | 10870.00 | 2026-02-13 09:55:00 | 10900.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-19 09:40:00 | 11280.00 | 2026-02-19 09:45:00 | 11311.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-25 10:20:00 | 11279.00 | 2026-02-25 10:25:00 | 11229.64 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-25 10:20:00 | 11279.00 | 2026-02-25 15:20:00 | 11152.00 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-03-05 10:50:00 | 10625.00 | 2026-03-05 11:00:00 | 10643.13 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-03-11 10:10:00 | 10282.00 | 2026-03-11 10:50:00 | 10240.01 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-11 10:10:00 | 10282.00 | 2026-03-11 13:10:00 | 10250.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-12 10:45:00 | 9990.00 | 2026-03-12 11:05:00 | 9942.49 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-12 10:45:00 | 9990.00 | 2026-03-12 11:15:00 | 9990.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:30:00 | 9820.00 | 2026-03-18 09:45:00 | 9790.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-20 11:05:00 | 9693.00 | 2026-03-20 11:20:00 | 9711.15 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-13 09:55:00 | 9990.00 | 2026-04-13 10:15:00 | 9949.39 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-16 09:45:00 | 10223.50 | 2026-04-16 10:20:00 | 10188.99 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-17 09:45:00 | 10313.50 | 2026-04-17 14:10:00 | 10365.33 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-17 09:45:00 | 10313.50 | 2026-04-17 15:20:00 | 10372.00 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-04-23 09:45:00 | 10420.00 | 2026-04-23 09:50:00 | 10456.83 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-23 09:45:00 | 10420.00 | 2026-04-23 10:15:00 | 10420.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:30:00 | 10264.00 | 2026-04-24 09:45:00 | 10301.88 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-27 10:25:00 | 10394.50 | 2026-04-27 11:35:00 | 10364.05 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-28 11:00:00 | 10360.50 | 2026-04-28 11:55:00 | 10337.88 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-06 09:35:00 | 10660.00 | 2026-05-06 09:45:00 | 10707.90 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-05-06 09:35:00 | 10660.00 | 2026-05-06 10:30:00 | 10700.00 | TARGET_HIT | 0.50 | 0.38% |
