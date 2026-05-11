# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 10696.50
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 15 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 72
- **Target hits / Stop hits / Partials:** 15 / 72 / 31
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 14.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 22 | 35.5% | 7 | 40 | 15 | 0.08% | 5.0% |
| BUY @ 2nd Alert (retest1) | 62 | 22 | 35.5% | 7 | 40 | 15 | 0.08% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 24 | 42.9% | 8 | 32 | 16 | 0.17% | 9.5% |
| SELL @ 2nd Alert (retest1) | 56 | 24 | 42.9% | 8 | 32 | 16 | 0.17% | 9.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 118 | 46 | 39.0% | 15 | 72 | 31 | 0.12% | 14.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:05:00 | 8898.25 | 8932.54 | 0.00 | ORB-short ORB[8930.80,9025.00] vol=1.9x ATR=35.30 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 8933.55 | 8931.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 9029.00 | 9055.42 | 0.00 | ORB-short ORB[9065.25,9125.00] vol=2.1x ATR=24.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:00:00 | 8992.54 | 9047.99 | 0.00 | T1 1.5R @ 8992.54 |
| Target hit | 2024-05-15 15:20:00 | 8889.55 | 8965.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 8852.05 | 8902.58 | 0.00 | ORB-short ORB[8885.00,8955.55] vol=1.7x ATR=20.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:35:00 | 8821.15 | 8875.17 | 0.00 | T1 1.5R @ 8821.15 |
| Stop hit — per-position SL triggered | 2024-05-16 09:40:00 | 8852.05 | 8870.98 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 8828.20 | 8879.75 | 0.00 | ORB-short ORB[8871.85,8958.35] vol=1.5x ATR=21.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 11:25:00 | 8796.10 | 8866.59 | 0.00 | T1 1.5R @ 8796.10 |
| Target hit | 2024-05-17 15:20:00 | 8779.35 | 8820.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-06-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:10:00 | 9850.00 | 9866.08 | 0.00 | ORB-short ORB[9868.05,9940.00] vol=2.0x ATR=18.66 |
| Stop hit — per-position SL triggered | 2024-06-13 12:50:00 | 9868.66 | 9861.62 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 11:10:00 | 9945.75 | 9870.87 | 0.00 | ORB-long ORB[9813.05,9869.60] vol=2.5x ATR=20.63 |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 9925.12 | 9873.66 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:30:00 | 10030.00 | 9982.22 | 0.00 | ORB-long ORB[9922.00,10009.95] vol=1.8x ATR=28.72 |
| Stop hit — per-position SL triggered | 2024-06-18 09:40:00 | 10001.28 | 9988.17 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 9947.30 | 9891.11 | 0.00 | ORB-long ORB[9848.00,9939.50] vol=2.8x ATR=24.57 |
| Stop hit — per-position SL triggered | 2024-06-19 09:35:00 | 9922.73 | 9894.76 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 11:10:00 | 9697.50 | 9666.23 | 0.00 | ORB-long ORB[9613.10,9676.00] vol=1.7x ATR=22.24 |
| Stop hit — per-position SL triggered | 2024-06-21 11:40:00 | 9675.26 | 9668.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 10:45:00 | 9665.10 | 9629.86 | 0.00 | ORB-long ORB[9510.55,9595.00] vol=1.5x ATR=25.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 11:35:00 | 9703.55 | 9640.00 | 0.00 | T1 1.5R @ 9703.55 |
| Target hit | 2024-06-24 15:20:00 | 9742.75 | 9701.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 9468.00 | 9526.36 | 0.00 | ORB-short ORB[9548.00,9660.30] vol=1.8x ATR=19.94 |
| Stop hit — per-position SL triggered | 2024-06-26 11:45:00 | 9487.94 | 9508.70 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:00:00 | 9546.90 | 9469.93 | 0.00 | ORB-long ORB[9404.00,9501.25] vol=1.5x ATR=20.82 |
| Stop hit — per-position SL triggered | 2024-07-03 10:10:00 | 9526.08 | 9481.94 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:10:00 | 9503.45 | 9520.60 | 0.00 | ORB-short ORB[9511.95,9565.00] vol=1.7x ATR=15.51 |
| Stop hit — per-position SL triggered | 2024-07-09 11:20:00 | 9518.96 | 9520.23 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 9560.15 | 9518.23 | 0.00 | ORB-long ORB[9403.95,9544.00] vol=2.8x ATR=23.89 |
| Stop hit — per-position SL triggered | 2024-07-15 09:35:00 | 9536.26 | 9522.29 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 11:00:00 | 9398.05 | 9365.29 | 0.00 | ORB-long ORB[9263.20,9360.00] vol=1.6x ATR=26.81 |
| Stop hit — per-position SL triggered | 2024-07-22 11:15:00 | 9371.24 | 9366.44 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:05:00 | 9460.00 | 9408.48 | 0.00 | ORB-long ORB[9350.00,9454.95] vol=2.8x ATR=23.09 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 9436.91 | 9413.18 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 11:00:00 | 9288.70 | 9321.98 | 0.00 | ORB-short ORB[9313.00,9372.70] vol=2.6x ATR=25.52 |
| Stop hit — per-position SL triggered | 2024-07-24 11:40:00 | 9314.22 | 9313.26 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:55:00 | 9407.80 | 9344.78 | 0.00 | ORB-long ORB[9268.45,9319.00] vol=1.7x ATR=24.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:40:00 | 9444.26 | 9381.62 | 0.00 | T1 1.5R @ 9444.26 |
| Target hit | 2024-07-26 15:20:00 | 9496.20 | 9442.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-07-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:25:00 | 9582.25 | 9560.54 | 0.00 | ORB-long ORB[9512.00,9579.15] vol=1.8x ATR=16.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:00:00 | 9607.68 | 9574.30 | 0.00 | T1 1.5R @ 9607.68 |
| Stop hit — per-position SL triggered | 2024-07-30 12:05:00 | 9582.25 | 9592.21 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:45:00 | 9657.65 | 9620.95 | 0.00 | ORB-long ORB[9590.05,9634.95] vol=2.9x ATR=15.57 |
| Stop hit — per-position SL triggered | 2024-07-31 10:50:00 | 9642.08 | 9621.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 10:25:00 | 9620.50 | 9655.72 | 0.00 | ORB-short ORB[9660.50,9750.95] vol=5.1x ATR=21.07 |
| Stop hit — per-position SL triggered | 2024-08-12 10:35:00 | 9641.57 | 9654.40 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 9848.00 | 9816.25 | 0.00 | ORB-long ORB[9741.15,9836.95] vol=2.0x ATR=26.91 |
| Stop hit — per-position SL triggered | 2024-08-16 09:35:00 | 9821.09 | 9817.70 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:25:00 | 9955.00 | 9894.47 | 0.00 | ORB-long ORB[9850.85,9912.35] vol=4.0x ATR=25.51 |
| Stop hit — per-position SL triggered | 2024-08-22 10:30:00 | 9929.49 | 9903.05 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 10463.15 | 10387.86 | 0.00 | ORB-long ORB[10350.00,10418.95] vol=1.9x ATR=24.95 |
| Stop hit — per-position SL triggered | 2024-08-27 09:45:00 | 10438.20 | 10400.41 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:35:00 | 10567.80 | 10515.71 | 0.00 | ORB-long ORB[10480.00,10558.35] vol=1.6x ATR=25.02 |
| Stop hit — per-position SL triggered | 2024-08-28 10:40:00 | 10542.78 | 10516.85 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:30:00 | 10814.20 | 10857.69 | 0.00 | ORB-short ORB[10833.00,10913.95] vol=1.8x ATR=29.25 |
| Stop hit — per-position SL triggered | 2024-09-06 10:40:00 | 10843.45 | 10853.72 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 10873.70 | 10834.13 | 0.00 | ORB-long ORB[10780.05,10866.70] vol=1.8x ATR=24.17 |
| Stop hit — per-position SL triggered | 2024-09-10 09:40:00 | 10849.53 | 10839.26 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 11228.15 | 11102.07 | 0.00 | ORB-long ORB[10986.60,11121.30] vol=2.7x ATR=45.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:45:00 | 11296.09 | 11165.43 | 0.00 | T1 1.5R @ 11296.09 |
| Target hit | 2024-09-11 15:20:00 | 11426.45 | 11361.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-09-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:10:00 | 11723.40 | 11576.45 | 0.00 | ORB-long ORB[11443.35,11548.80] vol=2.0x ATR=42.51 |
| Stop hit — per-position SL triggered | 2024-09-12 12:15:00 | 11680.89 | 11670.77 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:50:00 | 11800.00 | 11729.57 | 0.00 | ORB-long ORB[11643.10,11759.40] vol=2.0x ATR=28.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:55:00 | 11843.44 | 11749.53 | 0.00 | T1 1.5R @ 11843.44 |
| Target hit | 2024-09-17 15:20:00 | 11952.90 | 11868.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 11965.90 | 11908.69 | 0.00 | ORB-long ORB[11811.20,11949.95] vol=2.0x ATR=41.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:45:00 | 12028.34 | 11940.04 | 0.00 | T1 1.5R @ 12028.34 |
| Stop hit — per-position SL triggered | 2024-09-19 09:55:00 | 11965.90 | 11952.18 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:10:00 | 11950.20 | 11947.63 | 0.00 | ORB-long ORB[11868.25,11940.00] vol=2.1x ATR=32.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:20:00 | 11998.68 | 11953.57 | 0.00 | T1 1.5R @ 11998.68 |
| Target hit | 2024-09-20 10:45:00 | 11960.00 | 11963.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — BUY (started 2024-09-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:55:00 | 12200.00 | 12128.49 | 0.00 | ORB-long ORB[12000.00,12159.90] vol=1.5x ATR=44.90 |
| Stop hit — per-position SL triggered | 2024-09-23 10:05:00 | 12155.10 | 12133.27 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:45:00 | 12658.95 | 12601.93 | 0.00 | ORB-long ORB[12551.60,12635.00] vol=2.0x ATR=27.18 |
| Stop hit — per-position SL triggered | 2024-09-27 11:25:00 | 12631.77 | 12609.08 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:40:00 | 12383.95 | 12509.69 | 0.00 | ORB-short ORB[12540.25,12651.15] vol=2.0x ATR=36.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 10:55:00 | 12329.03 | 12484.86 | 0.00 | T1 1.5R @ 12329.03 |
| Stop hit — per-position SL triggered | 2024-09-30 11:45:00 | 12383.95 | 12455.17 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 11657.90 | 11717.43 | 0.00 | ORB-short ORB[11733.60,11817.00] vol=1.7x ATR=38.70 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 11696.60 | 11712.84 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:45:00 | 11615.00 | 11494.49 | 0.00 | ORB-long ORB[11421.00,11588.80] vol=2.2x ATR=47.44 |
| Stop hit — per-position SL triggered | 2024-10-16 10:50:00 | 11567.56 | 11501.43 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:45:00 | 9900.75 | 10004.56 | 0.00 | ORB-short ORB[10000.00,10093.00] vol=1.6x ATR=41.26 |
| Stop hit — per-position SL triggered | 2024-10-21 09:55:00 | 9942.01 | 9990.34 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 11:10:00 | 10550.95 | 10441.57 | 0.00 | ORB-long ORB[10380.00,10525.00] vol=2.0x ATR=43.14 |
| Stop hit — per-position SL triggered | 2024-10-22 11:50:00 | 10507.81 | 10484.24 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:00:00 | 10298.80 | 10460.71 | 0.00 | ORB-short ORB[10455.00,10593.45] vol=1.7x ATR=39.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 12:15:00 | 10238.83 | 10400.17 | 0.00 | T1 1.5R @ 10238.83 |
| Stop hit — per-position SL triggered | 2024-10-24 12:45:00 | 10298.80 | 10384.97 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:30:00 | 9911.00 | 9968.23 | 0.00 | ORB-short ORB[9941.00,10073.95] vol=2.4x ATR=35.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:35:00 | 9857.33 | 9936.66 | 0.00 | T1 1.5R @ 9857.33 |
| Target hit | 2024-10-29 14:40:00 | 9793.05 | 9759.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 9901.00 | 9957.84 | 0.00 | ORB-short ORB[9905.05,10030.90] vol=1.8x ATR=27.96 |
| Stop hit — per-position SL triggered | 2024-10-31 11:20:00 | 9928.96 | 9956.89 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:30:00 | 9935.00 | 9878.33 | 0.00 | ORB-long ORB[9777.25,9925.80] vol=1.8x ATR=33.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:45:00 | 9984.60 | 9897.64 | 0.00 | T1 1.5R @ 9984.60 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 9935.00 | 9900.66 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:40:00 | 9963.60 | 9896.82 | 0.00 | ORB-long ORB[9800.00,9927.65] vol=2.0x ATR=29.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:55:00 | 10007.75 | 9923.60 | 0.00 | T1 1.5R @ 10007.75 |
| Stop hit — per-position SL triggered | 2024-11-11 10:10:00 | 9963.60 | 9941.07 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:40:00 | 9811.15 | 9866.86 | 0.00 | ORB-short ORB[9897.10,9999.00] vol=1.5x ATR=25.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 12:10:00 | 9772.72 | 9844.47 | 0.00 | T1 1.5R @ 9772.72 |
| Target hit | 2024-11-12 15:20:00 | 9675.95 | 9772.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:35:00 | 9573.50 | 9622.42 | 0.00 | ORB-short ORB[9603.05,9710.00] vol=3.1x ATR=33.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:00:00 | 9523.99 | 9596.43 | 0.00 | T1 1.5R @ 9523.99 |
| Stop hit — per-position SL triggered | 2024-11-13 10:50:00 | 9573.50 | 9550.94 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:45:00 | 9601.55 | 9579.43 | 0.00 | ORB-long ORB[9528.35,9594.75] vol=1.6x ATR=21.50 |
| Stop hit — per-position SL triggered | 2024-11-19 10:50:00 | 9580.05 | 9580.26 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 10:50:00 | 9541.05 | 9569.41 | 0.00 | ORB-short ORB[9551.20,9636.35] vol=2.2x ATR=23.14 |
| Stop hit — per-position SL triggered | 2024-11-25 11:55:00 | 9564.19 | 9564.35 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 9221.90 | 9188.25 | 0.00 | ORB-long ORB[9103.00,9210.00] vol=1.7x ATR=26.25 |
| Stop hit — per-position SL triggered | 2024-11-27 10:10:00 | 9195.65 | 9206.56 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:25:00 | 9230.85 | 9203.58 | 0.00 | ORB-long ORB[9114.50,9208.00] vol=2.1x ATR=24.74 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 9206.11 | 9204.17 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 9156.00 | 9115.70 | 0.00 | ORB-long ORB[9065.90,9125.00] vol=1.5x ATR=26.49 |
| Stop hit — per-position SL triggered | 2024-12-02 10:25:00 | 9129.51 | 9131.03 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:15:00 | 9202.95 | 9163.18 | 0.00 | ORB-long ORB[9105.00,9179.90] vol=2.1x ATR=20.21 |
| Stop hit — per-position SL triggered | 2024-12-03 10:25:00 | 9182.74 | 9166.56 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 8919.45 | 8957.90 | 0.00 | ORB-short ORB[8934.50,9044.15] vol=1.5x ATR=20.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:45:00 | 8888.90 | 8935.58 | 0.00 | T1 1.5R @ 8888.90 |
| Target hit | 2024-12-05 13:10:00 | 8817.00 | 8814.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — SELL (started 2024-12-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:40:00 | 9007.85 | 9022.69 | 0.00 | ORB-short ORB[9015.55,9078.75] vol=1.9x ATR=16.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:10:00 | 8983.22 | 9014.01 | 0.00 | T1 1.5R @ 8983.22 |
| Stop hit — per-position SL triggered | 2024-12-16 11:40:00 | 9007.85 | 9010.22 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:45:00 | 8995.00 | 8916.94 | 0.00 | ORB-long ORB[8800.00,8909.10] vol=1.7x ATR=22.13 |
| Stop hit — per-position SL triggered | 2024-12-18 10:50:00 | 8972.87 | 8919.67 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:55:00 | 8865.80 | 8827.76 | 0.00 | ORB-long ORB[8802.35,8863.70] vol=1.7x ATR=19.40 |
| Stop hit — per-position SL triggered | 2024-12-26 11:05:00 | 8846.40 | 8831.12 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:35:00 | 8823.20 | 8862.25 | 0.00 | ORB-short ORB[8890.00,8962.45] vol=1.7x ATR=20.67 |
| Stop hit — per-position SL triggered | 2024-12-30 10:40:00 | 8843.87 | 8858.28 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 8720.00 | 8751.88 | 0.00 | ORB-short ORB[8755.00,8830.95] vol=3.1x ATR=14.60 |
| Stop hit — per-position SL triggered | 2025-01-08 11:30:00 | 8734.60 | 8749.16 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 11:15:00 | 8738.60 | 8740.11 | 0.00 | ORB-short ORB[8755.85,8855.00] vol=2.4x ATR=25.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 11:25:00 | 8700.32 | 8737.37 | 0.00 | T1 1.5R @ 8700.32 |
| Stop hit — per-position SL triggered | 2025-01-10 11:30:00 | 8738.60 | 8737.57 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:00:00 | 8553.40 | 8589.39 | 0.00 | ORB-short ORB[8578.55,8639.80] vol=2.7x ATR=14.78 |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 8568.18 | 8586.46 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:35:00 | 8609.50 | 8584.74 | 0.00 | ORB-long ORB[8531.30,8583.90] vol=1.7x ATR=22.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:25:00 | 8642.77 | 8597.99 | 0.00 | T1 1.5R @ 8642.77 |
| Stop hit — per-position SL triggered | 2025-01-17 10:45:00 | 8609.50 | 8605.43 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 8506.70 | 8536.05 | 0.00 | ORB-short ORB[8520.05,8609.60] vol=2.5x ATR=18.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:40:00 | 8478.84 | 8520.93 | 0.00 | T1 1.5R @ 8478.84 |
| Stop hit — per-position SL triggered | 2025-01-20 10:30:00 | 8506.70 | 8503.11 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:40:00 | 8489.35 | 8523.55 | 0.00 | ORB-short ORB[8553.00,8608.80] vol=1.6x ATR=18.29 |
| Stop hit — per-position SL triggered | 2025-01-21 10:55:00 | 8507.64 | 8520.68 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:10:00 | 8433.00 | 8467.32 | 0.00 | ORB-short ORB[8480.00,8534.95] vol=2.5x ATR=21.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:05:00 | 8400.81 | 8449.85 | 0.00 | T1 1.5R @ 8400.81 |
| Stop hit — per-position SL triggered | 2025-01-22 12:30:00 | 8433.00 | 8445.43 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:30:00 | 8307.00 | 8356.03 | 0.00 | ORB-short ORB[8318.45,8380.40] vol=2.5x ATR=25.64 |
| Stop hit — per-position SL triggered | 2025-01-27 10:35:00 | 8332.64 | 8352.74 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:45:00 | 8906.00 | 8835.46 | 0.00 | ORB-long ORB[8744.05,8800.00] vol=3.5x ATR=26.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 10:55:00 | 8945.59 | 8849.53 | 0.00 | T1 1.5R @ 8945.59 |
| Stop hit — per-position SL triggered | 2025-01-31 11:10:00 | 8906.00 | 8856.62 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:30:00 | 8705.75 | 8747.80 | 0.00 | ORB-short ORB[8721.00,8819.40] vol=1.9x ATR=24.94 |
| Stop hit — per-position SL triggered | 2025-02-12 09:45:00 | 8730.69 | 8729.70 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:35:00 | 8603.15 | 8640.00 | 0.00 | ORB-short ORB[8656.95,8722.70] vol=3.8x ATR=21.13 |
| Stop hit — per-position SL triggered | 2025-02-14 10:45:00 | 8624.28 | 8634.93 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 11:05:00 | 8460.00 | 8472.53 | 0.00 | ORB-short ORB[8479.00,8550.00] vol=1.7x ATR=14.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:35:00 | 8438.00 | 8463.68 | 0.00 | T1 1.5R @ 8438.00 |
| Target hit | 2025-02-18 15:00:00 | 8446.00 | 8440.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — BUY (started 2025-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:55:00 | 8487.25 | 8457.28 | 0.00 | ORB-long ORB[8406.90,8470.00] vol=1.7x ATR=21.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 10:35:00 | 8519.07 | 8472.96 | 0.00 | T1 1.5R @ 8519.07 |
| Stop hit — per-position SL triggered | 2025-02-19 10:50:00 | 8487.25 | 8473.82 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 09:55:00 | 8364.90 | 8420.56 | 0.00 | ORB-short ORB[8402.05,8480.00] vol=2.4x ATR=25.16 |
| Stop hit — per-position SL triggered | 2025-02-24 10:35:00 | 8390.06 | 8401.70 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-02-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 10:40:00 | 8109.60 | 8123.07 | 0.00 | ORB-short ORB[8131.00,8234.75] vol=2.4x ATR=22.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 10:50:00 | 8075.18 | 8119.63 | 0.00 | T1 1.5R @ 8075.18 |
| Target hit | 2025-02-28 15:20:00 | 7890.00 | 8021.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2025-03-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 10:20:00 | 7405.35 | 7423.74 | 0.00 | ORB-short ORB[7413.90,7500.00] vol=1.9x ATR=22.23 |
| Stop hit — per-position SL triggered | 2025-03-06 11:05:00 | 7427.58 | 7418.62 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:05:00 | 7440.00 | 7420.06 | 0.00 | ORB-long ORB[7328.00,7439.00] vol=1.6x ATR=28.18 |
| Stop hit — per-position SL triggered | 2025-03-11 11:20:00 | 7411.82 | 7430.39 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 10:45:00 | 7556.85 | 7512.12 | 0.00 | ORB-long ORB[7452.55,7537.45] vol=3.0x ATR=20.95 |
| Stop hit — per-position SL triggered | 2025-03-12 11:25:00 | 7535.90 | 7525.87 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:55:00 | 7435.00 | 7452.91 | 0.00 | ORB-short ORB[7442.65,7545.25] vol=2.0x ATR=19.43 |
| Stop hit — per-position SL triggered | 2025-03-13 11:05:00 | 7454.43 | 7452.59 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-03-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:20:00 | 7715.10 | 7676.93 | 0.00 | ORB-long ORB[7619.25,7674.45] vol=1.8x ATR=21.43 |
| Stop hit — per-position SL triggered | 2025-03-19 10:45:00 | 7693.67 | 7683.59 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:40:00 | 7820.00 | 7782.33 | 0.00 | ORB-long ORB[7722.00,7796.20] vol=1.8x ATR=17.34 |
| Stop hit — per-position SL triggered | 2025-03-20 09:45:00 | 7802.66 | 7783.86 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 11:15:00 | 8045.05 | 8026.61 | 0.00 | ORB-long ORB[7938.00,7994.45] vol=2.0x ATR=21.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 13:10:00 | 8076.81 | 8033.97 | 0.00 | T1 1.5R @ 8076.81 |
| Target hit | 2025-03-21 15:10:00 | 8065.05 | 8072.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2025-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:45:00 | 7956.25 | 7930.46 | 0.00 | ORB-long ORB[7852.25,7949.75] vol=2.0x ATR=20.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:40:00 | 7987.26 | 7940.31 | 0.00 | T1 1.5R @ 7987.26 |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 7956.25 | 7977.19 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 10:50:00 | 8010.15 | 7972.36 | 0.00 | ORB-long ORB[7926.50,7997.70] vol=2.9x ATR=25.83 |
| Stop hit — per-position SL triggered | 2025-03-28 11:00:00 | 7984.32 | 7976.50 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:30:00 | 7846.35 | 7882.02 | 0.00 | ORB-short ORB[7853.20,7940.05] vol=2.5x ATR=31.44 |
| Stop hit — per-position SL triggered | 2025-04-03 09:40:00 | 7877.79 | 7877.53 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:45:00 | 8076.50 | 8015.05 | 0.00 | ORB-long ORB[7950.00,8050.00] vol=1.7x ATR=23.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:05:00 | 8111.34 | 8042.74 | 0.00 | T1 1.5R @ 8111.34 |
| Target hit | 2025-04-21 15:20:00 | 8247.50 | 8167.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — SELL (started 2025-04-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:40:00 | 8155.50 | 8211.38 | 0.00 | ORB-short ORB[8208.00,8259.50] vol=1.5x ATR=22.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 8121.17 | 8189.97 | 0.00 | T1 1.5R @ 8121.17 |
| Target hit | 2025-04-25 13:55:00 | 8088.50 | 8065.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 85 — SELL (started 2025-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:50:00 | 8034.50 | 8096.93 | 0.00 | ORB-short ORB[8085.00,8169.50] vol=1.5x ATR=23.54 |
| Stop hit — per-position SL triggered | 2025-04-29 11:05:00 | 8058.04 | 8091.75 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:30:00 | 8042.50 | 7993.25 | 0.00 | ORB-long ORB[7910.00,8018.50] vol=2.3x ATR=21.83 |
| Stop hit — per-position SL triggered | 2025-05-06 09:35:00 | 8020.67 | 7999.68 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 10:20:00 | 7894.00 | 7914.30 | 0.00 | ORB-short ORB[7900.00,7975.00] vol=1.7x ATR=22.79 |
| Stop hit — per-position SL triggered | 2025-05-07 10:30:00 | 7916.79 | 7913.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:05:00 | 8898.25 | 2024-05-13 11:15:00 | 8933.55 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-15 09:55:00 | 9029.00 | 2024-05-15 10:00:00 | 8992.54 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-15 09:55:00 | 9029.00 | 2024-05-15 15:20:00 | 8889.55 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2024-05-16 09:30:00 | 8852.05 | 2024-05-16 09:35:00 | 8821.15 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-16 09:30:00 | 8852.05 | 2024-05-16 09:40:00 | 8852.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-17 10:50:00 | 8828.20 | 2024-05-17 11:25:00 | 8796.10 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-17 10:50:00 | 8828.20 | 2024-05-17 15:20:00 | 8779.35 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-06-13 11:10:00 | 9850.00 | 2024-06-13 12:50:00 | 9868.66 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-06-14 11:10:00 | 9945.75 | 2024-06-14 11:15:00 | 9925.12 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-18 09:30:00 | 10030.00 | 2024-06-18 09:40:00 | 10001.28 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-19 09:30:00 | 9947.30 | 2024-06-19 09:35:00 | 9922.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-21 11:10:00 | 9697.50 | 2024-06-21 11:40:00 | 9675.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-24 10:45:00 | 9665.10 | 2024-06-24 11:35:00 | 9703.55 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-06-24 10:45:00 | 9665.10 | 2024-06-24 15:20:00 | 9742.75 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2024-06-26 11:00:00 | 9468.00 | 2024-06-26 11:45:00 | 9487.94 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-03 10:00:00 | 9546.90 | 2024-07-03 10:10:00 | 9526.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-09 11:10:00 | 9503.45 | 2024-07-09 11:20:00 | 9518.96 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-07-15 09:30:00 | 9560.15 | 2024-07-15 09:35:00 | 9536.26 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-22 11:00:00 | 9398.05 | 2024-07-22 11:15:00 | 9371.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-23 11:05:00 | 9460.00 | 2024-07-23 11:15:00 | 9436.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-24 11:00:00 | 9288.70 | 2024-07-24 11:40:00 | 9314.22 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-26 09:55:00 | 9407.80 | 2024-07-26 10:40:00 | 9444.26 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-26 09:55:00 | 9407.80 | 2024-07-26 15:20:00 | 9496.20 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2024-07-30 10:25:00 | 9582.25 | 2024-07-30 11:00:00 | 9607.68 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-07-30 10:25:00 | 9582.25 | 2024-07-30 12:05:00 | 9582.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 10:45:00 | 9657.65 | 2024-07-31 10:50:00 | 9642.08 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-08-12 10:25:00 | 9620.50 | 2024-08-12 10:35:00 | 9641.57 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-16 09:30:00 | 9848.00 | 2024-08-16 09:35:00 | 9821.09 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-22 10:25:00 | 9955.00 | 2024-08-22 10:30:00 | 9929.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-27 09:40:00 | 10463.15 | 2024-08-27 09:45:00 | 10438.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-28 10:35:00 | 10567.80 | 2024-08-28 10:40:00 | 10542.78 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-06 10:30:00 | 10814.20 | 2024-09-06 10:40:00 | 10843.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-10 09:30:00 | 10873.70 | 2024-09-10 09:40:00 | 10849.53 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-09-11 09:30:00 | 11228.15 | 2024-09-11 09:45:00 | 11296.09 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-09-11 09:30:00 | 11228.15 | 2024-09-11 15:20:00 | 11426.45 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2024-09-12 10:10:00 | 11723.40 | 2024-09-12 12:15:00 | 11680.89 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-17 10:50:00 | 11800.00 | 2024-09-17 10:55:00 | 11843.44 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-17 10:50:00 | 11800.00 | 2024-09-17 15:20:00 | 11952.90 | TARGET_HIT | 0.50 | 1.30% |
| BUY | retest1 | 2024-09-19 09:30:00 | 11965.90 | 2024-09-19 09:45:00 | 12028.34 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-19 09:30:00 | 11965.90 | 2024-09-19 09:55:00 | 11965.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 10:10:00 | 11950.20 | 2024-09-20 10:20:00 | 11998.68 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-20 10:10:00 | 11950.20 | 2024-09-20 10:45:00 | 11960.00 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2024-09-23 09:55:00 | 12200.00 | 2024-09-23 10:05:00 | 12155.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-27 10:45:00 | 12658.95 | 2024-09-27 11:25:00 | 12631.77 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-30 10:40:00 | 12383.95 | 2024-09-30 10:55:00 | 12329.03 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-30 10:40:00 | 12383.95 | 2024-09-30 11:45:00 | 12383.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 11:05:00 | 11657.90 | 2024-10-07 11:25:00 | 11696.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-16 10:45:00 | 11615.00 | 2024-10-16 10:50:00 | 11567.56 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-21 09:45:00 | 9900.75 | 2024-10-21 09:55:00 | 9942.01 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-22 11:10:00 | 10550.95 | 2024-10-22 11:50:00 | 10507.81 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-10-24 11:00:00 | 10298.80 | 2024-10-24 12:15:00 | 10238.83 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-24 11:00:00 | 10298.80 | 2024-10-24 12:45:00 | 10298.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 09:30:00 | 9911.00 | 2024-10-29 09:35:00 | 9857.33 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-29 09:30:00 | 9911.00 | 2024-10-29 14:40:00 | 9793.05 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2024-10-31 11:15:00 | 9901.00 | 2024-10-31 11:20:00 | 9928.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-08 09:30:00 | 9935.00 | 2024-11-08 09:45:00 | 9984.60 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-11-08 09:30:00 | 9935.00 | 2024-11-08 09:50:00 | 9935.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 09:40:00 | 9963.60 | 2024-11-11 09:55:00 | 10007.75 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-11-11 09:40:00 | 9963.60 | 2024-11-11 10:10:00 | 9963.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 10:40:00 | 9811.15 | 2024-11-12 12:10:00 | 9772.72 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-11-12 10:40:00 | 9811.15 | 2024-11-12 15:20:00 | 9675.95 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2024-11-13 09:35:00 | 9573.50 | 2024-11-13 10:00:00 | 9523.99 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-11-13 09:35:00 | 9573.50 | 2024-11-13 10:50:00 | 9573.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:45:00 | 9601.55 | 2024-11-19 10:50:00 | 9580.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-11-25 10:50:00 | 9541.05 | 2024-11-25 11:55:00 | 9564.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-27 09:35:00 | 9221.90 | 2024-11-27 10:10:00 | 9195.65 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-28 10:25:00 | 9230.85 | 2024-11-28 10:30:00 | 9206.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-02 09:30:00 | 9156.00 | 2024-12-02 10:25:00 | 9129.51 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-03 10:15:00 | 9202.95 | 2024-12-03 10:25:00 | 9182.74 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-05 09:30:00 | 8919.45 | 2024-12-05 09:45:00 | 8888.90 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-05 09:30:00 | 8919.45 | 2024-12-05 13:10:00 | 8817.00 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-12-16 10:40:00 | 9007.85 | 2024-12-16 11:10:00 | 8983.22 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-12-16 10:40:00 | 9007.85 | 2024-12-16 11:40:00 | 9007.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-18 10:45:00 | 8995.00 | 2024-12-18 10:50:00 | 8972.87 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-12-26 10:55:00 | 8865.80 | 2024-12-26 11:05:00 | 8846.40 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-30 10:35:00 | 8823.20 | 2024-12-30 10:40:00 | 8843.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-08 11:15:00 | 8720.00 | 2025-01-08 11:30:00 | 8734.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-01-10 11:15:00 | 8738.60 | 2025-01-10 11:25:00 | 8700.32 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-10 11:15:00 | 8738.60 | 2025-01-10 11:30:00 | 8738.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-16 11:00:00 | 8553.40 | 2025-01-16 11:15:00 | 8568.18 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-01-17 09:35:00 | 8609.50 | 2025-01-17 10:25:00 | 8642.77 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-01-17 09:35:00 | 8609.50 | 2025-01-17 10:45:00 | 8609.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-20 09:30:00 | 8506.70 | 2025-01-20 09:40:00 | 8478.84 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-20 09:30:00 | 8506.70 | 2025-01-20 10:30:00 | 8506.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 10:40:00 | 8489.35 | 2025-01-21 10:55:00 | 8507.64 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-22 11:10:00 | 8433.00 | 2025-01-22 12:05:00 | 8400.81 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-22 11:10:00 | 8433.00 | 2025-01-22 12:30:00 | 8433.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:30:00 | 8307.00 | 2025-01-27 10:35:00 | 8332.64 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-31 10:45:00 | 8906.00 | 2025-01-31 10:55:00 | 8945.59 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-31 10:45:00 | 8906.00 | 2025-01-31 11:10:00 | 8906.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-12 09:30:00 | 8705.75 | 2025-02-12 09:45:00 | 8730.69 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-14 10:35:00 | 8603.15 | 2025-02-14 10:45:00 | 8624.28 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-18 11:05:00 | 8460.00 | 2025-02-18 11:35:00 | 8438.00 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-02-18 11:05:00 | 8460.00 | 2025-02-18 15:00:00 | 8446.00 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-02-19 09:55:00 | 8487.25 | 2025-02-19 10:35:00 | 8519.07 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-02-19 09:55:00 | 8487.25 | 2025-02-19 10:50:00 | 8487.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-24 09:55:00 | 8364.90 | 2025-02-24 10:35:00 | 8390.06 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-28 10:40:00 | 8109.60 | 2025-02-28 10:50:00 | 8075.18 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-02-28 10:40:00 | 8109.60 | 2025-02-28 15:20:00 | 7890.00 | TARGET_HIT | 0.50 | 2.71% |
| SELL | retest1 | 2025-03-06 10:20:00 | 7405.35 | 2025-03-06 11:05:00 | 7427.58 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-11 10:05:00 | 7440.00 | 2025-03-11 11:20:00 | 7411.82 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-03-12 10:45:00 | 7556.85 | 2025-03-12 11:25:00 | 7535.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-03-13 10:55:00 | 7435.00 | 2025-03-13 11:05:00 | 7454.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-19 10:20:00 | 7715.10 | 2025-03-19 10:45:00 | 7693.67 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-20 09:40:00 | 7820.00 | 2025-03-20 09:45:00 | 7802.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-21 11:15:00 | 8045.05 | 2025-03-21 13:10:00 | 8076.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-03-21 11:15:00 | 8045.05 | 2025-03-21 15:10:00 | 8065.05 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-03-27 10:45:00 | 7956.25 | 2025-03-27 11:40:00 | 7987.26 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-03-27 10:45:00 | 7956.25 | 2025-03-27 14:15:00 | 7956.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 10:50:00 | 8010.15 | 2025-03-28 11:00:00 | 7984.32 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-03 09:30:00 | 7846.35 | 2025-04-03 09:40:00 | 7877.79 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-04-21 09:45:00 | 8076.50 | 2025-04-21 10:05:00 | 8111.34 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-21 09:45:00 | 8076.50 | 2025-04-21 15:20:00 | 8247.50 | TARGET_HIT | 0.50 | 2.12% |
| SELL | retest1 | 2025-04-25 09:40:00 | 8155.50 | 2025-04-25 09:55:00 | 8121.17 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-04-25 09:40:00 | 8155.50 | 2025-04-25 13:55:00 | 8088.50 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-04-29 10:50:00 | 8034.50 | 2025-04-29 11:05:00 | 8058.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-06 09:30:00 | 8042.50 | 2025-05-06 09:35:00 | 8020.67 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-07 10:20:00 | 7894.00 | 2025-05-07 10:30:00 | 7916.79 | STOP_HIT | 1.00 | -0.29% |
