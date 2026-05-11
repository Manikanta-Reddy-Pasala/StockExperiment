# ABB India Ltd. (ABB)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 7010.00
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
| ENTRY1 | 77 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 17 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 116 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 60
- **Target hits / Stop hits / Partials:** 17 / 60 / 39
- **Avg / median % per leg:** 0.26% / 0.00%
- **Sum % (uncompounded):** 30.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 28 | 45.2% | 9 | 34 | 19 | 0.27% | 16.6% |
| BUY @ 2nd Alert (retest1) | 62 | 28 | 45.2% | 9 | 34 | 19 | 0.27% | 16.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 28 | 51.9% | 8 | 26 | 20 | 0.25% | 13.7% |
| SELL @ 2nd Alert (retest1) | 54 | 28 | 51.9% | 8 | 26 | 20 | 0.25% | 13.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 116 | 56 | 48.3% | 17 | 60 | 39 | 0.26% | 30.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:00:00 | 8001.45 | 8041.83 | 0.00 | ORB-short ORB[8030.00,8125.00] vol=2.0x ATR=29.96 |
| Stop hit — per-position SL triggered | 2024-05-15 10:05:00 | 8031.41 | 8040.37 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:45:00 | 8482.55 | 8461.10 | 0.00 | ORB-long ORB[8433.90,8480.00] vol=1.9x ATR=20.10 |
| Stop hit — per-position SL triggered | 2024-05-24 11:45:00 | 8462.45 | 8470.28 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 8270.50 | 8305.29 | 0.00 | ORB-short ORB[8285.00,8374.70] vol=2.2x ATR=26.89 |
| Stop hit — per-position SL triggered | 2024-05-28 10:00:00 | 8297.39 | 8300.62 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 8125.80 | 8159.42 | 0.00 | ORB-short ORB[8141.75,8205.00] vol=1.7x ATR=20.52 |
| Stop hit — per-position SL triggered | 2024-05-30 10:20:00 | 8146.32 | 8158.61 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-03 11:05:00 | 8589.00 | 8483.04 | 0.00 | ORB-long ORB[8396.00,8515.00] vol=2.0x ATR=32.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-03 11:25:00 | 8638.01 | 8506.11 | 0.00 | T1 1.5R @ 8638.01 |
| Target hit | 2024-06-03 15:20:00 | 8740.85 | 8621.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-06-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 10:20:00 | 8407.30 | 8356.31 | 0.00 | ORB-long ORB[8305.05,8397.70] vol=1.6x ATR=20.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:25:00 | 8438.11 | 8373.64 | 0.00 | T1 1.5R @ 8438.11 |
| Stop hit — per-position SL triggered | 2024-06-13 11:20:00 | 8407.30 | 8402.27 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:40:00 | 8680.05 | 8568.42 | 0.00 | ORB-long ORB[8453.05,8574.95] vol=3.0x ATR=38.21 |
| Stop hit — per-position SL triggered | 2024-06-14 09:50:00 | 8641.84 | 8583.67 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:20:00 | 8626.40 | 8795.76 | 0.00 | ORB-short ORB[8804.05,8913.75] vol=1.8x ATR=47.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 11:35:00 | 8554.89 | 8718.48 | 0.00 | T1 1.5R @ 8554.89 |
| Stop hit — per-position SL triggered | 2024-06-19 11:40:00 | 8626.40 | 8713.83 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 8552.00 | 8523.20 | 0.00 | ORB-long ORB[8477.55,8544.95] vol=1.6x ATR=27.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:40:00 | 8592.58 | 8537.96 | 0.00 | T1 1.5R @ 8592.58 |
| Stop hit — per-position SL triggered | 2024-06-25 09:45:00 | 8552.00 | 8540.01 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 8277.70 | 8343.83 | 0.00 | ORB-short ORB[8324.90,8435.00] vol=1.8x ATR=26.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:45:00 | 8237.94 | 8328.37 | 0.00 | T1 1.5R @ 8237.94 |
| Stop hit — per-position SL triggered | 2024-06-26 10:00:00 | 8277.70 | 8314.95 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 8617.00 | 8587.19 | 0.00 | ORB-long ORB[8530.00,8605.00] vol=2.3x ATR=29.13 |
| Stop hit — per-position SL triggered | 2024-07-03 09:55:00 | 8587.87 | 8596.29 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 11:15:00 | 8597.85 | 8649.74 | 0.00 | ORB-short ORB[8651.60,8699.00] vol=2.2x ATR=24.73 |
| Stop hit — per-position SL triggered | 2024-07-04 11:30:00 | 8622.58 | 8645.09 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 8621.35 | 8651.43 | 0.00 | ORB-short ORB[8627.65,8700.00] vol=1.6x ATR=22.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:10:00 | 8587.87 | 8646.32 | 0.00 | T1 1.5R @ 8587.87 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 8621.35 | 8645.41 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 8694.50 | 8636.37 | 0.00 | ORB-long ORB[8570.00,8668.75] vol=2.9x ATR=27.57 |
| Stop hit — per-position SL triggered | 2024-07-09 09:35:00 | 8666.93 | 8651.85 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 8480.15 | 8547.21 | 0.00 | ORB-short ORB[8556.00,8624.95] vol=1.8x ATR=22.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:10:00 | 8446.92 | 8535.95 | 0.00 | T1 1.5R @ 8446.92 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 8480.15 | 8533.45 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:35:00 | 8123.00 | 8191.87 | 0.00 | ORB-short ORB[8150.20,8260.00] vol=1.8x ATR=28.19 |
| Stop hit — per-position SL triggered | 2024-07-15 09:40:00 | 8151.19 | 8186.19 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 7713.80 | 7746.08 | 0.00 | ORB-short ORB[7725.05,7812.05] vol=1.6x ATR=34.19 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 7747.99 | 7745.58 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:45:00 | 7600.00 | 7513.72 | 0.00 | ORB-long ORB[7410.05,7489.95] vol=2.3x ATR=22.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 11:10:00 | 7634.27 | 7534.31 | 0.00 | T1 1.5R @ 7634.27 |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 7600.00 | 7537.61 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 11:10:00 | 7920.00 | 7959.63 | 0.00 | ORB-short ORB[7950.00,8049.00] vol=1.7x ATR=17.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 12:00:00 | 7893.68 | 7953.51 | 0.00 | T1 1.5R @ 7893.68 |
| Stop hit — per-position SL triggered | 2024-07-30 12:25:00 | 7920.00 | 7947.96 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:15:00 | 7926.00 | 7868.11 | 0.00 | ORB-long ORB[7790.00,7901.95] vol=1.7x ATR=18.85 |
| Stop hit — per-position SL triggered | 2024-07-31 11:25:00 | 7907.15 | 7872.55 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:20:00 | 7626.10 | 7544.59 | 0.00 | ORB-long ORB[7478.60,7583.60] vol=1.6x ATR=39.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 10:25:00 | 7685.51 | 7567.67 | 0.00 | T1 1.5R @ 7685.51 |
| Target hit | 2024-08-07 15:20:00 | 7880.00 | 7768.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:45:00 | 7911.60 | 7843.49 | 0.00 | ORB-long ORB[7776.00,7887.05] vol=2.1x ATR=35.98 |
| Stop hit — per-position SL triggered | 2024-08-08 09:55:00 | 7875.62 | 7848.11 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 7850.00 | 7813.81 | 0.00 | ORB-long ORB[7769.15,7841.00] vol=1.7x ATR=26.12 |
| Stop hit — per-position SL triggered | 2024-08-16 09:45:00 | 7823.88 | 7821.59 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:05:00 | 7870.15 | 7911.60 | 0.00 | ORB-short ORB[7907.55,7984.40] vol=1.5x ATR=19.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 12:25:00 | 7841.03 | 7897.91 | 0.00 | T1 1.5R @ 7841.03 |
| Target hit | 2024-08-19 15:20:00 | 7825.00 | 7864.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2024-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:50:00 | 7800.00 | 7823.86 | 0.00 | ORB-short ORB[7826.00,7881.00] vol=2.2x ATR=19.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:05:00 | 7770.26 | 7817.35 | 0.00 | T1 1.5R @ 7770.26 |
| Stop hit — per-position SL triggered | 2024-08-20 12:40:00 | 7800.00 | 7792.47 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:55:00 | 7844.90 | 7824.29 | 0.00 | ORB-long ORB[7765.80,7838.60] vol=1.7x ATR=16.40 |
| Stop hit — per-position SL triggered | 2024-08-21 11:05:00 | 7828.50 | 7824.83 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 7749.00 | 7772.15 | 0.00 | ORB-short ORB[7759.05,7820.00] vol=1.8x ATR=18.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 09:35:00 | 7721.01 | 7762.01 | 0.00 | T1 1.5R @ 7721.01 |
| Stop hit — per-position SL triggered | 2024-08-26 09:45:00 | 7749.00 | 7741.67 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:40:00 | 8010.00 | 7966.64 | 0.00 | ORB-long ORB[7891.20,7990.00] vol=2.1x ATR=22.63 |
| Stop hit — per-position SL triggered | 2024-08-29 09:50:00 | 7987.37 | 7974.06 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:55:00 | 7944.95 | 7885.03 | 0.00 | ORB-long ORB[7852.55,7931.65] vol=1.6x ATR=20.81 |
| Stop hit — per-position SL triggered | 2024-08-30 11:20:00 | 7924.14 | 7892.24 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:25:00 | 7636.55 | 7602.04 | 0.00 | ORB-long ORB[7539.95,7625.50] vol=1.5x ATR=20.83 |
| Stop hit — per-position SL triggered | 2024-09-11 10:50:00 | 7615.72 | 7606.50 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:20:00 | 7701.50 | 7656.32 | 0.00 | ORB-long ORB[7615.05,7689.90] vol=2.4x ATR=21.43 |
| Stop hit — per-position SL triggered | 2024-09-12 10:40:00 | 7680.07 | 7663.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 7805.00 | 7759.37 | 0.00 | ORB-long ORB[7688.00,7774.25] vol=1.8x ATR=24.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 09:45:00 | 7842.47 | 7796.00 | 0.00 | T1 1.5R @ 7842.47 |
| Stop hit — per-position SL triggered | 2024-09-13 10:00:00 | 7805.00 | 7812.34 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 7719.80 | 7770.46 | 0.00 | ORB-short ORB[7767.00,7832.85] vol=1.5x ATR=23.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:55:00 | 7684.77 | 7758.20 | 0.00 | T1 1.5R @ 7684.77 |
| Target hit | 2024-09-19 15:05:00 | 7509.15 | 7493.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2024-09-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:45:00 | 8147.10 | 8125.47 | 0.00 | ORB-long ORB[8065.10,8127.80] vol=1.7x ATR=29.15 |
| Stop hit — per-position SL triggered | 2024-09-27 11:25:00 | 8117.95 | 8129.55 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:45:00 | 8216.70 | 8168.71 | 0.00 | ORB-long ORB[8085.05,8182.00] vol=2.2x ATR=31.01 |
| Stop hit — per-position SL triggered | 2024-10-01 09:55:00 | 8185.69 | 8175.71 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 11:10:00 | 8232.00 | 8194.22 | 0.00 | ORB-long ORB[8130.10,8224.65] vol=3.0x ATR=21.85 |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 8210.15 | 8197.83 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:15:00 | 7992.75 | 7892.10 | 0.00 | ORB-long ORB[7745.05,7827.20] vol=1.8x ATR=32.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 12:50:00 | 8041.61 | 7929.89 | 0.00 | T1 1.5R @ 8041.61 |
| Target hit | 2024-10-08 15:20:00 | 8160.70 | 8032.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-10-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:40:00 | 8530.00 | 8479.63 | 0.00 | ORB-long ORB[8421.35,8510.30] vol=1.8x ATR=26.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:50:00 | 8569.84 | 8490.04 | 0.00 | T1 1.5R @ 8569.84 |
| Stop hit — per-position SL triggered | 2024-10-11 11:00:00 | 8530.00 | 8496.11 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 8637.45 | 8611.23 | 0.00 | ORB-long ORB[8570.00,8631.95] vol=1.8x ATR=26.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:35:00 | 8676.93 | 8626.87 | 0.00 | T1 1.5R @ 8676.93 |
| Stop hit — per-position SL triggered | 2024-10-14 09:50:00 | 8637.45 | 8635.23 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 8590.00 | 8615.19 | 0.00 | ORB-short ORB[8593.25,8656.95] vol=1.5x ATR=25.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:50:00 | 8551.15 | 8600.98 | 0.00 | T1 1.5R @ 8551.15 |
| Stop hit — per-position SL triggered | 2024-10-15 10:35:00 | 8590.00 | 8592.12 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:45:00 | 8759.00 | 8704.96 | 0.00 | ORB-long ORB[8617.50,8725.00] vol=2.0x ATR=27.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:50:00 | 8800.64 | 8719.26 | 0.00 | T1 1.5R @ 8800.64 |
| Stop hit — per-position SL triggered | 2024-10-16 10:40:00 | 8759.00 | 8776.70 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 8650.30 | 8702.08 | 0.00 | ORB-short ORB[8682.20,8790.00] vol=1.7x ATR=36.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:45:00 | 8596.20 | 8683.86 | 0.00 | T1 1.5R @ 8596.20 |
| Target hit | 2024-10-17 10:25:00 | 8585.60 | 8575.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2024-10-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:40:00 | 7480.00 | 7622.13 | 0.00 | ORB-short ORB[7675.00,7737.35] vol=2.0x ATR=38.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:50:00 | 7422.55 | 7598.94 | 0.00 | T1 1.5R @ 7422.55 |
| Stop hit — per-position SL triggered | 2024-10-25 10:55:00 | 7480.00 | 7592.09 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 10:50:00 | 7321.10 | 7364.27 | 0.00 | ORB-short ORB[7405.95,7479.85] vol=1.7x ATR=23.19 |
| Stop hit — per-position SL triggered | 2024-10-31 11:15:00 | 7344.29 | 7358.86 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:45:00 | 7061.50 | 7010.53 | 0.00 | ORB-long ORB[6952.00,7047.95] vol=2.1x ATR=25.81 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 7035.69 | 7012.39 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:55:00 | 7109.25 | 7074.36 | 0.00 | ORB-long ORB[6995.35,7086.20] vol=2.0x ATR=28.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 10:25:00 | 7152.08 | 7095.60 | 0.00 | T1 1.5R @ 7152.08 |
| Target hit | 2024-11-11 15:20:00 | 7227.00 | 7192.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2024-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:35:00 | 7167.55 | 7239.14 | 0.00 | ORB-short ORB[7210.00,7308.75] vol=1.5x ATR=24.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:45:00 | 7131.03 | 7215.20 | 0.00 | T1 1.5R @ 7131.03 |
| Target hit | 2024-11-12 11:45:00 | 7141.50 | 7135.22 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — SELL (started 2024-11-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 11:05:00 | 6626.45 | 6658.34 | 0.00 | ORB-short ORB[6652.25,6745.45] vol=1.6x ATR=25.12 |
| Stop hit — per-position SL triggered | 2024-11-18 11:25:00 | 6651.57 | 6656.50 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:50:00 | 6720.00 | 6683.24 | 0.00 | ORB-long ORB[6615.50,6714.90] vol=2.4x ATR=27.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 11:15:00 | 6760.93 | 6691.17 | 0.00 | T1 1.5R @ 6760.93 |
| Target hit | 2024-11-21 15:20:00 | 6779.90 | 6736.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2024-11-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:45:00 | 7405.10 | 7457.79 | 0.00 | ORB-short ORB[7449.15,7520.00] vol=2.1x ATR=22.66 |
| Stop hit — per-position SL triggered | 2024-11-28 11:00:00 | 7427.76 | 7453.91 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:50:00 | 7435.80 | 7389.84 | 0.00 | ORB-long ORB[7340.05,7418.00] vol=2.0x ATR=24.62 |
| Stop hit — per-position SL triggered | 2024-12-02 11:50:00 | 7411.18 | 7410.83 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:10:00 | 7582.95 | 7519.05 | 0.00 | ORB-long ORB[7480.05,7545.00] vol=2.3x ATR=24.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:15:00 | 7619.43 | 7541.34 | 0.00 | T1 1.5R @ 7619.43 |
| Stop hit — per-position SL triggered | 2024-12-03 10:25:00 | 7582.95 | 7550.23 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:40:00 | 7702.75 | 7633.46 | 0.00 | ORB-long ORB[7575.00,7666.55] vol=3.4x ATR=24.20 |
| Stop hit — per-position SL triggered | 2024-12-04 10:55:00 | 7678.55 | 7649.87 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:35:00 | 7552.15 | 7607.41 | 0.00 | ORB-short ORB[7601.00,7682.00] vol=1.9x ATR=21.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:50:00 | 7519.62 | 7579.70 | 0.00 | T1 1.5R @ 7519.62 |
| Target hit | 2024-12-05 12:20:00 | 7504.25 | 7494.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — BUY (started 2024-12-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:05:00 | 7599.85 | 7581.18 | 0.00 | ORB-long ORB[7532.55,7597.40] vol=1.8x ATR=17.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:15:00 | 7625.46 | 7586.09 | 0.00 | T1 1.5R @ 7625.46 |
| Stop hit — per-position SL triggered | 2024-12-09 11:45:00 | 7599.85 | 7589.43 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 11:05:00 | 7782.55 | 7722.28 | 0.00 | ORB-long ORB[7650.00,7735.00] vol=4.4x ATR=18.41 |
| Stop hit — per-position SL triggered | 2024-12-11 11:10:00 | 7764.14 | 7725.44 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:00:00 | 7678.35 | 7679.06 | 0.00 | ORB-short ORB[7692.00,7755.00] vol=1.6x ATR=17.33 |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 7695.68 | 7679.43 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 7598.90 | 7610.32 | 0.00 | ORB-short ORB[7605.15,7653.90] vol=2.0x ATR=17.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:30:00 | 7572.24 | 7604.88 | 0.00 | T1 1.5R @ 7572.24 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 7598.90 | 7601.84 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:05:00 | 6896.05 | 6951.79 | 0.00 | ORB-short ORB[6942.05,7015.95] vol=1.9x ATR=18.25 |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 6914.30 | 6948.15 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:45:00 | 6907.00 | 6962.15 | 0.00 | ORB-short ORB[6920.05,6985.00] vol=1.7x ATR=19.30 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 6926.30 | 6955.87 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:15:00 | 6940.35 | 6895.37 | 0.00 | ORB-long ORB[6863.15,6929.45] vol=2.0x ATR=16.55 |
| Stop hit — per-position SL triggered | 2025-01-01 11:25:00 | 6923.80 | 6897.96 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 6857.30 | 6880.83 | 0.00 | ORB-short ORB[6868.00,6937.90] vol=2.8x ATR=16.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:50:00 | 6832.91 | 6866.56 | 0.00 | T1 1.5R @ 6832.91 |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 6857.30 | 6856.43 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:15:00 | 6659.10 | 6618.27 | 0.00 | ORB-long ORB[6590.00,6650.50] vol=1.5x ATR=18.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:35:00 | 6686.89 | 6628.84 | 0.00 | T1 1.5R @ 6686.89 |
| Stop hit — per-position SL triggered | 2025-01-09 10:40:00 | 6659.10 | 6632.55 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:40:00 | 6311.05 | 6276.82 | 0.00 | ORB-long ORB[6226.25,6275.00] vol=1.8x ATR=22.72 |
| Stop hit — per-position SL triggered | 2025-01-16 10:50:00 | 6288.33 | 6277.74 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:50:00 | 6358.00 | 6438.45 | 0.00 | ORB-short ORB[6461.80,6542.20] vol=1.7x ATR=22.48 |
| Stop hit — per-position SL triggered | 2025-01-21 11:00:00 | 6380.48 | 6429.28 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:15:00 | 6265.00 | 6292.96 | 0.00 | ORB-short ORB[6276.10,6340.00] vol=2.0x ATR=23.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:30:00 | 6230.49 | 6286.86 | 0.00 | T1 1.5R @ 6230.49 |
| Target hit | 2025-01-22 14:20:00 | 6230.10 | 6215.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 67 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:15:00 | 5624.50 | 5679.42 | 0.00 | ORB-short ORB[5675.00,5731.70] vol=1.8x ATR=25.81 |
| Stop hit — per-position SL triggered | 2025-02-07 10:35:00 | 5650.31 | 5665.07 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 10:30:00 | 5339.25 | 5271.06 | 0.00 | ORB-long ORB[5194.90,5257.45] vol=2.6x ATR=22.30 |
| Stop hit — per-position SL triggered | 2025-02-24 10:35:00 | 5316.95 | 5276.50 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:55:00 | 5207.10 | 5160.34 | 0.00 | ORB-long ORB[5111.15,5174.95] vol=1.7x ATR=18.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:05:00 | 5234.18 | 5171.68 | 0.00 | T1 1.5R @ 5234.18 |
| Target hit | 2025-03-13 10:55:00 | 5213.00 | 5225.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — BUY (started 2025-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:30:00 | 5189.90 | 5143.48 | 0.00 | ORB-long ORB[5101.00,5160.00] vol=2.4x ATR=20.21 |
| Stop hit — per-position SL triggered | 2025-03-17 09:35:00 | 5169.69 | 5146.64 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 5277.00 | 5250.06 | 0.00 | ORB-long ORB[5216.95,5274.60] vol=1.5x ATR=17.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:10:00 | 5302.93 | 5269.39 | 0.00 | T1 1.5R @ 5302.93 |
| Target hit | 2025-03-18 15:20:00 | 5402.00 | 5356.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2025-04-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 10:55:00 | 5436.95 | 5472.13 | 0.00 | ORB-short ORB[5469.50,5548.00] vol=4.1x ATR=20.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:30:00 | 5406.79 | 5460.80 | 0.00 | T1 1.5R @ 5406.79 |
| Target hit | 2025-04-01 15:20:00 | 5392.00 | 5427.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:45:00 | 5358.50 | 5316.17 | 0.00 | ORB-long ORB[5252.50,5310.00] vol=2.3x ATR=16.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 09:50:00 | 5383.79 | 5337.59 | 0.00 | T1 1.5R @ 5383.79 |
| Target hit | 2025-04-16 12:05:00 | 5403.00 | 5404.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — BUY (started 2025-04-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:25:00 | 5494.00 | 5393.45 | 0.00 | ORB-long ORB[5341.50,5422.50] vol=6.2x ATR=30.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:30:00 | 5539.25 | 5431.65 | 0.00 | T1 1.5R @ 5539.25 |
| Target hit | 2025-04-17 15:20:00 | 5564.50 | 5548.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:40:00 | 5784.00 | 5752.40 | 0.00 | ORB-long ORB[5710.50,5775.00] vol=2.6x ATR=20.88 |
| Stop hit — per-position SL triggered | 2025-04-23 09:50:00 | 5763.12 | 5755.04 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 5667.50 | 5690.60 | 0.00 | ORB-short ORB[5672.00,5733.00] vol=2.0x ATR=13.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 09:35:00 | 5646.66 | 5685.17 | 0.00 | T1 1.5R @ 5646.66 |
| Stop hit — per-position SL triggered | 2025-04-24 10:05:00 | 5667.50 | 5671.54 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 5628.00 | 5670.27 | 0.00 | ORB-short ORB[5656.50,5708.50] vol=1.7x ATR=18.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:35:00 | 5600.01 | 5653.64 | 0.00 | T1 1.5R @ 5600.01 |
| Target hit | 2025-04-25 12:30:00 | 5493.00 | 5485.83 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:00:00 | 8001.45 | 2024-05-15 10:05:00 | 8031.41 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-05-24 10:45:00 | 8482.55 | 2024-05-24 11:45:00 | 8462.45 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-28 09:35:00 | 8270.50 | 2024-05-28 10:00:00 | 8297.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-30 10:15:00 | 8125.80 | 2024-05-30 10:20:00 | 8146.32 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-03 11:05:00 | 8589.00 | 2024-06-03 11:25:00 | 8638.01 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-03 11:05:00 | 8589.00 | 2024-06-03 15:20:00 | 8740.85 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2024-06-13 10:20:00 | 8407.30 | 2024-06-13 10:25:00 | 8438.11 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-13 10:20:00 | 8407.30 | 2024-06-13 11:20:00 | 8407.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-14 09:40:00 | 8680.05 | 2024-06-14 09:50:00 | 8641.84 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-19 10:20:00 | 8626.40 | 2024-06-19 11:35:00 | 8554.89 | PARTIAL | 0.50 | 0.83% |
| SELL | retest1 | 2024-06-19 10:20:00 | 8626.40 | 2024-06-19 11:40:00 | 8626.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-25 09:30:00 | 8552.00 | 2024-06-25 09:40:00 | 8592.58 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-25 09:30:00 | 8552.00 | 2024-06-25 09:45:00 | 8552.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 09:40:00 | 8277.70 | 2024-06-26 09:45:00 | 8237.94 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-06-26 09:40:00 | 8277.70 | 2024-06-26 10:00:00 | 8277.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 09:30:00 | 8617.00 | 2024-07-03 09:55:00 | 8587.87 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-04 11:15:00 | 8597.85 | 2024-07-04 11:30:00 | 8622.58 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-08 10:50:00 | 8621.35 | 2024-07-08 11:10:00 | 8587.87 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-08 10:50:00 | 8621.35 | 2024-07-08 11:15:00 | 8621.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 09:30:00 | 8694.50 | 2024-07-09 09:35:00 | 8666.93 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-10 10:05:00 | 8480.15 | 2024-07-10 10:10:00 | 8446.92 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 10:05:00 | 8480.15 | 2024-07-10 10:15:00 | 8480.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 09:35:00 | 8123.00 | 2024-07-15 09:40:00 | 8151.19 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-23 11:15:00 | 7713.80 | 2024-07-23 11:20:00 | 7747.99 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-25 10:45:00 | 7600.00 | 2024-07-25 11:10:00 | 7634.27 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-25 10:45:00 | 7600.00 | 2024-07-25 11:15:00 | 7600.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-30 11:10:00 | 7920.00 | 2024-07-30 12:00:00 | 7893.68 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-07-30 11:10:00 | 7920.00 | 2024-07-30 12:25:00 | 7920.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-31 11:15:00 | 7926.00 | 2024-07-31 11:25:00 | 7907.15 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-07 10:20:00 | 7626.10 | 2024-08-07 10:25:00 | 7685.51 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-08-07 10:20:00 | 7626.10 | 2024-08-07 15:20:00 | 7880.00 | TARGET_HIT | 0.50 | 3.33% |
| BUY | retest1 | 2024-08-08 09:45:00 | 7911.60 | 2024-08-08 09:55:00 | 7875.62 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-16 09:40:00 | 7850.00 | 2024-08-16 09:45:00 | 7823.88 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-19 11:05:00 | 7870.15 | 2024-08-19 12:25:00 | 7841.03 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-19 11:05:00 | 7870.15 | 2024-08-19 15:20:00 | 7825.00 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2024-08-20 09:50:00 | 7800.00 | 2024-08-20 10:05:00 | 7770.26 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-20 09:50:00 | 7800.00 | 2024-08-20 12:40:00 | 7800.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 10:55:00 | 7844.90 | 2024-08-21 11:05:00 | 7828.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-26 09:30:00 | 7749.00 | 2024-08-26 09:35:00 | 7721.01 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-26 09:30:00 | 7749.00 | 2024-08-26 09:45:00 | 7749.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-29 09:40:00 | 8010.00 | 2024-08-29 09:50:00 | 7987.37 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-30 10:55:00 | 7944.95 | 2024-08-30 11:20:00 | 7924.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-11 10:25:00 | 7636.55 | 2024-09-11 10:50:00 | 7615.72 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-12 10:20:00 | 7701.50 | 2024-09-12 10:40:00 | 7680.07 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-13 09:30:00 | 7805.00 | 2024-09-13 09:45:00 | 7842.47 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-13 09:30:00 | 7805.00 | 2024-09-13 10:00:00 | 7805.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:45:00 | 7719.80 | 2024-09-19 09:55:00 | 7684.77 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-19 09:45:00 | 7719.80 | 2024-09-19 15:05:00 | 7509.15 | TARGET_HIT | 0.50 | 2.73% |
| BUY | retest1 | 2024-09-27 10:45:00 | 8147.10 | 2024-09-27 11:25:00 | 8117.95 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-01 09:45:00 | 8216.70 | 2024-10-01 09:55:00 | 8185.69 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-03 11:10:00 | 8232.00 | 2024-10-03 11:15:00 | 8210.15 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-08 11:15:00 | 7992.75 | 2024-10-08 12:50:00 | 8041.61 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-10-08 11:15:00 | 7992.75 | 2024-10-08 15:20:00 | 8160.70 | TARGET_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2024-10-11 10:40:00 | 8530.00 | 2024-10-11 10:50:00 | 8569.84 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-10-11 10:40:00 | 8530.00 | 2024-10-11 11:00:00 | 8530.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 09:30:00 | 8637.45 | 2024-10-14 09:35:00 | 8676.93 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-10-14 09:30:00 | 8637.45 | 2024-10-14 09:50:00 | 8637.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-15 09:30:00 | 8590.00 | 2024-10-15 09:50:00 | 8551.15 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-15 09:30:00 | 8590.00 | 2024-10-15 10:35:00 | 8590.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 09:45:00 | 8759.00 | 2024-10-16 09:50:00 | 8800.64 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-10-16 09:45:00 | 8759.00 | 2024-10-16 10:40:00 | 8759.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:40:00 | 8650.30 | 2024-10-17 09:45:00 | 8596.20 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-17 09:40:00 | 8650.30 | 2024-10-17 10:25:00 | 8585.60 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2024-10-25 10:40:00 | 7480.00 | 2024-10-25 10:50:00 | 7422.55 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-10-25 10:40:00 | 7480.00 | 2024-10-25 10:55:00 | 7480.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-31 10:50:00 | 7321.10 | 2024-10-31 11:15:00 | 7344.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-08 09:45:00 | 7061.50 | 2024-11-08 09:50:00 | 7035.69 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-11-11 09:55:00 | 7109.25 | 2024-11-11 10:25:00 | 7152.08 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-11-11 09:55:00 | 7109.25 | 2024-11-11 15:20:00 | 7227.00 | TARGET_HIT | 0.50 | 1.66% |
| SELL | retest1 | 2024-11-12 09:35:00 | 7167.55 | 2024-11-12 09:45:00 | 7131.03 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-11-12 09:35:00 | 7167.55 | 2024-11-12 11:45:00 | 7141.50 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-11-18 11:05:00 | 6626.45 | 2024-11-18 11:25:00 | 6651.57 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-21 10:50:00 | 6720.00 | 2024-11-21 11:15:00 | 6760.93 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-11-21 10:50:00 | 6720.00 | 2024-11-21 15:20:00 | 6779.90 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2024-11-28 10:45:00 | 7405.10 | 2024-11-28 11:00:00 | 7427.76 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-02 10:50:00 | 7435.80 | 2024-12-02 11:50:00 | 7411.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-03 10:10:00 | 7582.95 | 2024-12-03 10:15:00 | 7619.43 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-12-03 10:10:00 | 7582.95 | 2024-12-03 10:25:00 | 7582.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 10:40:00 | 7702.75 | 2024-12-04 10:55:00 | 7678.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-05 09:35:00 | 7552.15 | 2024-12-05 09:50:00 | 7519.62 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-05 09:35:00 | 7552.15 | 2024-12-05 12:20:00 | 7504.25 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-09 11:05:00 | 7599.85 | 2024-12-09 11:15:00 | 7625.46 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-12-09 11:05:00 | 7599.85 | 2024-12-09 11:45:00 | 7599.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 11:05:00 | 7782.55 | 2024-12-11 11:10:00 | 7764.14 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-12 10:00:00 | 7678.35 | 2024-12-12 10:15:00 | 7695.68 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-13 10:15:00 | 7598.90 | 2024-12-13 10:30:00 | 7572.24 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-12-13 10:15:00 | 7598.90 | 2024-12-13 10:55:00 | 7598.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-24 11:05:00 | 6896.05 | 2024-12-24 11:15:00 | 6914.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-27 10:45:00 | 6907.00 | 2024-12-27 11:10:00 | 6926.30 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-01 11:15:00 | 6940.35 | 2025-01-01 11:25:00 | 6923.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-02 09:30:00 | 6857.30 | 2025-01-02 09:50:00 | 6832.91 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-01-02 09:30:00 | 6857.30 | 2025-01-02 10:15:00 | 6857.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 10:15:00 | 6659.10 | 2025-01-09 10:35:00 | 6686.89 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-09 10:15:00 | 6659.10 | 2025-01-09 10:40:00 | 6659.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-16 10:40:00 | 6311.05 | 2025-01-16 10:50:00 | 6288.33 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-21 10:50:00 | 6358.00 | 2025-01-21 11:00:00 | 6380.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-22 11:15:00 | 6265.00 | 2025-01-22 11:30:00 | 6230.49 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-22 11:15:00 | 6265.00 | 2025-01-22 14:20:00 | 6230.10 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-07 10:15:00 | 5624.50 | 2025-02-07 10:35:00 | 5650.31 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-02-24 10:30:00 | 5339.25 | 2025-02-24 10:35:00 | 5316.95 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-13 09:55:00 | 5207.10 | 2025-03-13 10:05:00 | 5234.18 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-03-13 09:55:00 | 5207.10 | 2025-03-13 10:55:00 | 5213.00 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-03-17 09:30:00 | 5189.90 | 2025-03-17 09:35:00 | 5169.69 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-18 09:35:00 | 5277.00 | 2025-03-18 10:10:00 | 5302.93 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-03-18 09:35:00 | 5277.00 | 2025-03-18 15:20:00 | 5402.00 | TARGET_HIT | 0.50 | 2.37% |
| SELL | retest1 | 2025-04-01 10:55:00 | 5436.95 | 2025-04-01 11:30:00 | 5406.79 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-04-01 10:55:00 | 5436.95 | 2025-04-01 15:20:00 | 5392.00 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2025-04-16 09:45:00 | 5358.50 | 2025-04-16 09:50:00 | 5383.79 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-16 09:45:00 | 5358.50 | 2025-04-16 12:05:00 | 5403.00 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2025-04-17 10:25:00 | 5494.00 | 2025-04-17 10:30:00 | 5539.25 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-04-17 10:25:00 | 5494.00 | 2025-04-17 15:20:00 | 5564.50 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2025-04-23 09:40:00 | 5784.00 | 2025-04-23 09:50:00 | 5763.12 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-04-24 09:30:00 | 5667.50 | 2025-04-24 09:35:00 | 5646.66 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-04-24 09:30:00 | 5667.50 | 2025-04-24 10:05:00 | 5667.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:30:00 | 5628.00 | 2025-04-25 09:35:00 | 5600.01 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-04-25 09:30:00 | 5628.00 | 2025-04-25 12:30:00 | 5493.00 | TARGET_HIT | 0.50 | 2.40% |
