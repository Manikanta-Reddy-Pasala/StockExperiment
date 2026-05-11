# Apar Industries Ltd. (APARINDS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 12760.00
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
| ENTRY1 | 64 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 13 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 51
- **Target hits / Stop hits / Partials:** 13 / 51 / 28
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 13.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 15 | 40.5% | 5 | 22 | 10 | 0.09% | 3.5% |
| BUY @ 2nd Alert (retest1) | 37 | 15 | 40.5% | 5 | 22 | 10 | 0.09% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 26 | 47.3% | 8 | 29 | 18 | 0.18% | 9.9% |
| SELL @ 2nd Alert (retest1) | 55 | 26 | 47.3% | 8 | 29 | 18 | 0.18% | 9.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 92 | 41 | 44.6% | 13 | 51 | 28 | 0.15% | 13.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 09:40:00 | 7565.50 | 7489.07 | 0.00 | ORB-long ORB[7425.00,7525.00] vol=2.0x ATR=32.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 09:50:00 | 7614.40 | 7524.03 | 0.00 | T1 1.5R @ 7614.40 |
| Stop hit — per-position SL triggered | 2025-05-22 10:00:00 | 7565.50 | 7535.70 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:45:00 | 7690.00 | 7780.35 | 0.00 | ORB-short ORB[7725.50,7834.00] vol=1.6x ATR=30.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 11:55:00 | 7643.99 | 7755.24 | 0.00 | T1 1.5R @ 7643.99 |
| Target hit | 2025-05-26 15:20:00 | 7636.00 | 7680.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 7732.00 | 7695.55 | 0.00 | ORB-long ORB[7630.50,7720.00] vol=1.9x ATR=28.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:55:00 | 7775.46 | 7720.43 | 0.00 | T1 1.5R @ 7775.46 |
| Target hit | 2025-05-28 14:15:00 | 7806.00 | 7807.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — BUY (started 2025-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:30:00 | 8117.00 | 8081.92 | 0.00 | ORB-long ORB[8036.00,8108.00] vol=2.2x ATR=19.62 |
| Stop hit — per-position SL triggered | 2025-05-30 09:35:00 | 8097.38 | 8085.72 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:35:00 | 7958.50 | 7984.38 | 0.00 | ORB-short ORB[7968.00,8050.00] vol=1.8x ATR=21.98 |
| Stop hit — per-position SL triggered | 2025-06-05 11:10:00 | 7980.48 | 7978.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:35:00 | 7963.50 | 8039.97 | 0.00 | ORB-short ORB[8040.50,8144.50] vol=3.5x ATR=26.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:05:00 | 7923.74 | 8019.42 | 0.00 | T1 1.5R @ 7923.74 |
| Stop hit — per-position SL triggered | 2025-06-06 13:00:00 | 7963.50 | 7985.58 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:25:00 | 8050.00 | 8105.25 | 0.00 | ORB-short ORB[8107.00,8181.50] vol=1.5x ATR=22.28 |
| Stop hit — per-position SL triggered | 2025-06-11 10:45:00 | 8072.28 | 8102.68 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 7913.00 | 7845.30 | 0.00 | ORB-long ORB[7784.00,7875.00] vol=2.2x ATR=29.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 09:35:00 | 7957.50 | 7885.98 | 0.00 | T1 1.5R @ 7957.50 |
| Target hit | 2025-06-17 12:50:00 | 7952.00 | 7977.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-06-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:55:00 | 8024.00 | 7979.97 | 0.00 | ORB-long ORB[7914.50,8000.00] vol=2.6x ATR=37.19 |
| Stop hit — per-position SL triggered | 2025-06-24 10:10:00 | 7986.81 | 7982.73 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:35:00 | 8044.00 | 7995.89 | 0.00 | ORB-long ORB[7918.00,8015.00] vol=2.4x ATR=29.78 |
| Stop hit — per-position SL triggered | 2025-06-25 09:45:00 | 8014.22 | 8008.71 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:45:00 | 8687.00 | 8759.32 | 0.00 | ORB-short ORB[8775.00,8839.50] vol=2.0x ATR=30.20 |
| Stop hit — per-position SL triggered | 2025-07-02 10:25:00 | 8717.20 | 8725.91 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:00:00 | 8497.50 | 8514.58 | 0.00 | ORB-short ORB[8504.50,8596.50] vol=3.2x ATR=29.00 |
| Stop hit — per-position SL triggered | 2025-07-08 10:05:00 | 8526.50 | 8516.84 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:30:00 | 8731.50 | 8867.26 | 0.00 | ORB-short ORB[8859.50,8964.00] vol=1.7x ATR=28.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 8688.19 | 8834.89 | 0.00 | T1 1.5R @ 8688.19 |
| Target hit | 2025-07-11 15:20:00 | 8702.00 | 8705.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 8885.00 | 8955.38 | 0.00 | ORB-short ORB[8940.00,9069.00] vol=2.5x ATR=30.98 |
| Stop hit — per-position SL triggered | 2025-07-17 09:45:00 | 8915.98 | 8935.08 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 8976.00 | 8928.24 | 0.00 | ORB-long ORB[8831.00,8960.00] vol=2.4x ATR=32.74 |
| Stop hit — per-position SL triggered | 2025-07-18 10:05:00 | 8943.26 | 8939.26 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:30:00 | 9012.00 | 8952.48 | 0.00 | ORB-long ORB[8899.00,8979.50] vol=1.9x ATR=40.46 |
| Stop hit — per-position SL triggered | 2025-07-21 09:35:00 | 8971.54 | 8962.75 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:20:00 | 8970.00 | 9060.07 | 0.00 | ORB-short ORB[9035.50,9148.00] vol=2.7x ATR=31.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:30:00 | 8923.17 | 9022.22 | 0.00 | T1 1.5R @ 8923.17 |
| Stop hit — per-position SL triggered | 2025-07-25 11:55:00 | 8970.00 | 8975.72 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 8895.00 | 8934.16 | 0.00 | ORB-short ORB[8906.00,9027.00] vol=2.0x ATR=35.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:45:00 | 8841.85 | 8916.98 | 0.00 | T1 1.5R @ 8841.85 |
| Stop hit — per-position SL triggered | 2025-08-06 09:50:00 | 8895.00 | 8910.49 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 8754.00 | 8795.09 | 0.00 | ORB-short ORB[8756.00,8858.00] vol=1.5x ATR=19.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:15:00 | 8724.59 | 8789.53 | 0.00 | T1 1.5R @ 8724.59 |
| Target hit | 2025-08-07 14:50:00 | 8699.00 | 8640.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2025-08-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:55:00 | 8808.50 | 8714.56 | 0.00 | ORB-long ORB[8630.00,8743.00] vol=1.5x ATR=42.72 |
| Stop hit — per-position SL triggered | 2025-08-11 10:05:00 | 8765.78 | 8727.89 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:55:00 | 8440.00 | 8488.19 | 0.00 | ORB-short ORB[8468.50,8531.00] vol=2.4x ATR=22.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:25:00 | 8406.48 | 8473.46 | 0.00 | T1 1.5R @ 8406.48 |
| Stop hit — per-position SL triggered | 2025-08-19 11:35:00 | 8440.00 | 8466.68 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 8381.50 | 8440.86 | 0.00 | ORB-short ORB[8401.00,8511.50] vol=1.8x ATR=28.58 |
| Stop hit — per-position SL triggered | 2025-08-20 09:35:00 | 8410.08 | 8437.92 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:45:00 | 8454.00 | 8402.11 | 0.00 | ORB-long ORB[8340.50,8429.50] vol=1.5x ATR=23.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 09:55:00 | 8489.94 | 8418.70 | 0.00 | T1 1.5R @ 8489.94 |
| Stop hit — per-position SL triggered | 2025-08-21 11:25:00 | 8454.00 | 8463.57 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:15:00 | 8106.00 | 8157.01 | 0.00 | ORB-short ORB[8170.00,8268.50] vol=1.6x ATR=27.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 12:40:00 | 8064.61 | 8125.61 | 0.00 | T1 1.5R @ 8064.61 |
| Stop hit — per-position SL triggered | 2025-08-25 13:45:00 | 8106.00 | 8122.49 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:50:00 | 7985.00 | 8004.95 | 0.00 | ORB-short ORB[7992.00,8099.00] vol=2.3x ATR=19.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 11:00:00 | 7955.67 | 8000.85 | 0.00 | T1 1.5R @ 7955.67 |
| Target hit | 2025-08-26 15:20:00 | 7863.00 | 7945.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:35:00 | 7715.00 | 7737.93 | 0.00 | ORB-short ORB[7720.50,7778.00] vol=2.1x ATR=25.00 |
| Stop hit — per-position SL triggered | 2025-08-29 09:40:00 | 7740.00 | 7738.19 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:35:00 | 7730.50 | 7782.97 | 0.00 | ORB-short ORB[7752.00,7825.00] vol=2.9x ATR=20.97 |
| Stop hit — per-position SL triggered | 2025-09-08 11:30:00 | 7751.47 | 7771.67 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:00:00 | 7896.50 | 7863.09 | 0.00 | ORB-long ORB[7793.50,7884.00] vol=4.0x ATR=29.39 |
| Stop hit — per-position SL triggered | 2025-09-09 10:40:00 | 7867.11 | 7873.85 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:50:00 | 8802.00 | 8760.02 | 0.00 | ORB-long ORB[8700.00,8795.00] vol=2.8x ATR=18.62 |
| Stop hit — per-position SL triggered | 2025-09-25 11:00:00 | 8783.38 | 8762.87 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:35:00 | 8604.00 | 8634.22 | 0.00 | ORB-short ORB[8612.00,8692.50] vol=1.5x ATR=31.10 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 8635.10 | 8635.12 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 09:40:00 | 8450.50 | 8430.35 | 0.00 | ORB-long ORB[8380.00,8437.00] vol=2.8x ATR=28.73 |
| Stop hit — per-position SL triggered | 2025-10-06 09:45:00 | 8421.77 | 8430.05 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:10:00 | 8525.50 | 8482.67 | 0.00 | ORB-long ORB[8424.50,8490.00] vol=2.2x ATR=28.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:30:00 | 8568.03 | 8504.00 | 0.00 | T1 1.5R @ 8568.03 |
| Stop hit — per-position SL triggered | 2025-10-07 10:45:00 | 8525.50 | 8506.25 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:35:00 | 8384.50 | 8439.59 | 0.00 | ORB-short ORB[8428.00,8530.00] vol=2.9x ATR=29.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:25:00 | 8340.79 | 8407.43 | 0.00 | T1 1.5R @ 8340.79 |
| Stop hit — per-position SL triggered | 2025-10-09 14:20:00 | 8384.50 | 8398.59 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 11:10:00 | 8355.00 | 8414.31 | 0.00 | ORB-short ORB[8409.50,8505.00] vol=3.3x ATR=25.92 |
| Stop hit — per-position SL triggered | 2025-10-15 12:00:00 | 8380.92 | 8355.48 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:45:00 | 8540.50 | 8452.68 | 0.00 | ORB-long ORB[8335.00,8424.50] vol=3.8x ATR=29.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:50:00 | 8585.27 | 8494.31 | 0.00 | T1 1.5R @ 8585.27 |
| Target hit | 2025-10-16 10:15:00 | 8621.00 | 8646.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — SELL (started 2025-10-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:30:00 | 8611.00 | 8647.66 | 0.00 | ORB-short ORB[8636.00,8699.50] vol=1.6x ATR=20.65 |
| Stop hit — per-position SL triggered | 2025-10-20 10:35:00 | 8631.65 | 8643.56 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:25:00 | 8746.00 | 8654.16 | 0.00 | ORB-long ORB[8615.00,8737.50] vol=3.3x ATR=37.65 |
| Stop hit — per-position SL triggered | 2025-10-23 10:30:00 | 8708.35 | 8659.93 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:15:00 | 8763.00 | 8711.75 | 0.00 | ORB-long ORB[8659.00,8735.00] vol=2.0x ATR=27.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 10:20:00 | 8803.78 | 8736.27 | 0.00 | T1 1.5R @ 8803.78 |
| Stop hit — per-position SL triggered | 2025-10-24 10:40:00 | 8763.00 | 8756.82 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:45:00 | 8817.00 | 8788.72 | 0.00 | ORB-long ORB[8725.00,8806.00] vol=2.9x ATR=27.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 09:55:00 | 8858.42 | 8805.30 | 0.00 | T1 1.5R @ 8858.42 |
| Stop hit — per-position SL triggered | 2025-10-27 10:00:00 | 8817.00 | 8806.43 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:25:00 | 8415.00 | 8467.99 | 0.00 | ORB-short ORB[8460.00,8550.00] vol=1.7x ATR=24.22 |
| Stop hit — per-position SL triggered | 2025-11-04 11:05:00 | 8439.22 | 8454.55 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 11:00:00 | 8830.00 | 8938.66 | 0.00 | ORB-short ORB[8921.00,9038.50] vol=4.2x ATR=34.31 |
| Stop hit — per-position SL triggered | 2025-11-13 13:05:00 | 8864.31 | 8899.77 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:30:00 | 9015.00 | 8904.61 | 0.00 | ORB-long ORB[8837.50,8934.00] vol=1.6x ATR=36.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 11:00:00 | 9070.06 | 8939.52 | 0.00 | T1 1.5R @ 9070.06 |
| Target hit | 2025-11-14 15:20:00 | 9079.00 | 9004.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-11-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 09:40:00 | 9315.50 | 9208.58 | 0.00 | ORB-long ORB[9086.00,9220.50] vol=3.3x ATR=48.23 |
| Stop hit — per-position SL triggered | 2025-11-18 10:00:00 | 9267.27 | 9248.16 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:15:00 | 9073.50 | 9128.18 | 0.00 | ORB-short ORB[9084.00,9198.50] vol=1.8x ATR=25.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:20:00 | 9035.14 | 9123.05 | 0.00 | T1 1.5R @ 9035.14 |
| Stop hit — per-position SL triggered | 2025-11-21 11:05:00 | 9073.50 | 9082.96 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:00:00 | 9148.00 | 9053.98 | 0.00 | ORB-long ORB[8908.50,9038.00] vol=2.1x ATR=31.99 |
| Stop hit — per-position SL triggered | 2025-11-26 10:10:00 | 9116.01 | 9067.80 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-27 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:25:00 | 9206.00 | 9260.27 | 0.00 | ORB-short ORB[9215.50,9333.50] vol=1.6x ATR=29.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:55:00 | 9161.42 | 9254.46 | 0.00 | T1 1.5R @ 9161.42 |
| Stop hit — per-position SL triggered | 2025-11-27 12:00:00 | 9206.00 | 9234.46 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:45:00 | 9085.50 | 9156.79 | 0.00 | ORB-short ORB[9147.50,9232.50] vol=1.5x ATR=26.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:05:00 | 9045.97 | 9136.75 | 0.00 | T1 1.5R @ 9045.97 |
| Target hit | 2025-12-01 15:20:00 | 8973.50 | 9038.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 8887.00 | 8962.72 | 0.00 | ORB-short ORB[8920.50,9054.00] vol=1.6x ATR=32.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:05:00 | 8837.91 | 8915.47 | 0.00 | T1 1.5R @ 8837.91 |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 8887.00 | 8908.99 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:25:00 | 8787.50 | 8852.19 | 0.00 | ORB-short ORB[8845.00,8940.00] vol=1.5x ATR=27.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:05:00 | 8746.54 | 8813.28 | 0.00 | T1 1.5R @ 8746.54 |
| Target hit | 2025-12-03 15:20:00 | 8700.00 | 8735.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-12-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:10:00 | 8755.00 | 8822.74 | 0.00 | ORB-short ORB[8810.00,8895.50] vol=3.3x ATR=21.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:00:00 | 8722.24 | 8791.73 | 0.00 | T1 1.5R @ 8722.24 |
| Target hit | 2025-12-08 15:20:00 | 8674.00 | 8699.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:10:00 | 9077.00 | 9018.27 | 0.00 | ORB-long ORB[8930.00,9038.50] vol=1.6x ATR=35.98 |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 9041.02 | 9022.12 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 8777.00 | 8832.19 | 0.00 | ORB-short ORB[8860.00,8925.00] vol=1.6x ATR=17.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:35:00 | 8750.82 | 8815.49 | 0.00 | T1 1.5R @ 8750.82 |
| Target hit | 2025-12-24 15:20:00 | 8676.00 | 8740.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 10:15:00 | 8603.00 | 8634.59 | 0.00 | ORB-short ORB[8619.00,8688.00] vol=1.9x ATR=22.30 |
| Stop hit — per-position SL triggered | 2025-12-26 10:20:00 | 8625.30 | 8634.29 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:45:00 | 8455.00 | 8491.83 | 0.00 | ORB-short ORB[8460.50,8560.00] vol=1.5x ATR=19.86 |
| Stop hit — per-position SL triggered | 2025-12-30 09:55:00 | 8474.86 | 8489.33 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 8438.50 | 8400.67 | 0.00 | ORB-long ORB[8368.00,8430.00] vol=3.4x ATR=18.24 |
| Stop hit — per-position SL triggered | 2026-01-01 11:30:00 | 8420.26 | 8402.38 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 8337.00 | 8356.91 | 0.00 | ORB-short ORB[8340.00,8383.50] vol=2.6x ATR=16.31 |
| Stop hit — per-position SL triggered | 2026-01-02 09:40:00 | 8353.31 | 8355.61 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:00:00 | 8201.00 | 8237.09 | 0.00 | ORB-short ORB[8205.50,8293.00] vol=5.9x ATR=19.17 |
| Stop hit — per-position SL triggered | 2026-01-05 11:10:00 | 8220.17 | 8227.73 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 8348.50 | 8271.10 | 0.00 | ORB-long ORB[8174.00,8275.00] vol=3.3x ATR=23.07 |
| Stop hit — per-position SL triggered | 2026-01-08 09:35:00 | 8325.43 | 8286.91 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:30:00 | 7523.50 | 7615.10 | 0.00 | ORB-short ORB[7653.00,7748.00] vol=2.7x ATR=30.19 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 7553.69 | 7593.13 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:30:00 | 7107.50 | 7038.95 | 0.00 | ORB-long ORB[6957.50,7060.50] vol=1.8x ATR=35.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:35:00 | 7160.59 | 7069.30 | 0.00 | T1 1.5R @ 7160.59 |
| Target hit | 2026-01-27 15:00:00 | 7187.00 | 7191.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2026-01-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:05:00 | 7105.00 | 7185.66 | 0.00 | ORB-short ORB[7221.00,7300.00] vol=3.1x ATR=22.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:15:00 | 7071.47 | 7174.36 | 0.00 | T1 1.5R @ 7071.47 |
| Stop hit — per-position SL triggered | 2026-01-28 11:40:00 | 7105.00 | 7153.35 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 8012.00 | 7983.68 | 0.00 | ORB-long ORB[7919.00,8001.00] vol=2.2x ATR=26.11 |
| Stop hit — per-position SL triggered | 2026-02-01 12:00:00 | 7985.89 | 7988.11 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 9541.50 | 9486.29 | 0.00 | ORB-long ORB[9401.50,9539.50] vol=2.4x ATR=39.68 |
| Stop hit — per-position SL triggered | 2026-02-09 11:45:00 | 9501.82 | 9511.28 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:15:00 | 11215.00 | 11313.78 | 0.00 | ORB-short ORB[11274.50,11431.00] vol=2.3x ATR=52.02 |
| Stop hit — per-position SL triggered | 2026-04-15 10:20:00 | 11267.02 | 11309.83 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-22 09:40:00 | 7565.50 | 2025-05-22 09:50:00 | 7614.40 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-05-22 09:40:00 | 7565.50 | 2025-05-22 10:00:00 | 7565.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-26 10:45:00 | 7690.00 | 2025-05-26 11:55:00 | 7643.99 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-05-26 10:45:00 | 7690.00 | 2025-05-26 15:20:00 | 7636.00 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2025-05-28 09:30:00 | 7732.00 | 2025-05-28 09:55:00 | 7775.46 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-05-28 09:30:00 | 7732.00 | 2025-05-28 14:15:00 | 7806.00 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-05-30 09:30:00 | 8117.00 | 2025-05-30 09:35:00 | 8097.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-05 10:35:00 | 7958.50 | 2025-06-05 11:10:00 | 7980.48 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-06 10:35:00 | 7963.50 | 2025-06-06 11:05:00 | 7923.74 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-06-06 10:35:00 | 7963.50 | 2025-06-06 13:00:00 | 7963.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-11 10:25:00 | 8050.00 | 2025-06-11 10:45:00 | 8072.28 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-17 09:30:00 | 7913.00 | 2025-06-17 09:35:00 | 7957.50 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-17 09:30:00 | 7913.00 | 2025-06-17 12:50:00 | 7952.00 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-24 09:55:00 | 8024.00 | 2025-06-24 10:10:00 | 7986.81 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-25 09:35:00 | 8044.00 | 2025-06-25 09:45:00 | 8014.22 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-02 09:45:00 | 8687.00 | 2025-07-02 10:25:00 | 8717.20 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-08 10:00:00 | 8497.50 | 2025-07-08 10:05:00 | 8526.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-11 10:30:00 | 8731.50 | 2025-07-11 11:00:00 | 8688.19 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-11 10:30:00 | 8731.50 | 2025-07-11 15:20:00 | 8702.00 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-17 09:30:00 | 8885.00 | 2025-07-17 09:45:00 | 8915.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-18 09:45:00 | 8976.00 | 2025-07-18 10:05:00 | 8943.26 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-21 09:30:00 | 9012.00 | 2025-07-21 09:35:00 | 8971.54 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-25 10:20:00 | 8970.00 | 2025-07-25 10:30:00 | 8923.17 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-07-25 10:20:00 | 8970.00 | 2025-07-25 11:55:00 | 8970.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 09:30:00 | 8895.00 | 2025-08-06 09:45:00 | 8841.85 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-08-06 09:30:00 | 8895.00 | 2025-08-06 09:50:00 | 8895.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:00:00 | 8754.00 | 2025-08-07 11:15:00 | 8724.59 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-07 11:00:00 | 8754.00 | 2025-08-07 14:50:00 | 8699.00 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-08-11 09:55:00 | 8808.50 | 2025-08-11 10:05:00 | 8765.78 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-08-19 10:55:00 | 8440.00 | 2025-08-19 11:25:00 | 8406.48 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-19 10:55:00 | 8440.00 | 2025-08-19 11:35:00 | 8440.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-20 09:30:00 | 8381.50 | 2025-08-20 09:35:00 | 8410.08 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-21 09:45:00 | 8454.00 | 2025-08-21 09:55:00 | 8489.94 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-21 09:45:00 | 8454.00 | 2025-08-21 11:25:00 | 8454.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-25 10:15:00 | 8106.00 | 2025-08-25 12:40:00 | 8064.61 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-08-25 10:15:00 | 8106.00 | 2025-08-25 13:45:00 | 8106.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 10:50:00 | 7985.00 | 2025-08-26 11:00:00 | 7955.67 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-26 10:50:00 | 7985.00 | 2025-08-26 15:20:00 | 7863.00 | TARGET_HIT | 0.50 | 1.53% |
| SELL | retest1 | 2025-08-29 09:35:00 | 7715.00 | 2025-08-29 09:40:00 | 7740.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-08 10:35:00 | 7730.50 | 2025-09-08 11:30:00 | 7751.47 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-09 10:00:00 | 7896.50 | 2025-09-09 10:40:00 | 7867.11 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-25 10:50:00 | 8802.00 | 2025-09-25 11:00:00 | 8783.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-26 09:35:00 | 8604.00 | 2025-09-26 09:40:00 | 8635.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-06 09:40:00 | 8450.50 | 2025-10-06 09:45:00 | 8421.77 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-07 10:10:00 | 8525.50 | 2025-10-07 10:30:00 | 8568.03 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-07 10:10:00 | 8525.50 | 2025-10-07 10:45:00 | 8525.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-09 10:35:00 | 8384.50 | 2025-10-09 12:25:00 | 8340.79 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-10-09 10:35:00 | 8384.50 | 2025-10-09 14:20:00 | 8384.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-15 11:10:00 | 8355.00 | 2025-10-15 12:00:00 | 8380.92 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-16 09:45:00 | 8540.50 | 2025-10-16 09:50:00 | 8585.27 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-16 09:45:00 | 8540.50 | 2025-10-16 10:15:00 | 8621.00 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-10-20 10:30:00 | 8611.00 | 2025-10-20 10:35:00 | 8631.65 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-23 10:25:00 | 8746.00 | 2025-10-23 10:30:00 | 8708.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-10-24 10:15:00 | 8763.00 | 2025-10-24 10:20:00 | 8803.78 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-24 10:15:00 | 8763.00 | 2025-10-24 10:40:00 | 8763.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:45:00 | 8817.00 | 2025-10-27 09:55:00 | 8858.42 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-10-27 09:45:00 | 8817.00 | 2025-10-27 10:00:00 | 8817.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:25:00 | 8415.00 | 2025-11-04 11:05:00 | 8439.22 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-11-13 11:00:00 | 8830.00 | 2025-11-13 13:05:00 | 8864.31 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-11-14 10:30:00 | 9015.00 | 2025-11-14 11:00:00 | 9070.06 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-11-14 10:30:00 | 9015.00 | 2025-11-14 15:20:00 | 9079.00 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2025-11-18 09:40:00 | 9315.50 | 2025-11-18 10:00:00 | 9267.27 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-11-21 10:15:00 | 9073.50 | 2025-11-21 10:20:00 | 9035.14 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-11-21 10:15:00 | 9073.50 | 2025-11-21 11:05:00 | 9073.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 10:00:00 | 9148.00 | 2025-11-26 10:10:00 | 9116.01 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-27 10:25:00 | 9206.00 | 2025-11-27 10:55:00 | 9161.42 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-11-27 10:25:00 | 9206.00 | 2025-11-27 12:00:00 | 9206.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:45:00 | 9085.50 | 2025-12-01 11:05:00 | 9045.97 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-01 10:45:00 | 9085.50 | 2025-12-01 15:20:00 | 8973.50 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2025-12-02 09:30:00 | 8887.00 | 2025-12-02 10:05:00 | 8837.91 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-12-02 09:30:00 | 8887.00 | 2025-12-02 10:15:00 | 8887.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:25:00 | 8787.50 | 2025-12-03 11:05:00 | 8746.54 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-03 10:25:00 | 8787.50 | 2025-12-03 15:20:00 | 8700.00 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2025-12-08 11:10:00 | 8755.00 | 2025-12-08 12:00:00 | 8722.24 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-08 11:10:00 | 8755.00 | 2025-12-08 15:20:00 | 8674.00 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2025-12-12 10:10:00 | 9077.00 | 2025-12-12 10:15:00 | 9041.02 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-12-24 10:55:00 | 8777.00 | 2025-12-24 11:35:00 | 8750.82 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-24 10:55:00 | 8777.00 | 2025-12-24 15:20:00 | 8676.00 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2025-12-26 10:15:00 | 8603.00 | 2025-12-26 10:20:00 | 8625.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-30 09:45:00 | 8455.00 | 2025-12-30 09:55:00 | 8474.86 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-01 11:15:00 | 8438.50 | 2026-01-01 11:30:00 | 8420.26 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-02 09:30:00 | 8337.00 | 2026-01-02 09:40:00 | 8353.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-05 11:00:00 | 8201.00 | 2026-01-05 11:10:00 | 8220.17 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-08 09:30:00 | 8348.50 | 2026-01-08 09:35:00 | 8325.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-14 10:30:00 | 7523.50 | 2026-01-14 11:15:00 | 7553.69 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-27 09:30:00 | 7107.50 | 2026-01-27 09:35:00 | 7160.59 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-01-27 09:30:00 | 7107.50 | 2026-01-27 15:00:00 | 7187.00 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2026-01-28 11:05:00 | 7105.00 | 2026-01-28 11:15:00 | 7071.47 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-01-28 11:05:00 | 7105.00 | 2026-01-28 11:40:00 | 7105.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:10:00 | 8012.00 | 2026-02-01 12:00:00 | 7985.89 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-09 10:25:00 | 9541.50 | 2026-02-09 11:45:00 | 9501.82 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-15 10:15:00 | 11215.00 | 2026-04-15 10:20:00 | 11267.02 | STOP_HIT | 1.00 | -0.46% |
