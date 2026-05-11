# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
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
| ENTRY1 | 95 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 15 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 80
- **Target hits / Stop hits / Partials:** 15 / 80 / 37
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 18.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 37 | 48.1% | 10 | 40 | 27 | 0.25% | 19.4% |
| BUY @ 2nd Alert (retest1) | 77 | 37 | 48.1% | 10 | 40 | 27 | 0.25% | 19.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 55 | 15 | 27.3% | 5 | 40 | 10 | -0.02% | -1.1% |
| SELL @ 2nd Alert (retest1) | 55 | 15 | 27.3% | 5 | 40 | 10 | -0.02% | -1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 132 | 52 | 39.4% | 15 | 80 | 37 | 0.14% | 18.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:35:00 | 8172.00 | 8120.13 | 0.00 | ORB-long ORB[8032.00,8139.00] vol=1.8x ATR=27.00 |
| Stop hit — per-position SL triggered | 2025-05-14 10:10:00 | 8145.00 | 8147.66 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 11:00:00 | 8395.00 | 8336.96 | 0.00 | ORB-long ORB[8290.00,8367.00] vol=3.3x ATR=22.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 11:30:00 | 8428.13 | 8351.09 | 0.00 | T1 1.5R @ 8428.13 |
| Stop hit — per-position SL triggered | 2025-05-16 11:50:00 | 8395.00 | 8358.62 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:45:00 | 8675.50 | 8617.08 | 0.00 | ORB-long ORB[8535.50,8634.50] vol=3.0x ATR=33.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:05:00 | 8725.34 | 8642.28 | 0.00 | T1 1.5R @ 8725.34 |
| Stop hit — per-position SL triggered | 2025-05-21 11:35:00 | 8675.50 | 8685.85 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 8915.50 | 8856.62 | 0.00 | ORB-long ORB[8770.00,8885.00] vol=3.1x ATR=26.41 |
| Stop hit — per-position SL triggered | 2025-05-26 09:35:00 | 8889.09 | 8862.58 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 10:45:00 | 8559.50 | 8587.27 | 0.00 | ORB-short ORB[8563.50,8614.00] vol=1.7x ATR=12.61 |
| Stop hit — per-position SL triggered | 2025-06-05 11:25:00 | 8572.11 | 8581.74 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:35:00 | 8681.00 | 8602.82 | 0.00 | ORB-long ORB[8560.00,8608.00] vol=2.8x ATR=25.52 |
| Stop hit — per-position SL triggered | 2025-06-06 10:40:00 | 8655.48 | 8606.47 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 10:10:00 | 8623.00 | 8648.05 | 0.00 | ORB-short ORB[8631.00,8705.00] vol=5.0x ATR=18.81 |
| Stop hit — per-position SL triggered | 2025-06-09 10:45:00 | 8641.81 | 8641.18 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 09:50:00 | 8602.50 | 8627.83 | 0.00 | ORB-short ORB[8620.00,8664.00] vol=2.3x ATR=16.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 10:50:00 | 8577.66 | 8609.84 | 0.00 | T1 1.5R @ 8577.66 |
| Stop hit — per-position SL triggered | 2025-06-10 11:05:00 | 8602.50 | 8608.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:00:00 | 8738.00 | 8704.66 | 0.00 | ORB-long ORB[8610.00,8714.00] vol=2.1x ATR=19.17 |
| Stop hit — per-position SL triggered | 2025-06-11 10:25:00 | 8718.83 | 8715.08 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 11:00:00 | 8561.00 | 8499.43 | 0.00 | ORB-long ORB[8474.50,8515.50] vol=1.6x ATR=17.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 11:15:00 | 8586.74 | 8508.34 | 0.00 | T1 1.5R @ 8586.74 |
| Stop hit — per-position SL triggered | 2025-06-16 11:20:00 | 8561.00 | 8512.62 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:40:00 | 8467.50 | 8490.94 | 0.00 | ORB-short ORB[8474.00,8529.50] vol=1.5x ATR=16.65 |
| Stop hit — per-position SL triggered | 2025-06-17 10:35:00 | 8484.15 | 8479.85 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 8598.00 | 8542.37 | 0.00 | ORB-long ORB[8463.00,8572.00] vol=2.6x ATR=20.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:35:00 | 8628.03 | 8568.40 | 0.00 | T1 1.5R @ 8628.03 |
| Stop hit — per-position SL triggered | 2025-06-18 10:25:00 | 8598.00 | 8599.11 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 11:00:00 | 8548.50 | 8498.98 | 0.00 | ORB-long ORB[8451.50,8512.00] vol=1.8x ATR=16.37 |
| Stop hit — per-position SL triggered | 2025-06-19 11:10:00 | 8532.13 | 8501.02 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 8313.00 | 8348.96 | 0.00 | ORB-short ORB[8316.00,8384.00] vol=4.2x ATR=15.94 |
| Stop hit — per-position SL triggered | 2025-07-01 11:05:00 | 8328.94 | 8345.60 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:45:00 | 8442.50 | 8403.83 | 0.00 | ORB-long ORB[8364.50,8409.50] vol=1.6x ATR=13.21 |
| Stop hit — per-position SL triggered | 2025-07-04 10:55:00 | 8429.29 | 8408.04 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 11:00:00 | 8470.50 | 8450.69 | 0.00 | ORB-long ORB[8404.00,8454.00] vol=3.5x ATR=13.29 |
| Stop hit — per-position SL triggered | 2025-07-07 11:20:00 | 8457.21 | 8452.53 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:15:00 | 8360.50 | 8403.68 | 0.00 | ORB-short ORB[8425.00,8474.00] vol=2.1x ATR=14.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:20:00 | 8338.35 | 8395.72 | 0.00 | T1 1.5R @ 8338.35 |
| Target hit | 2025-07-08 15:00:00 | 8335.00 | 8334.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2025-07-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:05:00 | 8373.00 | 8354.11 | 0.00 | ORB-long ORB[8310.50,8365.00] vol=2.6x ATR=11.11 |
| Stop hit — per-position SL triggered | 2025-07-09 11:20:00 | 8361.89 | 8354.72 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:30:00 | 8353.00 | 8363.98 | 0.00 | ORB-short ORB[8353.50,8394.00] vol=1.9x ATR=11.04 |
| Stop hit — per-position SL triggered | 2025-07-10 10:05:00 | 8364.04 | 8359.56 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 8215.50 | 8239.56 | 0.00 | ORB-short ORB[8216.00,8300.00] vol=1.5x ATR=16.79 |
| Stop hit — per-position SL triggered | 2025-07-11 09:35:00 | 8232.29 | 8240.41 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:35:00 | 8161.00 | 8144.37 | 0.00 | ORB-long ORB[8090.00,8160.00] vol=3.6x ATR=15.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 11:00:00 | 8183.87 | 8152.24 | 0.00 | T1 1.5R @ 8183.87 |
| Target hit | 2025-07-15 15:20:00 | 8317.50 | 8231.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 8204.00 | 8234.96 | 0.00 | ORB-short ORB[8211.50,8311.50] vol=1.7x ATR=18.28 |
| Stop hit — per-position SL triggered | 2025-07-16 10:20:00 | 8222.28 | 8225.59 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:10:00 | 8326.00 | 8411.99 | 0.00 | ORB-short ORB[8423.50,8470.00] vol=1.9x ATR=18.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:30:00 | 8297.64 | 8396.24 | 0.00 | T1 1.5R @ 8297.64 |
| Stop hit — per-position SL triggered | 2025-07-22 12:25:00 | 8326.00 | 8378.90 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:10:00 | 8355.00 | 8376.30 | 0.00 | ORB-short ORB[8375.00,8435.00] vol=5.1x ATR=13.90 |
| Stop hit — per-position SL triggered | 2025-07-24 11:40:00 | 8368.90 | 8365.58 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:50:00 | 8050.50 | 7971.85 | 0.00 | ORB-long ORB[7930.50,7990.00] vol=2.2x ATR=19.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:00:00 | 8079.92 | 7980.35 | 0.00 | T1 1.5R @ 8079.92 |
| Stop hit — per-position SL triggered | 2025-07-31 11:10:00 | 8050.50 | 7983.54 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 09:30:00 | 8184.00 | 8129.55 | 0.00 | ORB-long ORB[8044.50,8148.00] vol=1.7x ATR=23.02 |
| Stop hit — per-position SL triggered | 2025-08-04 09:40:00 | 8160.98 | 8142.51 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 8146.00 | 8196.46 | 0.00 | ORB-short ORB[8210.50,8272.50] vol=2.3x ATR=19.96 |
| Stop hit — per-position SL triggered | 2025-08-11 11:20:00 | 8165.96 | 8195.13 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 09:40:00 | 8197.00 | 8211.36 | 0.00 | ORB-short ORB[8201.00,8260.50] vol=5.0x ATR=18.34 |
| Stop hit — per-position SL triggered | 2025-08-13 09:50:00 | 8215.34 | 8211.51 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:15:00 | 8198.50 | 8225.22 | 0.00 | ORB-short ORB[8228.50,8280.00] vol=3.5x ATR=11.85 |
| Stop hit — per-position SL triggered | 2025-08-14 11:20:00 | 8210.35 | 8224.68 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:45:00 | 8718.50 | 8642.61 | 0.00 | ORB-long ORB[8586.00,8628.50] vol=3.6x ATR=25.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 09:50:00 | 8756.42 | 8685.36 | 0.00 | T1 1.5R @ 8756.42 |
| Target hit | 2025-08-19 10:35:00 | 8742.50 | 8742.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:50:00 | 8749.00 | 8780.90 | 0.00 | ORB-short ORB[8775.00,8869.00] vol=1.8x ATR=17.23 |
| Stop hit — per-position SL triggered | 2025-08-21 10:00:00 | 8766.23 | 8776.74 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:00:00 | 8597.00 | 8623.02 | 0.00 | ORB-short ORB[8645.00,8692.00] vol=2.0x ATR=17.33 |
| Stop hit — per-position SL triggered | 2025-08-22 10:05:00 | 8614.33 | 8619.77 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:35:00 | 8774.50 | 8724.03 | 0.00 | ORB-long ORB[8665.00,8755.00] vol=2.2x ATR=20.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 10:40:00 | 8804.81 | 8735.40 | 0.00 | T1 1.5R @ 8804.81 |
| Target hit | 2025-09-01 15:20:00 | 8985.50 | 8896.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-09-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:50:00 | 9068.50 | 8988.08 | 0.00 | ORB-long ORB[8930.50,9004.00] vol=2.0x ATR=27.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:35:00 | 9109.00 | 9032.61 | 0.00 | T1 1.5R @ 9109.00 |
| Stop hit — per-position SL triggered | 2025-09-02 11:45:00 | 9068.50 | 9034.94 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 9165.00 | 9121.46 | 0.00 | ORB-long ORB[9073.00,9143.50] vol=2.3x ATR=20.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:35:00 | 9195.62 | 9146.06 | 0.00 | T1 1.5R @ 9195.62 |
| Target hit | 2025-09-08 15:20:00 | 9435.00 | 9360.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-09-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:35:00 | 9142.50 | 9168.31 | 0.00 | ORB-short ORB[9162.50,9243.50] vol=1.9x ATR=21.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:25:00 | 9110.14 | 9151.95 | 0.00 | T1 1.5R @ 9110.14 |
| Target hit | 2025-09-11 13:40:00 | 9128.50 | 9127.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:15:00 | 9079.00 | 9080.76 | 0.00 | ORB-short ORB[9080.00,9115.00] vol=1.9x ATR=12.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:35:00 | 9060.76 | 9075.13 | 0.00 | T1 1.5R @ 9060.76 |
| Target hit | 2025-09-18 14:20:00 | 9063.50 | 9062.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — SELL (started 2025-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 09:55:00 | 8975.00 | 9002.62 | 0.00 | ORB-short ORB[8988.00,9063.50] vol=1.8x ATR=14.65 |
| Stop hit — per-position SL triggered | 2025-09-19 10:05:00 | 8989.65 | 8999.93 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:45:00 | 9136.00 | 9079.26 | 0.00 | ORB-long ORB[8959.50,9057.00] vol=1.6x ATR=19.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 12:10:00 | 9165.21 | 9101.76 | 0.00 | T1 1.5R @ 9165.21 |
| Stop hit — per-position SL triggered | 2025-09-22 13:50:00 | 9136.00 | 9126.67 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:15:00 | 8862.00 | 8918.82 | 0.00 | ORB-short ORB[8885.00,8999.00] vol=1.6x ATR=20.87 |
| Stop hit — per-position SL triggered | 2025-09-24 12:40:00 | 8882.87 | 8900.29 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 8850.00 | 8814.02 | 0.00 | ORB-long ORB[8770.50,8844.00] vol=1.5x ATR=25.64 |
| Stop hit — per-position SL triggered | 2025-09-26 10:10:00 | 8824.36 | 8829.54 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 11:00:00 | 8696.00 | 8717.34 | 0.00 | ORB-short ORB[8712.00,8769.00] vol=2.6x ATR=19.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:25:00 | 8666.63 | 8713.16 | 0.00 | T1 1.5R @ 8666.63 |
| Stop hit — per-position SL triggered | 2025-09-30 14:50:00 | 8696.00 | 8687.68 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:50:00 | 8684.00 | 8756.13 | 0.00 | ORB-short ORB[8719.00,8846.50] vol=2.0x ATR=21.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 12:35:00 | 8652.07 | 8721.69 | 0.00 | T1 1.5R @ 8652.07 |
| Target hit | 2025-10-01 15:20:00 | 8622.00 | 8693.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-10-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:00:00 | 8840.00 | 8815.88 | 0.00 | ORB-long ORB[8790.00,8839.00] vol=1.9x ATR=18.64 |
| Stop hit — per-position SL triggered | 2025-10-07 10:25:00 | 8821.36 | 8821.26 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:50:00 | 8802.50 | 8829.91 | 0.00 | ORB-short ORB[8831.00,8903.00] vol=2.0x ATR=17.64 |
| Stop hit — per-position SL triggered | 2025-10-08 11:05:00 | 8820.14 | 8827.48 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 8884.00 | 8839.74 | 0.00 | ORB-long ORB[8781.50,8853.00] vol=1.7x ATR=18.33 |
| Stop hit — per-position SL triggered | 2025-10-10 09:50:00 | 8865.67 | 8853.36 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:40:00 | 9058.00 | 9018.29 | 0.00 | ORB-long ORB[8917.00,9033.00] vol=3.7x ATR=24.43 |
| Stop hit — per-position SL triggered | 2025-10-13 10:05:00 | 9033.57 | 9029.83 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:15:00 | 9171.00 | 9134.16 | 0.00 | ORB-long ORB[9110.00,9164.00] vol=2.1x ATR=18.35 |
| Stop hit — per-position SL triggered | 2025-10-17 11:40:00 | 9152.65 | 9155.04 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 9147.50 | 9167.00 | 0.00 | ORB-short ORB[9150.50,9236.50] vol=2.6x ATR=16.20 |
| Stop hit — per-position SL triggered | 2025-10-20 11:20:00 | 9163.70 | 9166.60 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 9044.00 | 9085.21 | 0.00 | ORB-short ORB[9090.00,9130.00] vol=2.1x ATR=13.37 |
| Stop hit — per-position SL triggered | 2025-10-28 11:05:00 | 9057.37 | 9079.19 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-31 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:50:00 | 9012.00 | 9001.12 | 0.00 | ORB-long ORB[8901.50,8972.50] vol=1.8x ATR=17.69 |
| Stop hit — per-position SL triggered | 2025-10-31 11:20:00 | 8994.31 | 9001.68 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 11:05:00 | 8867.00 | 8892.60 | 0.00 | ORB-short ORB[8871.50,8957.50] vol=1.5x ATR=19.23 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 8886.23 | 8891.73 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:35:00 | 8818.50 | 8847.60 | 0.00 | ORB-short ORB[8830.00,8920.00] vol=1.8x ATR=16.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:45:00 | 8793.94 | 8830.58 | 0.00 | T1 1.5R @ 8793.94 |
| Stop hit — per-position SL triggered | 2025-11-04 09:55:00 | 8818.50 | 8825.62 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:35:00 | 8605.00 | 8644.21 | 0.00 | ORB-short ORB[8622.00,8730.00] vol=1.8x ATR=20.86 |
| Stop hit — per-position SL triggered | 2025-11-07 09:40:00 | 8625.86 | 8642.87 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 11:00:00 | 8986.00 | 8962.40 | 0.00 | ORB-long ORB[8922.00,8978.00] vol=1.6x ATR=17.20 |
| Stop hit — per-position SL triggered | 2025-11-18 11:05:00 | 8968.80 | 8962.90 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 11:00:00 | 8958.00 | 8925.79 | 0.00 | ORB-long ORB[8890.00,8939.00] vol=2.3x ATR=14.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:10:00 | 8979.89 | 8936.85 | 0.00 | T1 1.5R @ 8979.89 |
| Stop hit — per-position SL triggered | 2025-11-20 11:50:00 | 8958.00 | 8951.36 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 11:10:00 | 9076.50 | 9056.35 | 0.00 | ORB-long ORB[9001.00,9054.00] vol=1.5x ATR=15.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:45:00 | 9100.38 | 9066.66 | 0.00 | T1 1.5R @ 9100.38 |
| Stop hit — per-position SL triggered | 2025-11-25 14:25:00 | 9076.50 | 9087.83 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:15:00 | 9102.50 | 9084.82 | 0.00 | ORB-long ORB[9028.00,9085.00] vol=1.8x ATR=17.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 12:15:00 | 9129.44 | 9102.21 | 0.00 | T1 1.5R @ 9129.44 |
| Target hit | 2025-11-26 15:20:00 | 9161.00 | 9129.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:55:00 | 8970.00 | 8980.21 | 0.00 | ORB-short ORB[9012.50,9106.00] vol=3.3x ATR=19.33 |
| Stop hit — per-position SL triggered | 2025-11-28 11:45:00 | 8989.33 | 8978.79 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:45:00 | 8997.00 | 9037.88 | 0.00 | ORB-short ORB[9026.00,9105.00] vol=2.0x ATR=18.79 |
| Stop hit — per-position SL triggered | 2025-12-03 09:55:00 | 9015.79 | 9028.83 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:35:00 | 9068.50 | 9040.22 | 0.00 | ORB-long ORB[8988.00,9034.50] vol=6.2x ATR=17.89 |
| Stop hit — per-position SL triggered | 2025-12-04 09:45:00 | 9050.61 | 9043.21 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:20:00 | 8972.00 | 8946.69 | 0.00 | ORB-long ORB[8904.00,8965.00] vol=2.4x ATR=17.74 |
| Stop hit — per-position SL triggered | 2025-12-10 10:40:00 | 8954.26 | 8947.80 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:10:00 | 9038.50 | 9004.23 | 0.00 | ORB-long ORB[8961.00,9004.00] vol=1.6x ATR=12.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:55:00 | 9057.04 | 9013.39 | 0.00 | T1 1.5R @ 9057.04 |
| Stop hit — per-position SL triggered | 2025-12-11 14:05:00 | 9038.50 | 9036.36 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:20:00 | 8926.50 | 8939.80 | 0.00 | ORB-short ORB[8940.00,9042.50] vol=2.3x ATR=14.26 |
| Stop hit — per-position SL triggered | 2025-12-15 10:45:00 | 8940.76 | 8936.92 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 11:05:00 | 8940.00 | 8896.09 | 0.00 | ORB-long ORB[8827.00,8905.00] vol=3.7x ATR=13.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:20:00 | 8960.41 | 8906.13 | 0.00 | T1 1.5R @ 8960.41 |
| Stop hit — per-position SL triggered | 2025-12-19 11:25:00 | 8940.00 | 8908.02 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 09:30:00 | 9109.50 | 9081.46 | 0.00 | ORB-long ORB[8997.50,9094.50] vol=2.6x ATR=18.80 |
| Stop hit — per-position SL triggered | 2025-12-22 10:05:00 | 9090.70 | 9098.28 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 9128.00 | 9110.65 | 0.00 | ORB-long ORB[9070.00,9125.00] vol=2.3x ATR=12.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:25:00 | 9147.36 | 9119.31 | 0.00 | T1 1.5R @ 9147.36 |
| Target hit | 2025-12-24 15:20:00 | 9171.50 | 9155.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2025-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:45:00 | 9158.00 | 9110.93 | 0.00 | ORB-long ORB[9025.50,9094.50] vol=1.9x ATR=20.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:55:00 | 9189.04 | 9132.62 | 0.00 | T1 1.5R @ 9189.04 |
| Stop hit — per-position SL triggered | 2025-12-30 10:00:00 | 9158.00 | 9135.96 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:00:00 | 9397.00 | 9349.23 | 0.00 | ORB-long ORB[9296.50,9380.00] vol=2.7x ATR=20.44 |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 9376.56 | 9351.64 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:50:00 | 9635.50 | 9552.45 | 0.00 | ORB-long ORB[9472.00,9525.00] vol=1.7x ATR=24.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:55:00 | 9671.51 | 9564.82 | 0.00 | T1 1.5R @ 9671.51 |
| Stop hit — per-position SL triggered | 2026-01-05 11:00:00 | 9635.50 | 9573.04 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:35:00 | 9683.50 | 9629.33 | 0.00 | ORB-long ORB[9543.00,9673.50] vol=2.1x ATR=27.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:45:00 | 9724.29 | 9669.88 | 0.00 | T1 1.5R @ 9724.29 |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 9683.50 | 9683.55 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:35:00 | 9854.50 | 9824.32 | 0.00 | ORB-long ORB[9749.50,9841.50] vol=2.7x ATR=23.99 |
| Stop hit — per-position SL triggered | 2026-01-08 09:45:00 | 9830.51 | 9828.25 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 11:10:00 | 9378.00 | 9469.96 | 0.00 | ORB-short ORB[9462.00,9588.50] vol=3.8x ATR=30.28 |
| Stop hit — per-position SL triggered | 2026-01-12 11:50:00 | 9408.28 | 9447.79 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-01-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:00:00 | 9489.00 | 9560.41 | 0.00 | ORB-short ORB[9528.50,9616.50] vol=1.7x ATR=21.10 |
| Stop hit — per-position SL triggered | 2026-01-16 11:25:00 | 9510.10 | 9546.32 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 10:50:00 | 9291.50 | 9323.69 | 0.00 | ORB-short ORB[9340.00,9468.00] vol=9.4x ATR=24.51 |
| Stop hit — per-position SL triggered | 2026-01-20 11:00:00 | 9316.01 | 9322.49 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:55:00 | 9120.00 | 9166.10 | 0.00 | ORB-short ORB[9150.00,9255.00] vol=1.7x ATR=26.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:40:00 | 9080.70 | 9144.68 | 0.00 | T1 1.5R @ 9080.70 |
| Stop hit — per-position SL triggered | 2026-01-21 12:00:00 | 9120.00 | 9140.54 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:30:00 | 9483.00 | 9421.35 | 0.00 | ORB-long ORB[9335.00,9428.50] vol=1.9x ATR=23.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:45:00 | 9518.29 | 9444.07 | 0.00 | T1 1.5R @ 9518.29 |
| Stop hit — per-position SL triggered | 2026-01-23 11:00:00 | 9483.00 | 9461.31 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:00:00 | 9435.00 | 9455.86 | 0.00 | ORB-short ORB[9444.50,9528.50] vol=1.6x ATR=27.13 |
| Stop hit — per-position SL triggered | 2026-01-29 11:30:00 | 9462.13 | 9453.15 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:05:00 | 9558.00 | 9668.17 | 0.00 | ORB-short ORB[9575.50,9712.00] vol=1.6x ATR=25.37 |
| Stop hit — per-position SL triggered | 2026-02-04 11:35:00 | 9583.37 | 9645.12 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-02-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:35:00 | 9499.00 | 9576.86 | 0.00 | ORB-short ORB[9581.00,9668.00] vol=2.4x ATR=26.62 |
| Stop hit — per-position SL triggered | 2026-02-06 09:45:00 | 9525.62 | 9564.37 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 9680.50 | 9657.57 | 0.00 | ORB-long ORB[9582.50,9672.50] vol=1.9x ATR=19.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:45:00 | 9709.75 | 9670.79 | 0.00 | T1 1.5R @ 9709.75 |
| Target hit | 2026-02-10 14:40:00 | 9769.00 | 9773.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 82 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 9873.50 | 9834.66 | 0.00 | ORB-long ORB[9804.50,9842.00] vol=1.5x ATR=19.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:35:00 | 9903.16 | 9850.93 | 0.00 | T1 1.5R @ 9903.16 |
| Target hit | 2026-02-18 10:10:00 | 9889.50 | 9891.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 9972.00 | 9917.84 | 0.00 | ORB-long ORB[9839.50,9928.00] vol=2.2x ATR=24.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:35:00 | 10008.80 | 9936.61 | 0.00 | T1 1.5R @ 10008.80 |
| Target hit | 2026-02-25 13:10:00 | 10039.00 | 10044.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 84 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 10044.50 | 10111.76 | 0.00 | ORB-short ORB[10103.00,10187.00] vol=2.0x ATR=20.96 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 10065.46 | 10110.78 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 10010.00 | 10043.74 | 0.00 | ORB-short ORB[10037.00,10091.50] vol=2.1x ATR=19.19 |
| Stop hit — per-position SL triggered | 2026-02-27 10:35:00 | 10029.19 | 10038.57 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-03-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:05:00 | 9507.00 | 9595.08 | 0.00 | ORB-short ORB[9540.50,9650.00] vol=1.9x ATR=39.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:40:00 | 9448.42 | 9569.20 | 0.00 | T1 1.5R @ 9448.42 |
| Target hit | 2026-03-11 15:20:00 | 9327.50 | 9414.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2026-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:35:00 | 8887.00 | 8993.50 | 0.00 | ORB-short ORB[9053.00,9113.50] vol=1.6x ATR=31.66 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 8918.66 | 8983.26 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-03-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-25 09:40:00 | 8995.00 | 9014.22 | 0.00 | ORB-short ORB[9002.00,9062.00] vol=1.7x ATR=33.32 |
| Stop hit — per-position SL triggered | 2026-03-25 09:45:00 | 9028.32 | 9014.66 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-04-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:00:00 | 8876.50 | 8844.64 | 0.00 | ORB-long ORB[8755.00,8851.00] vol=2.2x ATR=33.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:05:00 | 8926.60 | 8854.10 | 0.00 | T1 1.5R @ 8926.60 |
| Stop hit — per-position SL triggered | 2026-04-06 10:25:00 | 8876.50 | 8873.28 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:15:00 | 8955.00 | 8892.69 | 0.00 | ORB-long ORB[8800.50,8915.50] vol=2.1x ATR=29.28 |
| Stop hit — per-position SL triggered | 2026-04-07 10:50:00 | 8925.72 | 8922.34 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 9850.00 | 9900.62 | 0.00 | ORB-short ORB[9880.00,9976.00] vol=3.7x ATR=35.55 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 9885.55 | 9893.85 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 9796.50 | 9833.26 | 0.00 | ORB-short ORB[9837.00,9918.50] vol=3.0x ATR=23.74 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 9820.24 | 9813.20 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 9645.00 | 9632.12 | 0.00 | ORB-long ORB[9496.00,9640.00] vol=1.9x ATR=26.47 |
| Stop hit — per-position SL triggered | 2026-04-29 10:35:00 | 9618.53 | 9633.01 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:35:00 | 9573.00 | 9438.77 | 0.00 | ORB-long ORB[9411.00,9530.00] vol=4.7x ATR=35.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:40:00 | 9626.56 | 9569.12 | 0.00 | T1 1.5R @ 9626.56 |
| Target hit | 2026-04-30 15:20:00 | 10024.50 | 9808.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 95 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 10721.50 | 10645.93 | 0.00 | ORB-long ORB[10540.00,10647.50] vol=1.7x ATR=24.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:45:00 | 10758.93 | 10673.38 | 0.00 | T1 1.5R @ 10758.93 |
| Stop hit — per-position SL triggered | 2026-05-08 12:45:00 | 10721.50 | 10692.58 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:35:00 | 8172.00 | 2025-05-14 10:10:00 | 8145.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-16 11:00:00 | 8395.00 | 2025-05-16 11:30:00 | 8428.13 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-05-16 11:00:00 | 8395.00 | 2025-05-16 11:50:00 | 8395.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-21 09:45:00 | 8675.50 | 2025-05-21 10:05:00 | 8725.34 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-05-21 09:45:00 | 8675.50 | 2025-05-21 11:35:00 | 8675.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-26 09:30:00 | 8915.50 | 2025-05-26 09:35:00 | 8889.09 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-05 10:45:00 | 8559.50 | 2025-06-05 11:25:00 | 8572.11 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-06 10:35:00 | 8681.00 | 2025-06-06 10:40:00 | 8655.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-09 10:10:00 | 8623.00 | 2025-06-09 10:45:00 | 8641.81 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-10 09:50:00 | 8602.50 | 2025-06-10 10:50:00 | 8577.66 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-10 09:50:00 | 8602.50 | 2025-06-10 11:05:00 | 8602.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-11 10:00:00 | 8738.00 | 2025-06-11 10:25:00 | 8718.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-16 11:00:00 | 8561.00 | 2025-06-16 11:15:00 | 8586.74 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-06-16 11:00:00 | 8561.00 | 2025-06-16 11:20:00 | 8561.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-17 09:40:00 | 8467.50 | 2025-06-17 10:35:00 | 8484.15 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-18 09:30:00 | 8598.00 | 2025-06-18 09:35:00 | 8628.03 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-06-18 09:30:00 | 8598.00 | 2025-06-18 10:25:00 | 8598.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-19 11:00:00 | 8548.50 | 2025-06-19 11:10:00 | 8532.13 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-01 10:55:00 | 8313.00 | 2025-07-01 11:05:00 | 8328.94 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-04 10:45:00 | 8442.50 | 2025-07-04 10:55:00 | 8429.29 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-07 11:00:00 | 8470.50 | 2025-07-07 11:20:00 | 8457.21 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-08 10:15:00 | 8360.50 | 2025-07-08 10:20:00 | 8338.35 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-08 10:15:00 | 8360.50 | 2025-07-08 15:00:00 | 8335.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-07-09 11:05:00 | 8373.00 | 2025-07-09 11:20:00 | 8361.89 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-07-10 09:30:00 | 8353.00 | 2025-07-10 10:05:00 | 8364.04 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-07-11 09:30:00 | 8215.50 | 2025-07-11 09:35:00 | 8232.29 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-15 10:35:00 | 8161.00 | 2025-07-15 11:00:00 | 8183.87 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-15 10:35:00 | 8161.00 | 2025-07-15 15:20:00 | 8317.50 | TARGET_HIT | 0.50 | 1.92% |
| SELL | retest1 | 2025-07-16 09:40:00 | 8204.00 | 2025-07-16 10:20:00 | 8222.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-22 11:10:00 | 8326.00 | 2025-07-22 11:30:00 | 8297.64 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-22 11:10:00 | 8326.00 | 2025-07-22 12:25:00 | 8326.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 11:10:00 | 8355.00 | 2025-07-24 11:40:00 | 8368.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-31 10:50:00 | 8050.50 | 2025-07-31 11:00:00 | 8079.92 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-31 10:50:00 | 8050.50 | 2025-07-31 11:10:00 | 8050.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-04 09:30:00 | 8184.00 | 2025-08-04 09:40:00 | 8160.98 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-11 11:10:00 | 8146.00 | 2025-08-11 11:20:00 | 8165.96 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-13 09:40:00 | 8197.00 | 2025-08-13 09:50:00 | 8215.34 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-14 11:15:00 | 8198.50 | 2025-08-14 11:20:00 | 8210.35 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-08-19 09:45:00 | 8718.50 | 2025-08-19 09:50:00 | 8756.42 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-19 09:45:00 | 8718.50 | 2025-08-19 10:35:00 | 8742.50 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-21 09:50:00 | 8749.00 | 2025-08-21 10:00:00 | 8766.23 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-22 10:00:00 | 8597.00 | 2025-08-22 10:05:00 | 8614.33 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-01 10:35:00 | 8774.50 | 2025-09-01 10:40:00 | 8804.81 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-01 10:35:00 | 8774.50 | 2025-09-01 15:20:00 | 8985.50 | TARGET_HIT | 0.50 | 2.40% |
| BUY | retest1 | 2025-09-02 09:50:00 | 9068.50 | 2025-09-02 11:35:00 | 9109.00 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-02 09:50:00 | 9068.50 | 2025-09-02 11:45:00 | 9068.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 09:30:00 | 9165.00 | 2025-09-08 09:35:00 | 9195.62 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-08 09:30:00 | 9165.00 | 2025-09-08 15:20:00 | 9435.00 | TARGET_HIT | 0.50 | 2.95% |
| SELL | retest1 | 2025-09-11 10:35:00 | 9142.50 | 2025-09-11 11:25:00 | 9110.14 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-11 10:35:00 | 9142.50 | 2025-09-11 13:40:00 | 9128.50 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-09-18 10:15:00 | 9079.00 | 2025-09-18 11:35:00 | 9060.76 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-09-18 10:15:00 | 9079.00 | 2025-09-18 14:20:00 | 9063.50 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-09-19 09:55:00 | 8975.00 | 2025-09-19 10:05:00 | 8989.65 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-22 10:45:00 | 9136.00 | 2025-09-22 12:10:00 | 9165.21 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-22 10:45:00 | 9136.00 | 2025-09-22 13:50:00 | 9136.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 11:15:00 | 8862.00 | 2025-09-24 12:40:00 | 8882.87 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-26 09:30:00 | 8850.00 | 2025-09-26 10:10:00 | 8824.36 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-30 11:00:00 | 8696.00 | 2025-09-30 11:25:00 | 8666.63 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-30 11:00:00 | 8696.00 | 2025-09-30 14:50:00 | 8696.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-01 10:50:00 | 8684.00 | 2025-10-01 12:35:00 | 8652.07 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-01 10:50:00 | 8684.00 | 2025-10-01 15:20:00 | 8622.00 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2025-10-07 10:00:00 | 8840.00 | 2025-10-07 10:25:00 | 8821.36 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-08 10:50:00 | 8802.50 | 2025-10-08 11:05:00 | 8820.14 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-10 09:30:00 | 8884.00 | 2025-10-10 09:50:00 | 8865.67 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-13 09:40:00 | 9058.00 | 2025-10-13 10:05:00 | 9033.57 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-17 10:15:00 | 9171.00 | 2025-10-17 11:40:00 | 9152.65 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-20 11:15:00 | 9147.50 | 2025-10-20 11:20:00 | 9163.70 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-28 10:50:00 | 9044.00 | 2025-10-28 11:05:00 | 9057.37 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-10-31 10:50:00 | 9012.00 | 2025-10-31 11:20:00 | 8994.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-03 11:05:00 | 8867.00 | 2025-11-03 11:15:00 | 8886.23 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-04 09:35:00 | 8818.50 | 2025-11-04 09:45:00 | 8793.94 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-04 09:35:00 | 8818.50 | 2025-11-04 09:55:00 | 8818.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:35:00 | 8605.00 | 2025-11-07 09:40:00 | 8625.86 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-18 11:00:00 | 8986.00 | 2025-11-18 11:05:00 | 8968.80 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-20 11:00:00 | 8958.00 | 2025-11-20 11:10:00 | 8979.89 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-20 11:00:00 | 8958.00 | 2025-11-20 11:50:00 | 8958.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-25 11:10:00 | 9076.50 | 2025-11-25 12:45:00 | 9100.38 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-11-25 11:10:00 | 9076.50 | 2025-11-25 14:25:00 | 9076.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 10:15:00 | 9102.50 | 2025-11-26 12:15:00 | 9129.44 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-11-26 10:15:00 | 9102.50 | 2025-11-26 15:20:00 | 9161.00 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2025-11-28 10:55:00 | 8970.00 | 2025-11-28 11:45:00 | 8989.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-03 09:45:00 | 8997.00 | 2025-12-03 09:55:00 | 9015.79 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-04 09:35:00 | 9068.50 | 2025-12-04 09:45:00 | 9050.61 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-10 10:20:00 | 8972.00 | 2025-12-10 10:40:00 | 8954.26 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-11 11:10:00 | 9038.50 | 2025-12-11 11:55:00 | 9057.04 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-12-11 11:10:00 | 9038.50 | 2025-12-11 14:05:00 | 9038.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-15 10:20:00 | 8926.50 | 2025-12-15 10:45:00 | 8940.76 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-19 11:05:00 | 8940.00 | 2025-12-19 11:20:00 | 8960.41 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-12-19 11:05:00 | 8940.00 | 2025-12-19 11:25:00 | 8940.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 09:30:00 | 9109.50 | 2025-12-22 10:05:00 | 9090.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-24 10:55:00 | 9128.00 | 2025-12-24 11:25:00 | 9147.36 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-12-24 10:55:00 | 9128.00 | 2025-12-24 15:20:00 | 9171.50 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-12-30 09:45:00 | 9158.00 | 2025-12-30 09:55:00 | 9189.04 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-30 09:45:00 | 9158.00 | 2025-12-30 10:00:00 | 9158.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 11:00:00 | 9397.00 | 2025-12-31 11:15:00 | 9376.56 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-05 10:50:00 | 9635.50 | 2026-01-05 10:55:00 | 9671.51 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-05 10:50:00 | 9635.50 | 2026-01-05 11:00:00 | 9635.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 09:35:00 | 9683.50 | 2026-01-06 09:45:00 | 9724.29 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-06 09:35:00 | 9683.50 | 2026-01-06 10:15:00 | 9683.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-08 09:35:00 | 9854.50 | 2026-01-08 09:45:00 | 9830.51 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-12 11:10:00 | 9378.00 | 2026-01-12 11:50:00 | 9408.28 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-16 11:00:00 | 9489.00 | 2026-01-16 11:25:00 | 9510.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-20 10:50:00 | 9291.50 | 2026-01-20 11:00:00 | 9316.01 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-21 10:55:00 | 9120.00 | 2026-01-21 11:40:00 | 9080.70 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-21 10:55:00 | 9120.00 | 2026-01-21 12:00:00 | 9120.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 10:30:00 | 9483.00 | 2026-01-23 10:45:00 | 9518.29 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-23 10:30:00 | 9483.00 | 2026-01-23 11:00:00 | 9483.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 11:00:00 | 9435.00 | 2026-01-29 11:30:00 | 9462.13 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-04 11:05:00 | 9558.00 | 2026-02-04 11:35:00 | 9583.37 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-06 09:35:00 | 9499.00 | 2026-02-06 09:45:00 | 9525.62 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-10 09:35:00 | 9680.50 | 2026-02-10 09:45:00 | 9709.75 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-10 09:35:00 | 9680.50 | 2026-02-10 14:40:00 | 9769.00 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2026-02-18 09:30:00 | 9873.50 | 2026-02-18 09:35:00 | 9903.16 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-18 09:30:00 | 9873.50 | 2026-02-18 10:10:00 | 9889.50 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2026-02-25 10:25:00 | 9972.00 | 2026-02-25 10:35:00 | 10008.80 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-25 10:25:00 | 9972.00 | 2026-02-25 13:10:00 | 10039.00 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-02-26 11:00:00 | 10044.50 | 2026-02-26 11:05:00 | 10065.46 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-27 10:20:00 | 10010.00 | 2026-02-27 10:35:00 | 10029.19 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-03-11 10:05:00 | 9507.00 | 2026-03-11 10:40:00 | 9448.42 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2026-03-11 10:05:00 | 9507.00 | 2026-03-11 15:20:00 | 9327.50 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2026-03-13 10:35:00 | 8887.00 | 2026-03-13 10:50:00 | 8918.66 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-25 09:40:00 | 8995.00 | 2026-03-25 09:45:00 | 9028.32 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-06 10:00:00 | 8876.50 | 2026-04-06 10:05:00 | 8926.60 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-06 10:00:00 | 8876.50 | 2026-04-06 10:25:00 | 8876.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 10:15:00 | 8955.00 | 2026-04-07 10:50:00 | 8925.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-15 09:30:00 | 9850.00 | 2026-04-15 09:40:00 | 9885.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-16 09:50:00 | 9796.50 | 2026-04-16 11:40:00 | 9820.24 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-29 10:20:00 | 9645.00 | 2026-04-29 10:35:00 | 9618.53 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-30 10:35:00 | 9573.00 | 2026-04-30 10:40:00 | 9626.56 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-30 10:35:00 | 9573.00 | 2026-04-30 15:20:00 | 10024.50 | TARGET_HIT | 0.50 | 4.72% |
| BUY | retest1 | 2026-05-08 10:45:00 | 10721.50 | 2026-05-08 11:45:00 | 10758.93 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-05-08 10:45:00 | 10721.50 | 2026-05-08 12:45:00 | 10721.50 | STOP_HIT | 0.50 | 0.00% |
