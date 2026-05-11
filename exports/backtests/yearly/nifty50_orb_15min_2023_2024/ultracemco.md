# ULTRACEMCO (ULTRACEMCO)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (51079 bars)
- **Last close:** 11930.00
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 10 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 87 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 53
- **Target hits / Stop hits / Partials:** 10 / 53 / 24
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 9.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 20 | 40.0% | 6 | 30 | 14 | 0.13% | 6.5% |
| BUY @ 2nd Alert (retest1) | 50 | 20 | 40.0% | 6 | 30 | 14 | 0.13% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 37 | 14 | 37.8% | 4 | 23 | 10 | 0.09% | 3.3% |
| SELL @ 2nd Alert (retest1) | 37 | 14 | 37.8% | 4 | 23 | 10 | 0.09% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 87 | 34 | 39.1% | 10 | 53 | 24 | 0.11% | 9.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 7610.95 | 7628.67 | 0.00 | ORB-short ORB[7616.00,7661.50] vol=1.7x ATR=16.71 |
| Stop hit — per-position SL triggered | 2023-05-19 09:45:00 | 7627.66 | 7623.31 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 7942.45 | 7917.71 | 0.00 | ORB-long ORB[7861.00,7934.00] vol=1.9x ATR=17.98 |
| Stop hit — per-position SL triggered | 2023-06-02 09:40:00 | 7924.47 | 7920.74 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-06-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 11:05:00 | 7906.85 | 7901.74 | 0.00 | ORB-long ORB[7874.05,7905.00] vol=2.2x ATR=11.43 |
| Stop hit — per-position SL triggered | 2023-06-05 12:10:00 | 7895.42 | 7901.90 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:35:00 | 8204.00 | 8184.45 | 0.00 | ORB-long ORB[8160.00,8193.90] vol=1.8x ATR=13.49 |
| Stop hit — per-position SL triggered | 2023-06-08 09:50:00 | 8190.51 | 8188.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-06-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 10:30:00 | 8198.00 | 8167.16 | 0.00 | ORB-long ORB[8108.20,8158.00] vol=1.6x ATR=16.75 |
| Stop hit — per-position SL triggered | 2023-06-09 10:40:00 | 8181.25 | 8169.17 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:50:00 | 8273.90 | 8243.21 | 0.00 | ORB-long ORB[8211.75,8265.00] vol=1.7x ATR=13.00 |
| Stop hit — per-position SL triggered | 2023-06-13 10:55:00 | 8260.90 | 8245.11 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:50:00 | 8180.40 | 8216.57 | 0.00 | ORB-short ORB[8211.80,8291.40] vol=1.9x ATR=13.26 |
| Stop hit — per-position SL triggered | 2023-06-20 11:00:00 | 8193.66 | 8212.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 10:45:00 | 8238.85 | 8215.43 | 0.00 | ORB-long ORB[8182.00,8222.00] vol=1.5x ATR=12.30 |
| Stop hit — per-position SL triggered | 2023-06-28 10:55:00 | 8226.55 | 8218.44 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-07-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:40:00 | 8403.20 | 8423.04 | 0.00 | ORB-short ORB[8408.50,8452.00] vol=2.0x ATR=13.38 |
| Stop hit — per-position SL triggered | 2023-07-05 11:30:00 | 8416.58 | 8416.93 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 10:45:00 | 8457.25 | 8421.51 | 0.00 | ORB-long ORB[8380.00,8447.20] vol=2.2x ATR=14.97 |
| Stop hit — per-position SL triggered | 2023-07-06 11:10:00 | 8442.28 | 8427.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:30:00 | 8369.75 | 8329.09 | 0.00 | ORB-long ORB[8244.70,8369.15] vol=1.7x ATR=21.95 |
| Stop hit — per-position SL triggered | 2023-07-25 09:40:00 | 8347.80 | 8333.62 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 11:00:00 | 8377.00 | 8364.58 | 0.00 | ORB-long ORB[8301.85,8348.00] vol=1.6x ATR=14.49 |
| Stop hit — per-position SL triggered | 2023-07-31 11:05:00 | 8362.51 | 8364.71 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:00:00 | 8239.00 | 8260.20 | 0.00 | ORB-short ORB[8251.25,8315.00] vol=1.9x ATR=16.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:40:00 | 8214.49 | 8254.90 | 0.00 | T1 1.5R @ 8214.49 |
| Stop hit — per-position SL triggered | 2023-08-02 11:45:00 | 8239.00 | 8254.56 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-08-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 11:10:00 | 8166.45 | 8186.54 | 0.00 | ORB-short ORB[8175.45,8229.95] vol=1.6x ATR=12.06 |
| Stop hit — per-position SL triggered | 2023-08-08 11:25:00 | 8178.51 | 8183.60 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-08-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:30:00 | 8099.75 | 8153.09 | 0.00 | ORB-short ORB[8116.20,8205.00] vol=1.8x ATR=21.64 |
| Stop hit — per-position SL triggered | 2023-08-10 10:55:00 | 8121.39 | 8138.14 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 11:15:00 | 8048.75 | 8032.22 | 0.00 | ORB-long ORB[7987.65,8032.70] vol=1.7x ATR=12.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 11:55:00 | 8068.17 | 8037.57 | 0.00 | T1 1.5R @ 8068.17 |
| Target hit | 2023-08-16 15:20:00 | 8262.20 | 8148.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 8158.75 | 8189.03 | 0.00 | ORB-short ORB[8200.10,8233.10] vol=4.0x ATR=14.68 |
| Stop hit — per-position SL triggered | 2023-08-17 11:35:00 | 8173.43 | 8186.54 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-08-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 10:55:00 | 8180.00 | 8203.13 | 0.00 | ORB-short ORB[8200.00,8234.90] vol=2.3x ATR=10.70 |
| Stop hit — per-position SL triggered | 2023-08-23 14:20:00 | 8190.70 | 8187.73 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-08-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:40:00 | 8307.05 | 8252.22 | 0.00 | ORB-long ORB[8164.00,8226.80] vol=2.5x ATR=18.89 |
| Stop hit — per-position SL triggered | 2023-08-24 09:50:00 | 8288.16 | 8266.53 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 10:30:00 | 8340.90 | 8284.36 | 0.00 | ORB-long ORB[8226.00,8288.80] vol=2.7x ATR=15.20 |
| Stop hit — per-position SL triggered | 2023-08-31 10:35:00 | 8325.70 | 8288.48 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-09-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 10:55:00 | 8463.80 | 8508.72 | 0.00 | ORB-short ORB[8501.15,8550.00] vol=1.5x ATR=13.58 |
| Stop hit — per-position SL triggered | 2023-09-08 12:15:00 | 8477.38 | 8494.16 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:30:00 | 8262.00 | 8221.72 | 0.00 | ORB-long ORB[8179.20,8229.65] vol=2.8x ATR=15.76 |
| Stop hit — per-position SL triggered | 2023-10-11 09:55:00 | 8246.24 | 8241.44 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-10-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:10:00 | 8448.55 | 8420.80 | 0.00 | ORB-long ORB[8369.05,8442.75] vol=1.9x ATR=19.29 |
| Stop hit — per-position SL triggered | 2023-10-16 10:20:00 | 8429.26 | 8421.97 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-10-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:10:00 | 8299.35 | 8320.89 | 0.00 | ORB-short ORB[8306.70,8349.85] vol=1.5x ATR=10.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 10:55:00 | 8283.63 | 8312.91 | 0.00 | T1 1.5R @ 8283.63 |
| Stop hit — per-position SL triggered | 2023-10-18 11:05:00 | 8299.35 | 8311.05 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-10-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:50:00 | 8327.30 | 8283.00 | 0.00 | ORB-long ORB[8230.00,8275.00] vol=1.5x ATR=21.94 |
| Stop hit — per-position SL triggered | 2023-10-30 11:00:00 | 8305.36 | 8284.45 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-07 11:15:00 | 8648.00 | 8622.29 | 0.00 | ORB-long ORB[8607.95,8647.15] vol=1.6x ATR=11.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 12:00:00 | 8664.95 | 8634.11 | 0.00 | T1 1.5R @ 8664.95 |
| Stop hit — per-position SL triggered | 2023-11-07 12:20:00 | 8648.00 | 8637.91 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-11-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:35:00 | 8667.90 | 8680.51 | 0.00 | ORB-short ORB[8676.00,8719.00] vol=2.4x ATR=13.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:15:00 | 8648.32 | 8676.34 | 0.00 | T1 1.5R @ 8648.32 |
| Stop hit — per-position SL triggered | 2023-11-09 12:00:00 | 8667.90 | 8671.71 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-11-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 11:05:00 | 8664.65 | 8638.50 | 0.00 | ORB-long ORB[8588.60,8646.80] vol=1.6x ATR=14.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 12:10:00 | 8686.28 | 8648.05 | 0.00 | T1 1.5R @ 8686.28 |
| Target hit | 2023-11-10 15:20:00 | 8718.20 | 8679.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2023-11-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:30:00 | 8773.60 | 8741.70 | 0.00 | ORB-long ORB[8695.25,8746.10] vol=4.1x ATR=17.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 09:35:00 | 8799.15 | 8768.79 | 0.00 | T1 1.5R @ 8799.15 |
| Stop hit — per-position SL triggered | 2023-11-15 09:50:00 | 8773.60 | 8770.62 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-11-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 10:10:00 | 8862.00 | 8833.95 | 0.00 | ORB-long ORB[8784.45,8844.80] vol=2.0x ATR=17.61 |
| Stop hit — per-position SL triggered | 2023-11-17 10:15:00 | 8844.39 | 8835.68 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-11-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:45:00 | 8609.55 | 8619.40 | 0.00 | ORB-short ORB[8611.95,8655.00] vol=8.1x ATR=12.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 10:50:00 | 8590.32 | 8618.18 | 0.00 | T1 1.5R @ 8590.32 |
| Stop hit — per-position SL triggered | 2023-11-24 12:05:00 | 8609.55 | 8607.11 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 10:35:00 | 8677.25 | 8658.31 | 0.00 | ORB-long ORB[8588.25,8654.60] vol=1.8x ATR=16.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:20:00 | 8701.32 | 8667.15 | 0.00 | T1 1.5R @ 8701.32 |
| Target hit | 2023-11-28 15:20:00 | 8727.60 | 8697.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2023-12-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 11:05:00 | 9169.95 | 9117.29 | 0.00 | ORB-long ORB[9086.15,9163.85] vol=2.2x ATR=20.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 11:10:00 | 9200.24 | 9129.25 | 0.00 | T1 1.5R @ 9200.24 |
| Target hit | 2023-12-04 15:20:00 | 9324.80 | 9239.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2023-12-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 10:30:00 | 9255.45 | 9282.33 | 0.00 | ORB-short ORB[9293.35,9347.95] vol=1.9x ATR=16.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 11:15:00 | 9230.44 | 9268.86 | 0.00 | T1 1.5R @ 9230.44 |
| Target hit | 2023-12-06 14:15:00 | 9217.80 | 9205.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — BUY (started 2023-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:30:00 | 9508.70 | 9470.94 | 0.00 | ORB-long ORB[9400.00,9498.60] vol=2.4x ATR=20.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 09:35:00 | 9539.79 | 9485.25 | 0.00 | T1 1.5R @ 9539.79 |
| Stop hit — per-position SL triggered | 2023-12-11 09:40:00 | 9508.70 | 9488.40 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:35:00 | 9766.65 | 9719.29 | 0.00 | ORB-long ORB[9636.15,9742.35] vol=2.1x ATR=25.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:30:00 | 9804.87 | 9758.31 | 0.00 | T1 1.5R @ 9804.87 |
| Target hit | 2023-12-12 13:15:00 | 9854.05 | 9855.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2023-12-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:20:00 | 10105.20 | 10080.88 | 0.00 | ORB-long ORB[10010.20,10055.00] vol=1.5x ATR=17.89 |
| Stop hit — per-position SL triggered | 2023-12-20 10:35:00 | 10087.31 | 10082.44 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 11:05:00 | 10145.00 | 10120.41 | 0.00 | ORB-long ORB[10086.00,10139.90] vol=3.2x ATR=18.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 11:20:00 | 10172.36 | 10125.56 | 0.00 | T1 1.5R @ 10172.36 |
| Stop hit — per-position SL triggered | 2024-02-07 11:30:00 | 10145.00 | 10130.33 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 10086.00 | 10155.64 | 0.00 | ORB-short ORB[10181.00,10295.00] vol=2.5x ATR=27.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 11:30:00 | 10044.97 | 10131.12 | 0.00 | T1 1.5R @ 10044.97 |
| Stop hit — per-position SL triggered | 2024-02-08 11:45:00 | 10086.00 | 10126.15 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-02-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 09:30:00 | 9737.60 | 9779.38 | 0.00 | ORB-short ORB[9770.05,9824.85] vol=2.2x ATR=21.99 |
| Stop hit — per-position SL triggered | 2024-02-14 09:35:00 | 9759.59 | 9775.42 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:45:00 | 9928.10 | 9977.25 | 0.00 | ORB-short ORB[9970.00,10049.95] vol=2.1x ATR=19.87 |
| Stop hit — per-position SL triggered | 2024-02-26 15:20:00 | 9934.00 | 9937.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-02-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 11:10:00 | 9833.45 | 9764.67 | 0.00 | ORB-long ORB[9705.65,9785.00] vol=1.6x ATR=22.77 |
| Stop hit — per-position SL triggered | 2024-02-29 12:30:00 | 9810.68 | 9784.84 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-03-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 10:50:00 | 10065.90 | 10002.66 | 0.00 | ORB-long ORB[9892.40,9975.00] vol=1.6x ATR=23.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-01 11:35:00 | 10101.04 | 10029.80 | 0.00 | T1 1.5R @ 10101.04 |
| Stop hit — per-position SL triggered | 2024-03-01 12:15:00 | 10065.90 | 10035.65 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:30:00 | 10083.40 | 10124.38 | 0.00 | ORB-short ORB[10111.35,10166.00] vol=3.3x ATR=29.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 09:45:00 | 10039.48 | 10099.12 | 0.00 | T1 1.5R @ 10039.48 |
| Target hit | 2024-03-04 15:20:00 | 9968.45 | 10009.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2024-03-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 10:00:00 | 9746.90 | 9798.29 | 0.00 | ORB-short ORB[9805.35,9914.25] vol=1.7x ATR=22.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:25:00 | 9713.73 | 9770.56 | 0.00 | T1 1.5R @ 9713.73 |
| Target hit | 2024-03-06 15:05:00 | 9651.50 | 9648.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2024-03-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:30:00 | 9677.50 | 9634.69 | 0.00 | ORB-long ORB[9569.10,9672.00] vol=1.6x ATR=21.84 |
| Stop hit — per-position SL triggered | 2024-03-07 10:40:00 | 9655.66 | 9636.63 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:45:00 | 9624.00 | 9632.28 | 0.00 | ORB-short ORB[9632.00,9728.95] vol=2.0x ATR=17.04 |
| Stop hit — per-position SL triggered | 2024-03-12 11:15:00 | 9641.04 | 9628.70 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-03-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 11:00:00 | 9607.20 | 9550.79 | 0.00 | ORB-long ORB[9504.90,9563.00] vol=1.9x ATR=18.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 11:15:00 | 9634.31 | 9557.24 | 0.00 | T1 1.5R @ 9634.31 |
| Stop hit — per-position SL triggered | 2024-03-21 11:35:00 | 9607.20 | 9563.36 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-03-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:30:00 | 9596.80 | 9622.84 | 0.00 | ORB-short ORB[9631.10,9689.80] vol=2.1x ATR=18.30 |
| Stop hit — per-position SL triggered | 2024-03-26 13:10:00 | 9615.10 | 9605.90 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:50:00 | 9685.35 | 9659.02 | 0.00 | ORB-long ORB[9571.05,9634.00] vol=5.9x ATR=17.66 |
| Stop hit — per-position SL triggered | 2024-03-27 11:50:00 | 9667.69 | 9667.08 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-03-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:40:00 | 9735.00 | 9698.98 | 0.00 | ORB-long ORB[9641.55,9687.50] vol=1.7x ATR=20.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 09:50:00 | 9765.14 | 9723.68 | 0.00 | T1 1.5R @ 9765.14 |
| Target hit | 2024-03-28 14:55:00 | 9776.90 | 9783.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — SELL (started 2024-04-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 10:55:00 | 9874.95 | 9918.84 | 0.00 | ORB-short ORB[9882.00,9950.00] vol=2.3x ATR=22.48 |
| Stop hit — per-position SL triggered | 2024-04-02 11:05:00 | 9897.43 | 9915.87 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 9901.30 | 9983.20 | 0.00 | ORB-short ORB[10001.05,10078.90] vol=1.6x ATR=29.51 |
| Stop hit — per-position SL triggered | 2024-04-04 11:10:00 | 9930.81 | 9976.11 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-04-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:35:00 | 9924.20 | 9948.10 | 0.00 | ORB-short ORB[9925.05,10019.85] vol=1.5x ATR=25.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-05 10:55:00 | 9885.87 | 9920.69 | 0.00 | T1 1.5R @ 9885.87 |
| Target hit | 2024-04-05 15:20:00 | 9827.65 | 9877.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2024-04-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:00:00 | 9950.00 | 9906.46 | 0.00 | ORB-long ORB[9860.10,9917.90] vol=2.6x ATR=20.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 10:10:00 | 9980.32 | 9926.25 | 0.00 | T1 1.5R @ 9980.32 |
| Stop hit — per-position SL triggered | 2024-04-09 10:30:00 | 9950.00 | 9934.70 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-04-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-10 10:35:00 | 9810.05 | 9755.98 | 0.00 | ORB-long ORB[9725.00,9810.00] vol=1.7x ATR=21.36 |
| Stop hit — per-position SL triggered | 2024-04-10 12:25:00 | 9788.69 | 9785.90 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 9717.90 | 9750.49 | 0.00 | ORB-short ORB[9730.05,9789.95] vol=1.9x ATR=22.09 |
| Stop hit — per-position SL triggered | 2024-04-12 10:00:00 | 9739.99 | 9740.84 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:45:00 | 9450.00 | 9434.35 | 0.00 | ORB-long ORB[9387.10,9448.90] vol=1.8x ATR=21.75 |
| Stop hit — per-position SL triggered | 2024-04-16 09:50:00 | 9428.25 | 9434.52 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-19 11:15:00 | 9329.95 | 9291.79 | 0.00 | ORB-long ORB[9250.00,9325.00] vol=1.7x ATR=23.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 11:30:00 | 9364.73 | 9301.79 | 0.00 | T1 1.5R @ 9364.73 |
| Stop hit — per-position SL triggered | 2024-04-19 11:45:00 | 9329.95 | 9313.05 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 10:40:00 | 9535.10 | 9582.80 | 0.00 | ORB-short ORB[9540.00,9625.00] vol=1.9x ATR=15.49 |
| Stop hit — per-position SL triggered | 2024-04-23 10:50:00 | 9550.59 | 9581.83 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 09:35:00 | 9856.70 | 9815.35 | 0.00 | ORB-long ORB[9743.05,9849.00] vol=3.3x ATR=32.72 |
| Stop hit — per-position SL triggered | 2024-04-29 10:05:00 | 9823.98 | 9827.63 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-05-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 10:10:00 | 9801.85 | 9858.28 | 0.00 | ORB-short ORB[9834.05,9929.00] vol=2.5x ATR=30.38 |
| Stop hit — per-position SL triggered | 2024-05-06 10:20:00 | 9832.23 | 9850.84 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:35:00 | 9676.15 | 9715.70 | 0.00 | ORB-short ORB[9701.00,9824.95] vol=1.5x ATR=19.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:50:00 | 9647.43 | 9703.98 | 0.00 | T1 1.5R @ 9647.43 |
| Stop hit — per-position SL triggered | 2024-05-07 10:55:00 | 9676.15 | 9703.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 09:30:00 | 7610.95 | 2023-05-19 09:45:00 | 7627.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-02 09:30:00 | 7942.45 | 2023-06-02 09:40:00 | 7924.47 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-05 11:05:00 | 7906.85 | 2023-06-05 12:10:00 | 7895.42 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-06-08 09:35:00 | 8204.00 | 2023-06-08 09:50:00 | 8190.51 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-09 10:30:00 | 8198.00 | 2023-06-09 10:40:00 | 8181.25 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-13 10:50:00 | 8273.90 | 2023-06-13 10:55:00 | 8260.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-06-20 10:50:00 | 8180.40 | 2023-06-20 11:00:00 | 8193.66 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-28 10:45:00 | 8238.85 | 2023-06-28 10:55:00 | 8226.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-07-05 10:40:00 | 8403.20 | 2023-07-05 11:30:00 | 8416.58 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-07-06 10:45:00 | 8457.25 | 2023-07-06 11:10:00 | 8442.28 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-07-25 09:30:00 | 8369.75 | 2023-07-25 09:40:00 | 8347.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-31 11:00:00 | 8377.00 | 2023-07-31 11:05:00 | 8362.51 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-08-02 11:00:00 | 8239.00 | 2023-08-02 11:40:00 | 8214.49 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-08-02 11:00:00 | 8239.00 | 2023-08-02 11:45:00 | 8239.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-08 11:10:00 | 8166.45 | 2023-08-08 11:25:00 | 8178.51 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-08-10 10:30:00 | 8099.75 | 2023-08-10 10:55:00 | 8121.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-08-16 11:15:00 | 8048.75 | 2023-08-16 11:55:00 | 8068.17 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-08-16 11:15:00 | 8048.75 | 2023-08-16 15:20:00 | 8262.20 | TARGET_HIT | 0.50 | 2.65% |
| SELL | retest1 | 2023-08-17 11:15:00 | 8158.75 | 2023-08-17 11:35:00 | 8173.43 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-23 10:55:00 | 8180.00 | 2023-08-23 14:20:00 | 8190.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-08-24 09:40:00 | 8307.05 | 2023-08-24 09:50:00 | 8288.16 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-31 10:30:00 | 8340.90 | 2023-08-31 10:35:00 | 8325.70 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-09-08 10:55:00 | 8463.80 | 2023-09-08 12:15:00 | 8477.38 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-10-11 09:30:00 | 8262.00 | 2023-10-11 09:55:00 | 8246.24 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-16 10:10:00 | 8448.55 | 2023-10-16 10:20:00 | 8429.26 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-18 10:10:00 | 8299.35 | 2023-10-18 10:55:00 | 8283.63 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2023-10-18 10:10:00 | 8299.35 | 2023-10-18 11:05:00 | 8299.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-30 10:50:00 | 8327.30 | 2023-10-30 11:00:00 | 8305.36 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-07 11:15:00 | 8648.00 | 2023-11-07 12:00:00 | 8664.95 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-11-07 11:15:00 | 8648.00 | 2023-11-07 12:20:00 | 8648.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-09 10:35:00 | 8667.90 | 2023-11-09 11:15:00 | 8648.32 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2023-11-09 10:35:00 | 8667.90 | 2023-11-09 12:00:00 | 8667.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-10 11:05:00 | 8664.65 | 2023-11-10 12:10:00 | 8686.28 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-11-10 11:05:00 | 8664.65 | 2023-11-10 15:20:00 | 8718.20 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2023-11-15 09:30:00 | 8773.60 | 2023-11-15 09:35:00 | 8799.15 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-11-15 09:30:00 | 8773.60 | 2023-11-15 09:50:00 | 8773.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 10:10:00 | 8862.00 | 2023-11-17 10:15:00 | 8844.39 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-11-24 10:45:00 | 8609.55 | 2023-11-24 10:50:00 | 8590.32 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-11-24 10:45:00 | 8609.55 | 2023-11-24 12:05:00 | 8609.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-28 10:35:00 | 8677.25 | 2023-11-28 11:20:00 | 8701.32 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-28 10:35:00 | 8677.25 | 2023-11-28 15:20:00 | 8727.60 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2023-12-04 11:05:00 | 9169.95 | 2023-12-04 11:10:00 | 9200.24 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-12-04 11:05:00 | 9169.95 | 2023-12-04 15:20:00 | 9324.80 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2023-12-06 10:30:00 | 9255.45 | 2023-12-06 11:15:00 | 9230.44 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-12-06 10:30:00 | 9255.45 | 2023-12-06 14:15:00 | 9217.80 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2023-12-11 09:30:00 | 9508.70 | 2023-12-11 09:35:00 | 9539.79 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-12-11 09:30:00 | 9508.70 | 2023-12-11 09:40:00 | 9508.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-12 09:35:00 | 9766.65 | 2023-12-12 10:30:00 | 9804.87 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-12-12 09:35:00 | 9766.65 | 2023-12-12 13:15:00 | 9854.05 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2023-12-20 10:20:00 | 10105.20 | 2023-12-20 10:35:00 | 10087.31 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-02-07 11:05:00 | 10145.00 | 2024-02-07 11:20:00 | 10172.36 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-02-07 11:05:00 | 10145.00 | 2024-02-07 11:30:00 | 10145.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 11:00:00 | 10086.00 | 2024-02-08 11:30:00 | 10044.97 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-02-08 11:00:00 | 10086.00 | 2024-02-08 11:45:00 | 10086.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-14 09:30:00 | 9737.60 | 2024-02-14 09:35:00 | 9759.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-02-26 10:45:00 | 9928.10 | 2024-02-26 15:20:00 | 9934.00 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest1 | 2024-02-29 11:10:00 | 9833.45 | 2024-02-29 12:30:00 | 9810.68 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-03-01 10:50:00 | 10065.90 | 2024-03-01 11:35:00 | 10101.04 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-03-01 10:50:00 | 10065.90 | 2024-03-01 12:15:00 | 10065.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-04 09:30:00 | 10083.40 | 2024-03-04 09:45:00 | 10039.48 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-03-04 09:30:00 | 10083.40 | 2024-03-04 15:20:00 | 9968.45 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2024-03-06 10:00:00 | 9746.90 | 2024-03-06 10:25:00 | 9713.73 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-03-06 10:00:00 | 9746.90 | 2024-03-06 15:05:00 | 9651.50 | TARGET_HIT | 0.50 | 0.98% |
| BUY | retest1 | 2024-03-07 10:30:00 | 9677.50 | 2024-03-07 10:40:00 | 9655.66 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-12 10:45:00 | 9624.00 | 2024-03-12 11:15:00 | 9641.04 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-03-21 11:00:00 | 9607.20 | 2024-03-21 11:15:00 | 9634.31 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-03-21 11:00:00 | 9607.20 | 2024-03-21 11:35:00 | 9607.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-26 10:30:00 | 9596.80 | 2024-03-26 13:10:00 | 9615.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-03-27 10:50:00 | 9685.35 | 2024-03-27 11:50:00 | 9667.69 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-03-28 09:40:00 | 9735.00 | 2024-03-28 09:50:00 | 9765.14 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-03-28 09:40:00 | 9735.00 | 2024-03-28 14:55:00 | 9776.90 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-04-02 10:55:00 | 9874.95 | 2024-04-02 11:05:00 | 9897.43 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-04 10:50:00 | 9901.30 | 2024-04-04 11:10:00 | 9930.81 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-05 09:35:00 | 9924.20 | 2024-04-05 10:55:00 | 9885.87 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-04-05 09:35:00 | 9924.20 | 2024-04-05 15:20:00 | 9827.65 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-04-09 10:00:00 | 9950.00 | 2024-04-09 10:10:00 | 9980.32 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-09 10:00:00 | 9950.00 | 2024-04-09 10:30:00 | 9950.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-10 10:35:00 | 9810.05 | 2024-04-10 12:25:00 | 9788.69 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-04-12 09:40:00 | 9717.90 | 2024-04-12 10:00:00 | 9739.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-16 09:45:00 | 9450.00 | 2024-04-16 09:50:00 | 9428.25 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-04-19 11:15:00 | 9329.95 | 2024-04-19 11:30:00 | 9364.73 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-04-19 11:15:00 | 9329.95 | 2024-04-19 11:45:00 | 9329.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-23 10:40:00 | 9535.10 | 2024-04-23 10:50:00 | 9550.59 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-04-29 09:35:00 | 9856.70 | 2024-04-29 10:05:00 | 9823.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-06 10:10:00 | 9801.85 | 2024-05-06 10:20:00 | 9832.23 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-07 10:35:00 | 9676.15 | 2024-05-07 10:50:00 | 9647.43 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-07 10:35:00 | 9676.15 | 2024-05-07 10:55:00 | 9676.15 | STOP_HIT | 0.50 | 0.00% |
