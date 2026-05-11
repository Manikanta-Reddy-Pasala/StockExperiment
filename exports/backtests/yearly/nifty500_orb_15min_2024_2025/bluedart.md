# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 5695.00
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
| PARTIAL | 33 |
| TARGET_HIT | 15 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 120 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 73
- **Target hits / Stop hits / Partials:** 15 / 72 / 33
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 10.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 15 | 38.5% | 5 | 23 | 11 | 0.13% | 5.1% |
| BUY @ 2nd Alert (retest1) | 39 | 15 | 38.5% | 5 | 23 | 11 | 0.13% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 81 | 32 | 39.5% | 10 | 49 | 22 | 0.07% | 5.7% |
| SELL @ 2nd Alert (retest1) | 81 | 32 | 39.5% | 10 | 49 | 22 | 0.07% | 5.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 120 | 47 | 39.2% | 15 | 72 | 33 | 0.09% | 10.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 7160.70 | 7181.78 | 0.00 | ORB-short ORB[7178.55,7207.95] vol=2.0x ATR=19.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:45:00 | 7131.46 | 7172.05 | 0.00 | T1 1.5R @ 7131.46 |
| Stop hit — per-position SL triggered | 2024-05-23 10:10:00 | 7160.70 | 7160.08 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:05:00 | 7268.60 | 7265.43 | 0.00 | ORB-long ORB[7160.05,7251.20] vol=18.9x ATR=18.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:10:00 | 7295.63 | 7265.47 | 0.00 | T1 1.5R @ 7295.63 |
| Target hit | 2024-05-24 11:10:00 | 7255.10 | 7265.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 7209.80 | 7228.55 | 0.00 | ORB-short ORB[7230.60,7295.90] vol=3.2x ATR=21.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:55:00 | 7176.82 | 7211.17 | 0.00 | T1 1.5R @ 7176.82 |
| Target hit | 2024-05-28 11:45:00 | 7204.95 | 7200.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 7831.00 | 7874.93 | 0.00 | ORB-short ORB[7854.80,7924.95] vol=2.4x ATR=28.51 |
| Stop hit — per-position SL triggered | 2024-06-13 11:05:00 | 7859.51 | 7859.29 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:55:00 | 7960.00 | 7971.00 | 0.00 | ORB-short ORB[7974.45,8057.65] vol=2.7x ATR=24.90 |
| Stop hit — per-position SL triggered | 2024-06-18 11:40:00 | 7984.90 | 7970.81 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:10:00 | 7871.65 | 7923.12 | 0.00 | ORB-short ORB[7901.00,8000.70] vol=1.5x ATR=30.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:25:00 | 7826.42 | 7906.56 | 0.00 | T1 1.5R @ 7826.42 |
| Stop hit — per-position SL triggered | 2024-06-19 10:30:00 | 7871.65 | 7903.52 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 7861.45 | 7821.68 | 0.00 | ORB-long ORB[7714.00,7799.00] vol=5.6x ATR=47.91 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 7813.54 | 7823.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 09:30:00 | 7708.00 | 7738.35 | 0.00 | ORB-short ORB[7730.00,7785.85] vol=1.9x ATR=39.92 |
| Stop hit — per-position SL triggered | 2024-06-24 09:50:00 | 7747.92 | 7732.69 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:40:00 | 7720.50 | 7690.28 | 0.00 | ORB-long ORB[7657.00,7703.35] vol=3.7x ATR=19.95 |
| Stop hit — per-position SL triggered | 2024-06-26 09:45:00 | 7700.55 | 7691.50 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 8159.45 | 8178.81 | 0.00 | ORB-short ORB[8164.05,8248.95] vol=1.8x ATR=22.68 |
| Stop hit — per-position SL triggered | 2024-07-03 09:40:00 | 8182.13 | 8177.48 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:20:00 | 8090.00 | 8137.48 | 0.00 | ORB-short ORB[8154.05,8239.00] vol=1.8x ATR=23.38 |
| Stop hit — per-position SL triggered | 2024-07-04 11:25:00 | 8113.38 | 8117.01 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:10:00 | 8225.90 | 8197.19 | 0.00 | ORB-long ORB[8167.65,8219.95] vol=1.6x ATR=17.61 |
| Stop hit — per-position SL triggered | 2024-07-05 10:35:00 | 8208.29 | 8204.36 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:30:00 | 8350.00 | 8300.03 | 0.00 | ORB-long ORB[8196.20,8260.00] vol=8.1x ATR=27.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:35:00 | 8390.64 | 8376.46 | 0.00 | T1 1.5R @ 8390.64 |
| Stop hit — per-position SL triggered | 2024-07-08 09:40:00 | 8350.00 | 8375.13 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:30:00 | 8453.25 | 8509.94 | 0.00 | ORB-short ORB[8525.05,8600.05] vol=6.3x ATR=28.94 |
| Stop hit — per-position SL triggered | 2024-07-12 10:40:00 | 8482.19 | 8509.25 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 8340.00 | 8378.35 | 0.00 | ORB-short ORB[8360.00,8467.00] vol=1.9x ATR=24.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:40:00 | 8302.88 | 8364.88 | 0.00 | T1 1.5R @ 8302.88 |
| Stop hit — per-position SL triggered | 2024-07-15 09:45:00 | 8340.00 | 8364.43 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 8577.40 | 8539.80 | 0.00 | ORB-long ORB[8495.05,8560.95] vol=2.3x ATR=25.24 |
| Stop hit — per-position SL triggered | 2024-07-16 09:35:00 | 8552.16 | 8542.34 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 7723.50 | 7785.64 | 0.00 | ORB-short ORB[7734.60,7843.10] vol=2.7x ATR=38.96 |
| Stop hit — per-position SL triggered | 2024-07-23 10:15:00 | 7762.46 | 7754.53 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:10:00 | 7973.35 | 7918.83 | 0.00 | ORB-long ORB[7768.65,7848.00] vol=3.2x ATR=22.86 |
| Stop hit — per-position SL triggered | 2024-07-26 11:30:00 | 7950.49 | 7921.29 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:35:00 | 7890.90 | 7943.87 | 0.00 | ORB-short ORB[7899.40,7981.95] vol=2.2x ATR=34.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 13:30:00 | 7839.63 | 7905.99 | 0.00 | T1 1.5R @ 7839.63 |
| Target hit | 2024-07-29 15:20:00 | 7714.75 | 7861.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 7975.60 | 7910.17 | 0.00 | ORB-long ORB[7846.65,7914.00] vol=6.7x ATR=30.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:35:00 | 8021.73 | 7944.19 | 0.00 | T1 1.5R @ 8021.73 |
| Target hit | 2024-07-31 15:00:00 | 8159.65 | 8165.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2024-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:25:00 | 8035.00 | 8073.47 | 0.00 | ORB-short ORB[8045.00,8157.75] vol=1.6x ATR=19.37 |
| Stop hit — per-position SL triggered | 2024-08-09 10:50:00 | 8054.37 | 8071.43 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:10:00 | 7981.70 | 7947.65 | 0.00 | ORB-long ORB[7875.25,7951.70] vol=3.6x ATR=40.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:55:00 | 8041.76 | 7968.43 | 0.00 | T1 1.5R @ 8041.76 |
| Target hit | 2024-08-12 15:20:00 | 8169.30 | 8074.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2024-08-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:40:00 | 8079.95 | 8085.10 | 0.00 | ORB-short ORB[8101.00,8200.00] vol=2.8x ATR=23.84 |
| Stop hit — per-position SL triggered | 2024-08-13 10:45:00 | 8103.79 | 8086.35 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:50:00 | 7987.35 | 7927.58 | 0.00 | ORB-long ORB[7872.00,7949.60] vol=4.2x ATR=24.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:55:00 | 8023.78 | 7938.33 | 0.00 | T1 1.5R @ 8023.78 |
| Target hit | 2024-08-16 10:40:00 | 8049.85 | 8057.22 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2024-08-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:10:00 | 7983.20 | 8026.11 | 0.00 | ORB-short ORB[8032.05,8094.90] vol=1.5x ATR=18.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:40:00 | 7955.02 | 7998.59 | 0.00 | T1 1.5R @ 7955.02 |
| Target hit | 2024-08-19 15:05:00 | 7949.00 | 7940.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2024-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:50:00 | 7932.95 | 7941.35 | 0.00 | ORB-short ORB[7934.80,7982.40] vol=4.3x ATR=17.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:55:00 | 7906.45 | 7938.79 | 0.00 | T1 1.5R @ 7906.45 |
| Stop hit — per-position SL triggered | 2024-08-20 10:00:00 | 7932.95 | 7937.65 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:25:00 | 7960.00 | 7985.37 | 0.00 | ORB-short ORB[7970.25,8050.00] vol=1.8x ATR=19.43 |
| Stop hit — per-position SL triggered | 2024-08-21 10:45:00 | 7979.43 | 7985.17 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:00:00 | 8180.00 | 8122.44 | 0.00 | ORB-long ORB[8028.85,8087.00] vol=9.9x ATR=24.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:05:00 | 8216.70 | 8133.88 | 0.00 | T1 1.5R @ 8216.70 |
| Stop hit — per-position SL triggered | 2024-08-22 10:10:00 | 8180.00 | 8137.56 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 8244.85 | 8266.62 | 0.00 | ORB-short ORB[8271.00,8350.00] vol=4.3x ATR=37.55 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 8282.40 | 8267.20 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:50:00 | 8167.50 | 8220.19 | 0.00 | ORB-short ORB[8204.30,8316.95] vol=2.1x ATR=20.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:55:00 | 8136.94 | 8208.65 | 0.00 | T1 1.5R @ 8136.94 |
| Stop hit — per-position SL triggered | 2024-08-29 11:00:00 | 8167.50 | 8206.48 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:50:00 | 8066.55 | 8090.72 | 0.00 | ORB-short ORB[8140.50,8187.50] vol=4.1x ATR=24.70 |
| Stop hit — per-position SL triggered | 2024-08-30 10:05:00 | 8091.25 | 8089.72 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:00:00 | 8210.00 | 8168.03 | 0.00 | ORB-long ORB[8061.40,8170.00] vol=4.6x ATR=31.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:55:00 | 8257.05 | 8202.27 | 0.00 | T1 1.5R @ 8257.05 |
| Target hit | 2024-09-03 14:45:00 | 8252.10 | 8254.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 8114.05 | 8183.67 | 0.00 | ORB-short ORB[8144.20,8235.00] vol=1.5x ATR=31.28 |
| Stop hit — per-position SL triggered | 2024-09-06 10:25:00 | 8145.33 | 8163.02 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:30:00 | 8187.15 | 8183.32 | 0.00 | ORB-long ORB[8149.80,8185.95] vol=5.1x ATR=19.98 |
| Stop hit — per-position SL triggered | 2024-09-11 10:50:00 | 8167.17 | 8182.43 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:55:00 | 8160.70 | 8191.84 | 0.00 | ORB-short ORB[8182.40,8243.80] vol=5.5x ATR=19.13 |
| Stop hit — per-position SL triggered | 2024-09-13 14:00:00 | 8179.83 | 8176.68 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 8055.90 | 8126.73 | 0.00 | ORB-short ORB[8118.55,8209.45] vol=2.3x ATR=28.57 |
| Stop hit — per-position SL triggered | 2024-09-25 10:10:00 | 8084.47 | 8125.03 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:55:00 | 8080.00 | 8102.90 | 0.00 | ORB-short ORB[8092.00,8174.35] vol=3.0x ATR=21.99 |
| Stop hit — per-position SL triggered | 2024-09-26 12:25:00 | 8101.99 | 8096.75 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 09:50:00 | 8158.05 | 8119.88 | 0.00 | ORB-long ORB[8089.25,8138.00] vol=1.7x ATR=24.55 |
| Stop hit — per-position SL triggered | 2024-09-30 10:00:00 | 8133.50 | 8125.39 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 09:40:00 | 8685.90 | 8567.12 | 0.00 | ORB-long ORB[8490.00,8600.00] vol=2.0x ATR=44.67 |
| Stop hit — per-position SL triggered | 2024-10-04 09:50:00 | 8641.23 | 8607.70 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:00:00 | 8725.00 | 8608.54 | 0.00 | ORB-long ORB[8484.05,8574.90] vol=1.7x ATR=43.04 |
| Stop hit — per-position SL triggered | 2024-10-08 11:25:00 | 8681.96 | 8615.33 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:15:00 | 8597.65 | 8638.36 | 0.00 | ORB-short ORB[8606.05,8702.00] vol=3.6x ATR=24.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 12:20:00 | 8560.18 | 8627.87 | 0.00 | T1 1.5R @ 8560.18 |
| Target hit | 2024-10-10 15:20:00 | 8547.70 | 8580.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-10-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:05:00 | 8457.10 | 8503.10 | 0.00 | ORB-short ORB[8490.00,8575.00] vol=6.5x ATR=22.07 |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 8479.17 | 8499.01 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 8554.45 | 8480.07 | 0.00 | ORB-long ORB[8419.00,8500.00] vol=4.1x ATR=31.32 |
| Stop hit — per-position SL triggered | 2024-10-14 09:40:00 | 8523.13 | 8488.91 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:50:00 | 8492.45 | 8516.39 | 0.00 | ORB-short ORB[8513.60,8556.15] vol=2.4x ATR=20.80 |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 8513.25 | 8516.86 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 8502.05 | 8525.84 | 0.00 | ORB-short ORB[8510.00,8595.75] vol=1.6x ATR=20.01 |
| Stop hit — per-position SL triggered | 2024-10-16 10:25:00 | 8522.06 | 8524.38 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:25:00 | 7633.20 | 7670.55 | 0.00 | ORB-short ORB[7665.80,7744.70] vol=1.7x ATR=26.68 |
| Stop hit — per-position SL triggered | 2024-10-29 11:30:00 | 7659.88 | 7657.05 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-11-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:30:00 | 7903.00 | 7939.63 | 0.00 | ORB-short ORB[7935.05,8001.35] vol=5.2x ATR=24.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:40:00 | 7866.89 | 7916.58 | 0.00 | T1 1.5R @ 7866.89 |
| Stop hit — per-position SL triggered | 2024-11-05 09:45:00 | 7903.00 | 7910.73 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:45:00 | 7866.00 | 7939.83 | 0.00 | ORB-short ORB[7915.70,7993.65] vol=2.9x ATR=31.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:50:00 | 7818.89 | 7912.97 | 0.00 | T1 1.5R @ 7818.89 |
| Stop hit — per-position SL triggered | 2024-11-06 12:20:00 | 7866.00 | 7894.75 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 09:35:00 | 7809.35 | 7872.87 | 0.00 | ORB-short ORB[7850.00,7952.60] vol=2.5x ATR=25.05 |
| Stop hit — per-position SL triggered | 2024-11-08 09:40:00 | 7834.40 | 7877.22 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-11-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:35:00 | 7923.85 | 7907.92 | 0.00 | ORB-long ORB[7841.80,7920.65] vol=7.5x ATR=41.93 |
| Stop hit — per-position SL triggered | 2024-11-11 09:40:00 | 7881.92 | 7906.59 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:35:00 | 7812.75 | 7837.31 | 0.00 | ORB-short ORB[7825.30,7927.30] vol=10.5x ATR=37.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:40:00 | 7755.76 | 7828.96 | 0.00 | T1 1.5R @ 7755.76 |
| Target hit | 2024-11-12 15:20:00 | 7655.00 | 7775.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2024-11-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 09:40:00 | 7369.65 | 7429.16 | 0.00 | ORB-short ORB[7435.10,7539.95] vol=4.3x ATR=31.75 |
| Stop hit — per-position SL triggered | 2024-11-14 09:55:00 | 7401.40 | 7416.37 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-11-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 11:00:00 | 7369.80 | 7390.28 | 0.00 | ORB-short ORB[7390.00,7464.95] vol=1.7x ATR=21.10 |
| Stop hit — per-position SL triggered | 2024-11-19 12:35:00 | 7390.90 | 7388.94 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:45:00 | 7499.95 | 7522.02 | 0.00 | ORB-short ORB[7500.00,7553.10] vol=2.3x ATR=17.77 |
| Stop hit — per-position SL triggered | 2024-11-27 11:35:00 | 7517.72 | 7529.63 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-11-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:45:00 | 7547.30 | 7522.65 | 0.00 | ORB-long ORB[7460.00,7514.00] vol=7.3x ATR=27.65 |
| Stop hit — per-position SL triggered | 2024-11-29 10:10:00 | 7519.65 | 7525.14 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 10:40:00 | 7448.00 | 7508.91 | 0.00 | ORB-short ORB[7470.05,7570.00] vol=3.0x ATR=22.57 |
| Stop hit — per-position SL triggered | 2024-12-02 10:50:00 | 7470.57 | 7481.12 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:55:00 | 7511.10 | 7484.46 | 0.00 | ORB-long ORB[7440.45,7493.15] vol=1.6x ATR=21.34 |
| Stop hit — per-position SL triggered | 2024-12-04 10:20:00 | 7489.76 | 7493.06 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 7789.10 | 7852.33 | 0.00 | ORB-short ORB[7853.95,7905.00] vol=2.3x ATR=30.42 |
| Stop hit — per-position SL triggered | 2024-12-16 09:40:00 | 7819.52 | 7839.12 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:45:00 | 7734.35 | 7770.88 | 0.00 | ORB-short ORB[7749.65,7847.25] vol=1.6x ATR=23.67 |
| Stop hit — per-position SL triggered | 2024-12-17 09:55:00 | 7758.02 | 7770.48 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:05:00 | 7674.50 | 7706.71 | 0.00 | ORB-short ORB[7705.00,7756.20] vol=2.4x ATR=21.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:20:00 | 7642.95 | 7697.52 | 0.00 | T1 1.5R @ 7642.95 |
| Stop hit — per-position SL triggered | 2024-12-18 11:50:00 | 7674.50 | 7673.93 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 6943.60 | 6959.79 | 0.00 | ORB-short ORB[6944.00,7028.00] vol=2.0x ATR=17.26 |
| Stop hit — per-position SL triggered | 2025-01-02 14:25:00 | 6960.86 | 6948.65 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:30:00 | 6669.00 | 6694.38 | 0.00 | ORB-short ORB[6695.00,6741.00] vol=3.9x ATR=17.64 |
| Stop hit — per-position SL triggered | 2025-01-08 09:35:00 | 6686.64 | 6696.50 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 6475.30 | 6516.95 | 0.00 | ORB-short ORB[6495.00,6579.65] vol=1.9x ATR=27.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:45:00 | 6433.34 | 6504.10 | 0.00 | T1 1.5R @ 6433.34 |
| Stop hit — per-position SL triggered | 2025-01-10 10:10:00 | 6475.30 | 6491.67 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 6205.00 | 6252.14 | 0.00 | ORB-short ORB[6232.60,6320.05] vol=2.4x ATR=29.23 |
| Stop hit — per-position SL triggered | 2025-01-15 09:35:00 | 6234.23 | 6250.17 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:55:00 | 6508.90 | 6480.58 | 0.00 | ORB-long ORB[6434.00,6488.95] vol=1.9x ATR=19.94 |
| Stop hit — per-position SL triggered | 2025-01-20 10:05:00 | 6488.96 | 6484.14 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:05:00 | 6475.55 | 6549.90 | 0.00 | ORB-short ORB[6573.40,6649.00] vol=1.8x ATR=21.03 |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 6496.58 | 6545.94 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 10:40:00 | 6524.20 | 6562.21 | 0.00 | ORB-short ORB[6555.60,6639.10] vol=6.2x ATR=18.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:50:00 | 6496.97 | 6550.04 | 0.00 | T1 1.5R @ 6496.97 |
| Stop hit — per-position SL triggered | 2025-01-23 11:00:00 | 6524.20 | 6547.62 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:35:00 | 6545.95 | 6565.69 | 0.00 | ORB-short ORB[6564.75,6651.85] vol=2.9x ATR=21.61 |
| Stop hit — per-position SL triggered | 2025-01-24 09:40:00 | 6567.56 | 6569.28 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:50:00 | 6378.30 | 6416.95 | 0.00 | ORB-short ORB[6410.40,6499.00] vol=2.7x ATR=25.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:00:00 | 6339.45 | 6388.63 | 0.00 | T1 1.5R @ 6339.45 |
| Target hit | 2025-01-27 11:10:00 | 6345.85 | 6332.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 70 — BUY (started 2025-01-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:40:00 | 6403.10 | 6392.02 | 0.00 | ORB-long ORB[6349.50,6380.85] vol=1.7x ATR=18.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:00:00 | 6431.45 | 6396.79 | 0.00 | T1 1.5R @ 6431.45 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 6403.10 | 6397.75 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:15:00 | 6594.75 | 6610.33 | 0.00 | ORB-short ORB[6599.95,6645.80] vol=4.5x ATR=11.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:40:00 | 6576.94 | 6608.02 | 0.00 | T1 1.5R @ 6576.94 |
| Target hit | 2025-02-01 15:10:00 | 6573.05 | 6558.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2025-02-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:25:00 | 6549.30 | 6503.00 | 0.00 | ORB-long ORB[6455.05,6520.00] vol=1.5x ATR=15.90 |
| Stop hit — per-position SL triggered | 2025-02-06 10:35:00 | 6533.40 | 6507.80 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 11:10:00 | 6660.40 | 6733.53 | 0.00 | ORB-short ORB[6750.00,6845.00] vol=1.8x ATR=27.54 |
| Stop hit — per-position SL triggered | 2025-02-10 11:55:00 | 6687.94 | 6726.39 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-02-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:50:00 | 5950.35 | 6008.40 | 0.00 | ORB-short ORB[6002.85,6092.20] vol=4.5x ATR=20.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:00:00 | 5919.35 | 5996.94 | 0.00 | T1 1.5R @ 5919.35 |
| Stop hit — per-position SL triggered | 2025-02-14 11:10:00 | 5950.35 | 5993.77 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:30:00 | 5901.50 | 5941.56 | 0.00 | ORB-short ORB[5936.20,6009.20] vol=1.6x ATR=27.92 |
| Stop hit — per-position SL triggered | 2025-02-18 09:35:00 | 5929.42 | 5939.60 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:35:00 | 6457.10 | 6448.26 | 0.00 | ORB-long ORB[6369.00,6419.60] vol=2.7x ATR=30.80 |
| Stop hit — per-position SL triggered | 2025-02-21 09:40:00 | 6426.30 | 6430.70 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 10:10:00 | 6109.00 | 6135.99 | 0.00 | ORB-short ORB[6117.25,6180.80] vol=3.1x ATR=18.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:30:00 | 6081.23 | 6118.35 | 0.00 | T1 1.5R @ 6081.23 |
| Stop hit — per-position SL triggered | 2025-02-27 11:55:00 | 6109.00 | 6116.52 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:05:00 | 5667.70 | 5694.40 | 0.00 | ORB-short ORB[5690.00,5750.00] vol=2.1x ATR=13.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 5646.88 | 5685.33 | 0.00 | T1 1.5R @ 5646.88 |
| Target hit | 2025-03-12 14:50:00 | 5657.40 | 5650.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 79 — BUY (started 2025-03-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:45:00 | 5825.90 | 5802.48 | 0.00 | ORB-long ORB[5725.00,5773.80] vol=14.2x ATR=18.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:55:00 | 5853.48 | 5808.62 | 0.00 | T1 1.5R @ 5853.48 |
| Stop hit — per-position SL triggered | 2025-03-19 11:00:00 | 5825.90 | 5809.28 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 11:10:00 | 5892.55 | 5909.27 | 0.00 | ORB-short ORB[5897.65,5957.60] vol=2.5x ATR=11.05 |
| Stop hit — per-position SL triggered | 2025-03-24 11:30:00 | 5903.60 | 5908.45 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:55:00 | 6137.15 | 6069.62 | 0.00 | ORB-long ORB[5999.90,6083.95] vol=1.7x ATR=31.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 10:05:00 | 6184.02 | 6137.61 | 0.00 | T1 1.5R @ 6184.02 |
| Stop hit — per-position SL triggered | 2025-03-25 10:10:00 | 6137.15 | 6139.99 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-03-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 10:55:00 | 6202.00 | 6267.66 | 0.00 | ORB-short ORB[6233.30,6303.35] vol=3.9x ATR=25.28 |
| Stop hit — per-position SL triggered | 2025-03-28 11:00:00 | 6227.28 | 6265.75 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2025-04-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:35:00 | 6161.65 | 6190.95 | 0.00 | ORB-short ORB[6188.05,6244.00] vol=2.9x ATR=20.46 |
| Stop hit — per-position SL triggered | 2025-04-03 09:40:00 | 6182.11 | 6189.97 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:40:00 | 6609.50 | 6565.45 | 0.00 | ORB-long ORB[6511.50,6595.00] vol=1.7x ATR=20.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:45:00 | 6640.05 | 6584.11 | 0.00 | T1 1.5R @ 6640.05 |
| Stop hit — per-position SL triggered | 2025-04-23 09:50:00 | 6609.50 | 6585.65 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 6433.50 | 6479.10 | 0.00 | ORB-short ORB[6460.50,6534.00] vol=2.8x ATR=27.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:55:00 | 6392.65 | 6447.36 | 0.00 | T1 1.5R @ 6392.65 |
| Target hit | 2025-04-25 12:55:00 | 6397.00 | 6386.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 86 — SELL (started 2025-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:50:00 | 6450.50 | 6484.21 | 0.00 | ORB-short ORB[6480.50,6529.00] vol=1.8x ATR=18.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:05:00 | 6422.63 | 6470.20 | 0.00 | T1 1.5R @ 6422.63 |
| Target hit | 2025-04-29 13:35:00 | 6423.00 | 6374.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 87 — BUY (started 2025-05-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 09:40:00 | 6666.00 | 6587.11 | 0.00 | ORB-long ORB[6547.50,6625.50] vol=2.6x ATR=48.95 |
| Stop hit — per-position SL triggered | 2025-05-09 09:45:00 | 6617.05 | 6589.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-23 09:30:00 | 7160.70 | 2024-05-23 09:45:00 | 7131.46 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-23 09:30:00 | 7160.70 | 2024-05-23 10:10:00 | 7160.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-24 11:05:00 | 7268.60 | 2024-05-24 11:10:00 | 7295.63 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-05-24 11:05:00 | 7268.60 | 2024-05-24 11:10:00 | 7255.10 | TARGET_HIT | 0.50 | -0.19% |
| SELL | retest1 | 2024-05-28 09:35:00 | 7209.80 | 2024-05-28 09:55:00 | 7176.82 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-28 09:35:00 | 7209.80 | 2024-05-28 11:45:00 | 7204.95 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2024-06-13 09:40:00 | 7831.00 | 2024-06-13 11:05:00 | 7859.51 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-18 10:55:00 | 7960.00 | 2024-06-18 11:40:00 | 7984.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-19 10:10:00 | 7871.65 | 2024-06-19 10:25:00 | 7826.42 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-19 10:10:00 | 7871.65 | 2024-06-19 10:30:00 | 7871.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 09:35:00 | 7861.45 | 2024-06-21 09:40:00 | 7813.54 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2024-06-24 09:30:00 | 7708.00 | 2024-06-24 09:50:00 | 7747.92 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-06-26 09:40:00 | 7720.50 | 2024-06-26 09:45:00 | 7700.55 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-03 09:30:00 | 8159.45 | 2024-07-03 09:40:00 | 8182.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-04 10:20:00 | 8090.00 | 2024-07-04 11:25:00 | 8113.38 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-05 10:10:00 | 8225.90 | 2024-07-05 10:35:00 | 8208.29 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-08 09:30:00 | 8350.00 | 2024-07-08 09:35:00 | 8390.64 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-07-08 09:30:00 | 8350.00 | 2024-07-08 09:40:00 | 8350.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:30:00 | 8453.25 | 2024-07-12 10:40:00 | 8482.19 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-15 09:30:00 | 8340.00 | 2024-07-15 09:40:00 | 8302.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-15 09:30:00 | 8340.00 | 2024-07-15 09:45:00 | 8340.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 09:30:00 | 8577.40 | 2024-07-16 09:35:00 | 8552.16 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-23 09:30:00 | 7723.50 | 2024-07-23 10:15:00 | 7762.46 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-07-26 11:10:00 | 7973.35 | 2024-07-26 11:30:00 | 7950.49 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-29 09:35:00 | 7890.90 | 2024-07-29 13:30:00 | 7839.63 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-07-29 09:35:00 | 7890.90 | 2024-07-29 15:20:00 | 7714.75 | TARGET_HIT | 0.50 | 2.23% |
| BUY | retest1 | 2024-07-31 10:00:00 | 7975.60 | 2024-07-31 10:35:00 | 8021.73 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-31 10:00:00 | 7975.60 | 2024-07-31 15:00:00 | 8159.65 | TARGET_HIT | 0.50 | 2.31% |
| SELL | retest1 | 2024-08-09 10:25:00 | 8035.00 | 2024-08-09 10:50:00 | 8054.37 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-12 10:10:00 | 7981.70 | 2024-08-12 10:55:00 | 8041.76 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-08-12 10:10:00 | 7981.70 | 2024-08-12 15:20:00 | 8169.30 | TARGET_HIT | 0.50 | 2.35% |
| SELL | retest1 | 2024-08-13 10:40:00 | 8079.95 | 2024-08-13 10:45:00 | 8103.79 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-16 09:50:00 | 7987.35 | 2024-08-16 09:55:00 | 8023.78 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-16 09:50:00 | 7987.35 | 2024-08-16 10:40:00 | 8049.85 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2024-08-19 10:10:00 | 7983.20 | 2024-08-19 10:40:00 | 7955.02 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-08-19 10:10:00 | 7983.20 | 2024-08-19 15:05:00 | 7949.00 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-20 09:50:00 | 7932.95 | 2024-08-20 09:55:00 | 7906.45 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-08-20 09:50:00 | 7932.95 | 2024-08-20 10:00:00 | 7932.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-21 10:25:00 | 7960.00 | 2024-08-21 10:45:00 | 7979.43 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-22 10:00:00 | 8180.00 | 2024-08-22 10:05:00 | 8216.70 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-22 10:00:00 | 8180.00 | 2024-08-22 10:10:00 | 8180.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 8244.85 | 2024-08-28 09:35:00 | 8282.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-29 10:50:00 | 8167.50 | 2024-08-29 10:55:00 | 8136.94 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-29 10:50:00 | 8167.50 | 2024-08-29 11:00:00 | 8167.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 09:50:00 | 8066.55 | 2024-08-30 10:05:00 | 8091.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-03 10:00:00 | 8210.00 | 2024-09-03 10:55:00 | 8257.05 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-03 10:00:00 | 8210.00 | 2024-09-03 14:45:00 | 8252.10 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-06 09:50:00 | 8114.05 | 2024-09-06 10:25:00 | 8145.33 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-11 10:30:00 | 8187.15 | 2024-09-11 10:50:00 | 8167.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-13 10:55:00 | 8160.70 | 2024-09-13 14:00:00 | 8179.83 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-09-25 10:05:00 | 8055.90 | 2024-09-25 10:10:00 | 8084.47 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-26 09:55:00 | 8080.00 | 2024-09-26 12:25:00 | 8101.99 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-30 09:50:00 | 8158.05 | 2024-09-30 10:00:00 | 8133.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-04 09:40:00 | 8685.90 | 2024-10-04 09:50:00 | 8641.23 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-10-08 11:00:00 | 8725.00 | 2024-10-08 11:25:00 | 8681.96 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-10-10 11:15:00 | 8597.65 | 2024-10-10 12:20:00 | 8560.18 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-10 11:15:00 | 8597.65 | 2024-10-10 15:20:00 | 8547.70 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-11 11:05:00 | 8457.10 | 2024-10-11 11:15:00 | 8479.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-14 09:30:00 | 8554.45 | 2024-10-14 09:40:00 | 8523.13 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-15 09:50:00 | 8492.45 | 2024-10-15 10:15:00 | 8513.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-16 10:05:00 | 8502.05 | 2024-10-16 10:25:00 | 8522.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-29 10:25:00 | 7633.20 | 2024-10-29 11:30:00 | 7659.88 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-05 09:30:00 | 7903.00 | 2024-11-05 09:40:00 | 7866.89 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-05 09:30:00 | 7903.00 | 2024-11-05 09:45:00 | 7903.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-06 10:45:00 | 7866.00 | 2024-11-06 10:50:00 | 7818.89 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-06 10:45:00 | 7866.00 | 2024-11-06 12:20:00 | 7866.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-08 09:35:00 | 7809.35 | 2024-11-08 09:40:00 | 7834.40 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-11 09:35:00 | 7923.85 | 2024-11-11 09:40:00 | 7881.92 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-11-12 09:35:00 | 7812.75 | 2024-11-12 09:40:00 | 7755.76 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-11-12 09:35:00 | 7812.75 | 2024-11-12 15:20:00 | 7655.00 | TARGET_HIT | 0.50 | 2.02% |
| SELL | retest1 | 2024-11-14 09:40:00 | 7369.65 | 2024-11-14 09:55:00 | 7401.40 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-11-19 11:00:00 | 7369.80 | 2024-11-19 12:35:00 | 7390.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-27 10:45:00 | 7499.95 | 2024-11-27 11:35:00 | 7517.72 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-29 09:45:00 | 7547.30 | 2024-11-29 10:10:00 | 7519.65 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-02 10:40:00 | 7448.00 | 2024-12-02 10:50:00 | 7470.57 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-04 09:55:00 | 7511.10 | 2024-12-04 10:20:00 | 7489.76 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-16 09:30:00 | 7789.10 | 2024-12-16 09:40:00 | 7819.52 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-17 09:45:00 | 7734.35 | 2024-12-17 09:55:00 | 7758.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-18 10:05:00 | 7674.50 | 2024-12-18 10:20:00 | 7642.95 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-18 10:05:00 | 7674.50 | 2024-12-18 11:50:00 | 7674.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 11:05:00 | 6943.60 | 2025-01-02 14:25:00 | 6960.86 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-08 09:30:00 | 6669.00 | 2025-01-08 09:35:00 | 6686.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-10 09:35:00 | 6475.30 | 2025-01-10 09:45:00 | 6433.34 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-01-10 09:35:00 | 6475.30 | 2025-01-10 10:10:00 | 6475.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-15 09:30:00 | 6205.00 | 2025-01-15 09:35:00 | 6234.23 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-20 09:55:00 | 6508.90 | 2025-01-20 10:05:00 | 6488.96 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-21 11:05:00 | 6475.55 | 2025-01-21 11:15:00 | 6496.58 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-23 10:40:00 | 6524.20 | 2025-01-23 10:50:00 | 6496.97 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-23 10:40:00 | 6524.20 | 2025-01-23 11:00:00 | 6524.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:35:00 | 6545.95 | 2025-01-24 09:40:00 | 6567.56 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-27 09:50:00 | 6378.30 | 2025-01-27 10:00:00 | 6339.45 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-27 09:50:00 | 6378.30 | 2025-01-27 11:10:00 | 6345.85 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2025-01-29 10:40:00 | 6403.10 | 2025-01-29 11:00:00 | 6431.45 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-29 10:40:00 | 6403.10 | 2025-01-29 11:20:00 | 6403.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-01 11:15:00 | 6594.75 | 2025-02-01 11:40:00 | 6576.94 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-02-01 11:15:00 | 6594.75 | 2025-02-01 15:10:00 | 6573.05 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2025-02-06 10:25:00 | 6549.30 | 2025-02-06 10:35:00 | 6533.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-02-10 11:10:00 | 6660.40 | 2025-02-10 11:55:00 | 6687.94 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-02-14 10:50:00 | 5950.35 | 2025-02-14 11:00:00 | 5919.35 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-14 10:50:00 | 5950.35 | 2025-02-14 11:10:00 | 5950.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-18 09:30:00 | 5901.50 | 2025-02-18 09:35:00 | 5929.42 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-02-21 09:35:00 | 6457.10 | 2025-02-21 09:40:00 | 6426.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-27 10:10:00 | 6109.00 | 2025-02-27 11:30:00 | 6081.23 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-27 10:10:00 | 6109.00 | 2025-02-27 11:55:00 | 6109.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 11:05:00 | 5667.70 | 2025-03-12 11:25:00 | 5646.88 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-03-12 11:05:00 | 5667.70 | 2025-03-12 14:50:00 | 5657.40 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-03-19 10:45:00 | 5825.90 | 2025-03-19 10:55:00 | 5853.48 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-03-19 10:45:00 | 5825.90 | 2025-03-19 11:00:00 | 5825.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-24 11:10:00 | 5892.55 | 2025-03-24 11:30:00 | 5903.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-03-25 09:55:00 | 6137.15 | 2025-03-25 10:05:00 | 6184.02 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-03-25 09:55:00 | 6137.15 | 2025-03-25 10:10:00 | 6137.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-28 10:55:00 | 6202.00 | 2025-03-28 11:00:00 | 6227.28 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-03 09:35:00 | 6161.65 | 2025-04-03 09:40:00 | 6182.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-23 09:40:00 | 6609.50 | 2025-04-23 09:45:00 | 6640.05 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-23 09:40:00 | 6609.50 | 2025-04-23 09:50:00 | 6609.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:30:00 | 6433.50 | 2025-04-25 09:55:00 | 6392.65 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-04-25 09:30:00 | 6433.50 | 2025-04-25 12:55:00 | 6397.00 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-29 09:50:00 | 6450.50 | 2025-04-29 10:05:00 | 6422.63 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-04-29 09:50:00 | 6450.50 | 2025-04-29 13:35:00 | 6423.00 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-09 09:40:00 | 6666.00 | 2025-05-09 09:45:00 | 6617.05 | STOP_HIT | 1.00 | -0.73% |
