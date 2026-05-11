# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-02-04 15:25:00 (13833 bars)
- **Last close:** 6360.00
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 58
- **Target hits / Stop hits / Partials:** 9 / 58 / 23
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 9.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 15 | 31.9% | 3 | 32 | 12 | 0.16% | 7.7% |
| BUY @ 2nd Alert (retest1) | 47 | 15 | 31.9% | 3 | 32 | 12 | 0.16% | 7.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 17 | 39.5% | 6 | 26 | 11 | 0.03% | 1.4% |
| SELL @ 2nd Alert (retest1) | 43 | 17 | 39.5% | 6 | 26 | 11 | 0.03% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 32 | 35.6% | 9 | 58 | 23 | 0.10% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:05:00 | 7083.15 | 7099.96 | 0.00 | ORB-short ORB[7090.00,7119.55] vol=1.6x ATR=10.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:15:00 | 7067.44 | 7095.13 | 0.00 | T1 1.5R @ 7067.44 |
| Stop hit — per-position SL triggered | 2024-05-15 10:25:00 | 7083.15 | 7093.98 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 7208.15 | 7264.68 | 0.00 | ORB-short ORB[7224.55,7303.60] vol=1.6x ATR=21.28 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 7229.43 | 7261.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 7214.70 | 7285.55 | 0.00 | ORB-short ORB[7275.00,7343.25] vol=2.7x ATR=22.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:45:00 | 7181.57 | 7208.58 | 0.00 | T1 1.5R @ 7181.57 |
| Target hit | 2024-05-30 15:00:00 | 7207.45 | 7204.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-05-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:05:00 | 7131.70 | 7168.61 | 0.00 | ORB-short ORB[7181.00,7244.00] vol=1.5x ATR=16.90 |
| Stop hit — per-position SL triggered | 2024-05-31 11:10:00 | 7148.60 | 7167.40 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 7212.10 | 7176.45 | 0.00 | ORB-long ORB[7140.00,7190.00] vol=3.1x ATR=19.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:35:00 | 7241.65 | 7207.48 | 0.00 | T1 1.5R @ 7241.65 |
| Target hit | 2024-06-13 10:15:00 | 7229.50 | 7235.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-06-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:20:00 | 7346.25 | 7280.06 | 0.00 | ORB-long ORB[7206.95,7281.00] vol=7.5x ATR=25.39 |
| Stop hit — per-position SL triggered | 2024-06-14 10:25:00 | 7320.86 | 7293.33 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:30:00 | 7238.55 | 7266.54 | 0.00 | ORB-short ORB[7240.00,7316.65] vol=1.9x ATR=16.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:15:00 | 7214.51 | 7248.92 | 0.00 | T1 1.5R @ 7214.51 |
| Stop hit — per-position SL triggered | 2024-06-19 11:50:00 | 7238.55 | 7239.71 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 09:50:00 | 7203.00 | 7224.37 | 0.00 | ORB-short ORB[7212.65,7269.85] vol=2.1x ATR=15.25 |
| Stop hit — per-position SL triggered | 2024-06-20 10:40:00 | 7218.25 | 7219.01 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:45:00 | 7136.00 | 7160.31 | 0.00 | ORB-short ORB[7150.00,7228.80] vol=1.7x ATR=22.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 13:40:00 | 7101.91 | 7135.95 | 0.00 | T1 1.5R @ 7101.91 |
| Target hit | 2024-06-25 15:20:00 | 7102.00 | 7129.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 7141.75 | 7114.29 | 0.00 | ORB-long ORB[7075.00,7139.60] vol=1.9x ATR=11.99 |
| Stop hit — per-position SL triggered | 2024-06-26 11:50:00 | 7129.76 | 7119.89 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:20:00 | 7065.35 | 7029.20 | 0.00 | ORB-long ORB[6980.00,7040.35] vol=3.1x ATR=13.04 |
| Stop hit — per-position SL triggered | 2024-07-01 10:30:00 | 7052.31 | 7032.43 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 7024.50 | 7041.74 | 0.00 | ORB-short ORB[7026.00,7095.95] vol=1.6x ATR=12.68 |
| Target hit | 2024-07-08 15:20:00 | 7024.00 | 7030.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 6984.00 | 6994.58 | 0.00 | ORB-short ORB[6984.05,7027.00] vol=2.5x ATR=10.97 |
| Stop hit — per-position SL triggered | 2024-07-09 09:35:00 | 6994.97 | 6994.49 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 11:00:00 | 7006.45 | 6981.92 | 0.00 | ORB-long ORB[6950.05,6998.95] vol=3.9x ATR=11.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:05:00 | 7023.56 | 6984.79 | 0.00 | T1 1.5R @ 7023.56 |
| Stop hit — per-position SL triggered | 2024-07-12 11:20:00 | 7006.45 | 6987.68 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 6954.00 | 6982.32 | 0.00 | ORB-short ORB[6968.05,7028.95] vol=1.8x ATR=10.84 |
| Stop hit — per-position SL triggered | 2024-07-18 09:35:00 | 6964.84 | 6975.89 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 11:00:00 | 6970.80 | 7003.51 | 0.00 | ORB-short ORB[6991.55,7074.90] vol=1.7x ATR=15.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 11:10:00 | 6947.77 | 6996.89 | 0.00 | T1 1.5R @ 6947.77 |
| Stop hit — per-position SL triggered | 2024-07-19 14:45:00 | 6970.80 | 6978.39 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 6918.05 | 6939.59 | 0.00 | ORB-short ORB[6924.30,6997.70] vol=2.7x ATR=12.75 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 6930.80 | 6939.24 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:10:00 | 6875.00 | 6901.68 | 0.00 | ORB-short ORB[6892.00,6936.65] vol=1.6x ATR=9.74 |
| Stop hit — per-position SL triggered | 2024-07-25 10:45:00 | 6884.74 | 6891.34 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:00:00 | 6917.95 | 6914.09 | 0.00 | ORB-long ORB[6871.00,6917.85] vol=2.1x ATR=10.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 11:45:00 | 6933.29 | 6917.98 | 0.00 | T1 1.5R @ 6933.29 |
| Target hit | 2024-07-26 15:20:00 | 6965.10 | 6936.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 6965.00 | 6984.37 | 0.00 | ORB-short ORB[6975.00,7010.00] vol=2.1x ATR=12.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-29 10:15:00 | 6945.66 | 6973.48 | 0.00 | T1 1.5R @ 6945.66 |
| Target hit | 2024-07-29 15:20:00 | 6924.80 | 6950.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 6986.85 | 6968.31 | 0.00 | ORB-long ORB[6925.00,6980.00] vol=1.9x ATR=9.82 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 6977.03 | 6968.85 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:45:00 | 6894.15 | 6914.81 | 0.00 | ORB-short ORB[6902.00,6940.00] vol=2.0x ATR=14.44 |
| Stop hit — per-position SL triggered | 2024-08-02 09:50:00 | 6908.59 | 6915.09 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 11:15:00 | 6879.65 | 6834.49 | 0.00 | ORB-long ORB[6777.80,6873.55] vol=8.1x ATR=20.91 |
| Stop hit — per-position SL triggered | 2024-08-06 11:20:00 | 6858.74 | 6837.01 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:55:00 | 6786.85 | 6751.78 | 0.00 | ORB-long ORB[6724.50,6754.90] vol=1.6x ATR=17.72 |
| Stop hit — per-position SL triggered | 2024-08-07 11:00:00 | 6769.13 | 6752.03 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:25:00 | 6808.00 | 6850.77 | 0.00 | ORB-short ORB[6856.95,6899.90] vol=2.2x ATR=14.48 |
| Stop hit — per-position SL triggered | 2024-08-09 11:05:00 | 6822.48 | 6838.28 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:00:00 | 6822.00 | 6849.40 | 0.00 | ORB-short ORB[6827.75,6866.95] vol=1.7x ATR=15.80 |
| Stop hit — per-position SL triggered | 2024-08-16 10:20:00 | 6837.80 | 6845.61 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:50:00 | 6870.00 | 6852.94 | 0.00 | ORB-long ORB[6799.65,6868.00] vol=1.6x ATR=20.07 |
| Stop hit — per-position SL triggered | 2024-08-19 10:30:00 | 6849.93 | 6853.44 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:40:00 | 6974.65 | 6936.83 | 0.00 | ORB-long ORB[6901.00,6965.00] vol=2.1x ATR=16.64 |
| Stop hit — per-position SL triggered | 2024-08-20 10:50:00 | 6958.01 | 6939.59 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:55:00 | 6961.95 | 6936.99 | 0.00 | ORB-long ORB[6916.85,6950.00] vol=2.5x ATR=13.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:05:00 | 6982.41 | 6943.95 | 0.00 | T1 1.5R @ 6982.41 |
| Stop hit — per-position SL triggered | 2024-08-21 11:15:00 | 6961.95 | 6947.68 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:50:00 | 7008.80 | 6981.98 | 0.00 | ORB-long ORB[6938.85,6985.00] vol=4.1x ATR=18.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 09:55:00 | 7036.77 | 6991.82 | 0.00 | T1 1.5R @ 7036.77 |
| Stop hit — per-position SL triggered | 2024-08-22 10:00:00 | 7008.80 | 6994.38 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 7209.30 | 7160.33 | 0.00 | ORB-long ORB[7103.00,7173.40] vol=2.5x ATR=26.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:35:00 | 7248.58 | 7204.64 | 0.00 | T1 1.5R @ 7248.58 |
| Target hit | 2024-08-26 15:20:00 | 7899.00 | 7486.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-08-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:50:00 | 8282.10 | 8134.40 | 0.00 | ORB-long ORB[7952.05,8074.80] vol=4.3x ATR=55.75 |
| Stop hit — per-position SL triggered | 2024-08-30 10:00:00 | 8226.35 | 8177.26 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:50:00 | 7920.00 | 7843.29 | 0.00 | ORB-long ORB[7770.00,7849.70] vol=3.4x ATR=28.90 |
| Stop hit — per-position SL triggered | 2024-09-03 09:55:00 | 7891.10 | 7864.22 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 7498.35 | 7535.04 | 0.00 | ORB-short ORB[7510.00,7615.00] vol=1.6x ATR=22.39 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 7520.74 | 7534.17 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:40:00 | 7994.95 | 7931.23 | 0.00 | ORB-long ORB[7845.45,7947.00] vol=2.8x ATR=43.34 |
| Stop hit — per-position SL triggered | 2024-09-10 11:20:00 | 7951.61 | 7943.30 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 7780.15 | 7795.92 | 0.00 | ORB-short ORB[7782.00,7830.00] vol=1.6x ATR=21.48 |
| Stop hit — per-position SL triggered | 2024-09-13 10:00:00 | 7801.63 | 7795.77 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 7718.85 | 7701.89 | 0.00 | ORB-long ORB[7661.50,7715.95] vol=5.5x ATR=22.58 |
| Stop hit — per-position SL triggered | 2024-09-17 10:20:00 | 7696.27 | 7702.67 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 7571.00 | 7665.87 | 0.00 | ORB-short ORB[7673.45,7757.00] vol=1.6x ATR=24.63 |
| Stop hit — per-position SL triggered | 2024-09-19 10:20:00 | 7595.63 | 7662.18 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 7708.90 | 7649.87 | 0.00 | ORB-long ORB[7560.40,7674.00] vol=1.9x ATR=25.15 |
| Stop hit — per-position SL triggered | 2024-09-20 10:40:00 | 7683.75 | 7652.17 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:45:00 | 7957.60 | 7884.34 | 0.00 | ORB-long ORB[7840.20,7913.80] vol=2.8x ATR=36.91 |
| Stop hit — per-position SL triggered | 2024-09-23 10:55:00 | 7920.69 | 7897.06 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-09-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:35:00 | 7875.15 | 7829.15 | 0.00 | ORB-long ORB[7800.00,7855.00] vol=2.3x ATR=23.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:40:00 | 7910.79 | 7862.50 | 0.00 | T1 1.5R @ 7910.79 |
| Stop hit — per-position SL triggered | 2024-09-24 11:05:00 | 7875.15 | 7881.76 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:35:00 | 7872.00 | 7838.03 | 0.00 | ORB-long ORB[7800.00,7835.00] vol=2.2x ATR=21.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:40:00 | 7904.08 | 7858.51 | 0.00 | T1 1.5R @ 7904.08 |
| Stop hit — per-position SL triggered | 2024-09-26 10:00:00 | 7872.00 | 7872.66 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:15:00 | 7790.40 | 7763.35 | 0.00 | ORB-long ORB[7725.00,7785.85] vol=1.8x ATR=21.45 |
| Stop hit — per-position SL triggered | 2024-10-01 10:25:00 | 7768.95 | 7763.88 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:55:00 | 7499.70 | 7565.88 | 0.00 | ORB-short ORB[7576.10,7627.95] vol=1.6x ATR=30.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:10:00 | 7454.60 | 7533.97 | 0.00 | T1 1.5R @ 7454.60 |
| Target hit | 2024-10-07 11:20:00 | 7462.95 | 7459.68 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-10-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:35:00 | 7519.70 | 7447.29 | 0.00 | ORB-long ORB[7380.00,7436.00] vol=1.7x ATR=26.85 |
| Stop hit — per-position SL triggered | 2024-10-08 10:50:00 | 7492.85 | 7459.03 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 7547.90 | 7585.52 | 0.00 | ORB-short ORB[7571.65,7631.35] vol=1.8x ATR=18.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:35:00 | 7520.32 | 7569.00 | 0.00 | T1 1.5R @ 7520.32 |
| Target hit | 2024-10-15 15:20:00 | 7470.35 | 7502.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2024-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 09:30:00 | 7497.70 | 7466.19 | 0.00 | ORB-long ORB[7420.10,7485.00] vol=2.7x ATR=18.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:35:00 | 7524.70 | 7481.19 | 0.00 | T1 1.5R @ 7524.70 |
| Stop hit — per-position SL triggered | 2024-10-17 09:40:00 | 7497.70 | 7481.55 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 7308.15 | 7344.41 | 0.00 | ORB-short ORB[7320.00,7389.60] vol=2.0x ATR=18.83 |
| Stop hit — per-position SL triggered | 2024-10-21 09:40:00 | 7326.98 | 7342.54 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:45:00 | 7076.45 | 7111.41 | 0.00 | ORB-short ORB[7101.55,7194.95] vol=1.7x ATR=31.91 |
| Stop hit — per-position SL triggered | 2024-10-23 10:00:00 | 7108.36 | 7109.05 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 6892.20 | 6959.61 | 0.00 | ORB-short ORB[6971.00,7048.45] vol=1.5x ATR=28.29 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 6920.49 | 6955.17 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:45:00 | 7140.00 | 7102.98 | 0.00 | ORB-long ORB[7029.95,7127.00] vol=1.6x ATR=21.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:05:00 | 7171.71 | 7124.37 | 0.00 | T1 1.5R @ 7171.71 |
| Stop hit — per-position SL triggered | 2024-11-06 10:30:00 | 7140.00 | 7131.18 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:15:00 | 6587.50 | 6547.34 | 0.00 | ORB-long ORB[6497.60,6558.45] vol=2.0x ATR=17.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 12:10:00 | 6614.35 | 6568.06 | 0.00 | T1 1.5R @ 6614.35 |
| Stop hit — per-position SL triggered | 2024-11-22 12:50:00 | 6587.50 | 6573.98 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:25:00 | 6703.85 | 6670.65 | 0.00 | ORB-long ORB[6635.00,6690.00] vol=1.6x ATR=19.05 |
| Stop hit — per-position SL triggered | 2024-11-25 10:35:00 | 6684.80 | 6672.38 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:00:00 | 6813.40 | 6737.63 | 0.00 | ORB-long ORB[6684.40,6745.00] vol=2.0x ATR=26.48 |
| Stop hit — per-position SL triggered | 2024-11-27 11:45:00 | 6786.92 | 6772.35 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:20:00 | 7408.30 | 7359.12 | 0.00 | ORB-long ORB[7324.45,7394.00] vol=2.1x ATR=27.68 |
| Stop hit — per-position SL triggered | 2024-12-06 10:25:00 | 7380.62 | 7360.55 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:45:00 | 6973.70 | 6939.63 | 0.00 | ORB-long ORB[6895.20,6949.00] vol=3.8x ATR=23.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:00:00 | 7008.94 | 6957.65 | 0.00 | T1 1.5R @ 7008.94 |
| Stop hit — per-position SL triggered | 2024-12-24 10:20:00 | 6973.70 | 6962.78 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:05:00 | 6882.05 | 6915.39 | 0.00 | ORB-short ORB[6914.05,6974.80] vol=1.7x ATR=17.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:20:00 | 6855.88 | 6913.00 | 0.00 | T1 1.5R @ 6855.88 |
| Stop hit — per-position SL triggered | 2024-12-27 14:00:00 | 6882.05 | 6888.29 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:45:00 | 6866.00 | 6873.51 | 0.00 | ORB-short ORB[6872.10,6911.75] vol=1.8x ATR=16.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:00:00 | 6841.52 | 6872.17 | 0.00 | T1 1.5R @ 6841.52 |
| Stop hit — per-position SL triggered | 2024-12-30 11:20:00 | 6866.00 | 6870.87 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 11:00:00 | 6742.00 | 6784.86 | 0.00 | ORB-short ORB[6777.05,6840.00] vol=4.6x ATR=14.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:05:00 | 6719.99 | 6772.55 | 0.00 | T1 1.5R @ 6719.99 |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 6742.00 | 6769.29 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:55:00 | 6552.55 | 6510.14 | 0.00 | ORB-long ORB[6451.15,6539.05] vol=1.8x ATR=26.63 |
| Stop hit — per-position SL triggered | 2025-01-09 10:10:00 | 6525.92 | 6522.07 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:45:00 | 6184.25 | 6153.73 | 0.00 | ORB-long ORB[6130.00,6181.00] vol=1.6x ATR=16.46 |
| Stop hit — per-position SL triggered | 2025-01-16 09:55:00 | 6167.79 | 6158.50 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:45:00 | 6220.85 | 6205.02 | 0.00 | ORB-long ORB[6150.00,6199.95] vol=2.3x ATR=15.06 |
| Stop hit — per-position SL triggered | 2025-01-17 10:55:00 | 6205.79 | 6205.60 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:35:00 | 6117.00 | 6142.26 | 0.00 | ORB-short ORB[6131.00,6173.45] vol=2.5x ATR=20.06 |
| Stop hit — per-position SL triggered | 2025-01-22 09:40:00 | 6137.06 | 6140.42 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:30:00 | 6242.90 | 6280.50 | 0.00 | ORB-short ORB[6268.05,6327.55] vol=1.6x ATR=21.54 |
| Stop hit — per-position SL triggered | 2025-01-27 09:35:00 | 6264.44 | 6278.25 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 09:40:00 | 6113.70 | 6153.31 | 0.00 | ORB-short ORB[6129.05,6203.90] vol=2.0x ATR=25.96 |
| Stop hit — per-position SL triggered | 2025-01-28 09:45:00 | 6139.66 | 6149.98 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:35:00 | 6209.00 | 6172.07 | 0.00 | ORB-long ORB[6110.05,6180.00] vol=1.6x ATR=17.88 |
| Stop hit — per-position SL triggered | 2025-01-29 10:00:00 | 6191.12 | 6185.98 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:55:00 | 6131.75 | 6170.60 | 0.00 | ORB-short ORB[6145.00,6215.95] vol=1.8x ATR=15.27 |
| Stop hit — per-position SL triggered | 2025-01-30 11:55:00 | 6147.02 | 6156.18 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:05:00 | 7083.15 | 2024-05-15 10:15:00 | 7067.44 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-05-15 10:05:00 | 7083.15 | 2024-05-15 10:25:00 | 7083.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:40:00 | 7208.15 | 2024-05-22 09:45:00 | 7229.43 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-30 09:30:00 | 7214.70 | 2024-05-30 11:45:00 | 7181.57 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-30 09:30:00 | 7214.70 | 2024-05-30 15:00:00 | 7207.45 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-05-31 11:05:00 | 7131.70 | 2024-05-31 11:10:00 | 7148.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-13 09:30:00 | 7212.10 | 2024-06-13 09:35:00 | 7241.65 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-06-13 09:30:00 | 7212.10 | 2024-06-13 10:15:00 | 7229.50 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-06-14 10:20:00 | 7346.25 | 2024-06-14 10:25:00 | 7320.86 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-06-19 09:30:00 | 7238.55 | 2024-06-19 10:15:00 | 7214.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-06-19 09:30:00 | 7238.55 | 2024-06-19 11:50:00 | 7238.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-20 09:50:00 | 7203.00 | 2024-06-20 10:40:00 | 7218.25 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-06-25 09:45:00 | 7136.00 | 2024-06-25 13:40:00 | 7101.91 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-06-25 09:45:00 | 7136.00 | 2024-06-25 15:20:00 | 7102.00 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-26 10:55:00 | 7141.75 | 2024-06-26 11:50:00 | 7129.76 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-07-01 10:20:00 | 7065.35 | 2024-07-01 10:30:00 | 7052.31 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-07-08 09:55:00 | 7024.50 | 2024-07-08 15:20:00 | 7024.00 | TARGET_HIT | 1.00 | 0.01% |
| SELL | retest1 | 2024-07-09 09:30:00 | 6984.00 | 2024-07-09 09:35:00 | 6994.97 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-07-12 11:00:00 | 7006.45 | 2024-07-12 11:05:00 | 7023.56 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-07-12 11:00:00 | 7006.45 | 2024-07-12 11:20:00 | 7006.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 09:30:00 | 6954.00 | 2024-07-18 09:35:00 | 6964.84 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-07-19 11:00:00 | 6970.80 | 2024-07-19 11:10:00 | 6947.77 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-07-19 11:00:00 | 6970.80 | 2024-07-19 14:45:00 | 6970.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 6918.05 | 2024-07-23 11:20:00 | 6930.80 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-07-25 10:10:00 | 6875.00 | 2024-07-25 10:45:00 | 6884.74 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-07-26 11:00:00 | 6917.95 | 2024-07-26 11:45:00 | 6933.29 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2024-07-26 11:00:00 | 6917.95 | 2024-07-26 15:20:00 | 6965.10 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-07-29 09:30:00 | 6965.00 | 2024-07-29 10:15:00 | 6945.66 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-07-29 09:30:00 | 6965.00 | 2024-07-29 15:20:00 | 6924.80 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-31 09:40:00 | 6986.85 | 2024-07-31 09:45:00 | 6977.03 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-08-02 09:45:00 | 6894.15 | 2024-08-02 09:50:00 | 6908.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-06 11:15:00 | 6879.65 | 2024-08-06 11:20:00 | 6858.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-07 10:55:00 | 6786.85 | 2024-08-07 11:00:00 | 6769.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-09 10:25:00 | 6808.00 | 2024-08-09 11:05:00 | 6822.48 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-16 10:00:00 | 6822.00 | 2024-08-16 10:20:00 | 6837.80 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-19 09:50:00 | 6870.00 | 2024-08-19 10:30:00 | 6849.93 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-20 10:40:00 | 6974.65 | 2024-08-20 10:50:00 | 6958.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-21 10:55:00 | 6961.95 | 2024-08-21 11:05:00 | 6982.41 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-08-21 10:55:00 | 6961.95 | 2024-08-21 11:15:00 | 6961.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:50:00 | 7008.80 | 2024-08-22 09:55:00 | 7036.77 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-22 09:50:00 | 7008.80 | 2024-08-22 10:00:00 | 7008.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:35:00 | 7209.30 | 2024-08-26 10:35:00 | 7248.58 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-26 09:35:00 | 7209.30 | 2024-08-26 15:20:00 | 7899.00 | TARGET_HIT | 0.50 | 9.57% |
| BUY | retest1 | 2024-08-30 09:50:00 | 8282.10 | 2024-08-30 10:00:00 | 8226.35 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2024-09-03 09:50:00 | 7920.00 | 2024-09-03 09:55:00 | 7891.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-09 09:30:00 | 7498.35 | 2024-09-09 09:35:00 | 7520.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-10 10:40:00 | 7994.95 | 2024-09-10 11:20:00 | 7951.61 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-09-13 09:30:00 | 7780.15 | 2024-09-13 10:00:00 | 7801.63 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-17 10:10:00 | 7718.85 | 2024-09-17 10:20:00 | 7696.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-19 10:15:00 | 7571.00 | 2024-09-19 10:20:00 | 7595.63 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-20 10:35:00 | 7708.90 | 2024-09-20 10:40:00 | 7683.75 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-23 10:45:00 | 7957.60 | 2024-09-23 10:55:00 | 7920.69 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-09-24 10:35:00 | 7875.15 | 2024-09-24 10:40:00 | 7910.79 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-09-24 10:35:00 | 7875.15 | 2024-09-24 11:05:00 | 7875.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 09:35:00 | 7872.00 | 2024-09-26 09:40:00 | 7904.08 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-26 09:35:00 | 7872.00 | 2024-09-26 10:00:00 | 7872.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-01 10:15:00 | 7790.40 | 2024-10-01 10:25:00 | 7768.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-07 09:55:00 | 7499.70 | 2024-10-07 10:10:00 | 7454.60 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-07 09:55:00 | 7499.70 | 2024-10-07 11:20:00 | 7462.95 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2024-10-08 10:35:00 | 7519.70 | 2024-10-08 10:50:00 | 7492.85 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-15 09:30:00 | 7547.90 | 2024-10-15 09:35:00 | 7520.32 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-15 09:30:00 | 7547.90 | 2024-10-15 15:20:00 | 7470.35 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-10-17 09:30:00 | 7497.70 | 2024-10-17 09:35:00 | 7524.70 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-10-17 09:30:00 | 7497.70 | 2024-10-17 09:40:00 | 7497.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:35:00 | 7308.15 | 2024-10-21 09:40:00 | 7326.98 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-23 09:45:00 | 7076.45 | 2024-10-23 10:00:00 | 7108.36 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-25 10:55:00 | 6892.20 | 2024-10-25 11:15:00 | 6920.49 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-11-06 09:45:00 | 7140.00 | 2024-11-06 10:05:00 | 7171.71 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-11-06 09:45:00 | 7140.00 | 2024-11-06 10:30:00 | 7140.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:15:00 | 6587.50 | 2024-11-22 12:10:00 | 6614.35 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-11-22 10:15:00 | 6587.50 | 2024-11-22 12:50:00 | 6587.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 10:25:00 | 6703.85 | 2024-11-25 10:35:00 | 6684.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-11-27 10:00:00 | 6813.40 | 2024-11-27 11:45:00 | 6786.92 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-06 10:20:00 | 7408.30 | 2024-12-06 10:25:00 | 7380.62 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-24 09:45:00 | 6973.70 | 2024-12-24 10:00:00 | 7008.94 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-12-24 09:45:00 | 6973.70 | 2024-12-24 10:20:00 | 6973.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 11:05:00 | 6882.05 | 2024-12-27 11:20:00 | 6855.88 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-12-27 11:05:00 | 6882.05 | 2024-12-27 14:00:00 | 6882.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-30 10:45:00 | 6866.00 | 2024-12-30 11:00:00 | 6841.52 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-30 10:45:00 | 6866.00 | 2024-12-30 11:20:00 | 6866.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 11:00:00 | 6742.00 | 2025-01-01 11:05:00 | 6719.99 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-01-01 11:00:00 | 6742.00 | 2025-01-01 11:15:00 | 6742.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-09 09:55:00 | 6552.55 | 2025-01-09 10:10:00 | 6525.92 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-01-16 09:45:00 | 6184.25 | 2025-01-16 09:55:00 | 6167.79 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-17 10:45:00 | 6220.85 | 2025-01-17 10:55:00 | 6205.79 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-22 09:35:00 | 6117.00 | 2025-01-22 09:40:00 | 6137.06 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-27 09:30:00 | 6242.90 | 2025-01-27 09:35:00 | 6264.44 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-28 09:40:00 | 6113.70 | 2025-01-28 09:45:00 | 6139.66 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-01-29 09:35:00 | 6209.00 | 2025-01-29 10:00:00 | 6191.12 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-30 10:55:00 | 6131.75 | 2025-01-30 11:55:00 | 6147.02 | STOP_HIT | 1.00 | -0.25% |
