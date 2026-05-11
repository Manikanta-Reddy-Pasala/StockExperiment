# UltraTech Cement Ltd. (ULTRACEMCO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-07-09 15:25:00 (3021 bars)
- **Last close:** 11668.85
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
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 5
- **Avg / median % per leg:** 0.07% / -0.20%
- **Sum % (uncompounded):** 1.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 1 | 9 | 1 | -0.12% | -1.4% |
| BUY @ 2nd Alert (retest1) | 11 | 2 | 18.2% | 1 | 9 | 1 | -0.12% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.23% | 3.0% |
| SELL @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.23% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 8 | 33.3% | 3 | 16 | 5 | 0.07% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 9576.30 | 9612.27 | 0.00 | ORB-short ORB[9590.05,9655.00] vol=1.8x ATR=21.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:50:00 | 9543.94 | 9589.40 | 0.00 | T1 1.5R @ 9543.94 |
| Stop hit — per-position SL triggered | 2024-05-16 10:05:00 | 9576.30 | 9584.15 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:30:00 | 9793.00 | 9758.27 | 0.00 | ORB-long ORB[9685.25,9747.55] vol=1.9x ATR=23.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 10:35:00 | 9828.75 | 9766.75 | 0.00 | T1 1.5R @ 9828.75 |
| Target hit | 2024-05-17 15:20:00 | 9890.05 | 9836.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:40:00 | 10296.00 | 10201.13 | 0.00 | ORB-long ORB[10122.55,10200.00] vol=1.7x ATR=28.97 |
| Stop hit — per-position SL triggered | 2024-05-24 10:50:00 | 10267.03 | 10205.85 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 10311.20 | 10238.67 | 0.00 | ORB-long ORB[10180.65,10280.85] vol=2.7x ATR=39.74 |
| Stop hit — per-position SL triggered | 2024-05-27 09:45:00 | 10271.46 | 10264.44 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 10342.00 | 10270.76 | 0.00 | ORB-long ORB[10200.00,10282.40] vol=2.2x ATR=31.77 |
| Stop hit — per-position SL triggered | 2024-05-28 10:05:00 | 10310.23 | 10304.39 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:35:00 | 9997.05 | 10078.49 | 0.00 | ORB-short ORB[10103.30,10177.95] vol=2.2x ATR=28.28 |
| Stop hit — per-position SL triggered | 2024-05-29 11:25:00 | 10025.33 | 10044.39 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:50:00 | 9938.90 | 9963.90 | 0.00 | ORB-short ORB[9955.60,10025.00] vol=1.7x ATR=23.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 10:10:00 | 9903.81 | 9950.54 | 0.00 | T1 1.5R @ 9903.81 |
| Target hit | 2024-05-30 15:20:00 | 9855.00 | 9887.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:50:00 | 10131.40 | 10028.01 | 0.00 | ORB-long ORB[9931.10,10037.95] vol=2.2x ATR=37.54 |
| Stop hit — per-position SL triggered | 2024-06-06 11:30:00 | 10093.86 | 10043.72 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:10:00 | 11028.25 | 10964.92 | 0.00 | ORB-long ORB[10871.50,10978.90] vol=3.5x ATR=26.37 |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 11001.88 | 10966.07 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 11269.45 | 11214.02 | 0.00 | ORB-long ORB[11147.30,11225.00] vol=1.8x ATR=31.58 |
| Stop hit — per-position SL triggered | 2024-06-14 09:35:00 | 11237.87 | 11217.89 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:45:00 | 10979.40 | 11073.40 | 0.00 | ORB-short ORB[11042.15,11177.70] vol=1.7x ATR=33.75 |
| Stop hit — per-position SL triggered | 2024-06-19 10:55:00 | 11013.15 | 11066.42 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:05:00 | 10824.20 | 10876.54 | 0.00 | ORB-short ORB[10837.00,10961.45] vol=2.0x ATR=36.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:45:00 | 10768.94 | 10850.11 | 0.00 | T1 1.5R @ 10768.94 |
| Target hit | 2024-06-21 15:20:00 | 10618.55 | 10733.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:15:00 | 10740.10 | 10687.53 | 0.00 | ORB-long ORB[10594.65,10721.20] vol=1.9x ATR=26.21 |
| Stop hit — per-position SL triggered | 2024-06-24 11:20:00 | 10713.89 | 10688.40 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 10944.45 | 10899.51 | 0.00 | ORB-long ORB[10800.40,10917.75] vol=3.6x ATR=29.10 |
| Stop hit — per-position SL triggered | 2024-06-25 10:05:00 | 10915.35 | 10913.55 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:10:00 | 12060.10 | 11944.68 | 0.00 | ORB-long ORB[11817.70,11976.15] vol=2.3x ATR=41.06 |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 12019.04 | 11951.19 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 11738.90 | 11861.43 | 0.00 | ORB-short ORB[11806.10,11936.85] vol=2.5x ATR=34.15 |
| Stop hit — per-position SL triggered | 2024-07-04 10:30:00 | 11773.05 | 11845.94 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 11697.30 | 11735.46 | 0.00 | ORB-short ORB[11730.85,11824.55] vol=4.0x ATR=28.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 13:50:00 | 11654.80 | 11702.08 | 0.00 | T1 1.5R @ 11654.80 |
| Stop hit — per-position SL triggered | 2024-07-05 15:00:00 | 11697.30 | 11694.52 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 11560.00 | 11612.56 | 0.00 | ORB-short ORB[11609.90,11714.80] vol=2.5x ATR=23.00 |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 11583.00 | 11610.60 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:10:00 | 11505.65 | 11538.17 | 0.00 | ORB-short ORB[11520.00,11594.25] vol=1.7x ATR=20.79 |
| Stop hit — per-position SL triggered | 2024-07-09 13:00:00 | 11526.44 | 11526.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 9576.30 | 2024-05-16 09:50:00 | 9543.94 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-05-16 09:30:00 | 9576.30 | 2024-05-16 10:05:00 | 9576.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 10:30:00 | 9793.00 | 2024-05-17 10:35:00 | 9828.75 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-05-17 10:30:00 | 9793.00 | 2024-05-17 15:20:00 | 9890.05 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2024-05-24 10:40:00 | 10296.00 | 2024-05-24 10:50:00 | 10267.03 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-27 09:35:00 | 10311.20 | 2024-05-27 09:45:00 | 10271.46 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-28 09:45:00 | 10342.00 | 2024-05-28 10:05:00 | 10310.23 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-29 10:35:00 | 9997.05 | 2024-05-29 11:25:00 | 10025.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-30 09:50:00 | 9938.90 | 2024-05-30 10:10:00 | 9903.81 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-30 09:50:00 | 9938.90 | 2024-05-30 15:20:00 | 9855.00 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2024-06-06 10:50:00 | 10131.40 | 2024-06-06 11:30:00 | 10093.86 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-12 11:10:00 | 11028.25 | 2024-06-12 11:15:00 | 11001.88 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-14 09:30:00 | 11269.45 | 2024-06-14 09:35:00 | 11237.87 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-19 10:45:00 | 10979.40 | 2024-06-19 10:55:00 | 11013.15 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-21 10:05:00 | 10824.20 | 2024-06-21 10:45:00 | 10768.94 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-06-21 10:05:00 | 10824.20 | 2024-06-21 15:20:00 | 10618.55 | TARGET_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2024-06-24 11:15:00 | 10740.10 | 2024-06-24 11:20:00 | 10713.89 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-25 09:35:00 | 10944.45 | 2024-06-25 10:05:00 | 10915.35 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-02 10:10:00 | 12060.10 | 2024-07-02 10:15:00 | 12019.04 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-04 10:25:00 | 11738.90 | 2024-07-04 10:30:00 | 11773.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-05 09:35:00 | 11697.30 | 2024-07-05 13:50:00 | 11654.80 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-05 09:35:00 | 11697.30 | 2024-07-05 15:00:00 | 11697.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 11:10:00 | 11560.00 | 2024-07-08 11:15:00 | 11583.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-09 11:10:00 | 11505.65 | 2024-07-09 13:00:00 | 11526.44 | STOP_HIT | 1.00 | -0.18% |
