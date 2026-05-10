# Eicher Motors Ltd. (EICHERMOT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 7309.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 14
- **Target hits / Stop hits / Partials:** 1 / 14 / 4
- **Avg / median % per leg:** -0.07% / -0.19%
- **Sum % (uncompounded):** -1.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.12% | -1.2% |
| BUY @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.12% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.02% | -0.2% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.02% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 5 | 26.3% | 1 | 14 | 4 | -0.07% | -1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 7200.50 | 7147.69 | 0.00 | ORB-long ORB[7063.50,7144.50] vol=2.0x ATR=29.49 |
| Stop hit — per-position SL triggered | 2026-02-09 14:05:00 | 7171.01 | 7165.87 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:05:00 | 7289.50 | 7249.39 | 0.00 | ORB-long ORB[7201.00,7280.00] vol=2.1x ATR=14.11 |
| Stop hit — per-position SL triggered | 2026-02-10 11:10:00 | 7275.39 | 7250.29 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 7967.50 | 7979.47 | 0.00 | ORB-short ORB[7974.00,8043.50] vol=1.8x ATR=13.01 |
| Stop hit — per-position SL triggered | 2026-02-18 12:30:00 | 7980.51 | 7977.10 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 8120.00 | 8083.12 | 0.00 | ORB-long ORB[8016.00,8110.00] vol=2.1x ATR=19.58 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 8100.42 | 8095.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:40:00 | 7465.50 | 7395.50 | 0.00 | ORB-long ORB[7338.50,7437.00] vol=1.6x ATR=24.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:25:00 | 7501.97 | 7423.60 | 0.00 | T1 1.5R @ 7501.97 |
| Stop hit — per-position SL triggered | 2026-03-10 11:45:00 | 7465.50 | 7432.53 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 7422.00 | 7491.43 | 0.00 | ORB-short ORB[7480.00,7574.00] vol=1.8x ATR=22.46 |
| Stop hit — per-position SL triggered | 2026-03-11 12:00:00 | 7444.46 | 7473.56 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:00:00 | 6944.50 | 6901.98 | 0.00 | ORB-long ORB[6770.00,6837.50] vol=1.5x ATR=26.60 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 6917.90 | 6903.74 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 09:55:00 | 6844.50 | 6872.69 | 0.00 | ORB-short ORB[6870.50,6916.50] vol=2.0x ATR=20.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 6813.27 | 6863.55 | 0.00 | T1 1.5R @ 6813.27 |
| Target hit | 2026-03-27 12:50:00 | 6831.50 | 6815.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 7335.00 | 7280.84 | 0.00 | ORB-long ORB[7201.00,7310.00] vol=1.6x ATR=32.72 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 7302.28 | 7297.97 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:20:00 | 7113.00 | 7126.22 | 0.00 | ORB-short ORB[7115.00,7199.50] vol=2.9x ATR=20.05 |
| Stop hit — per-position SL triggered | 2026-04-15 10:25:00 | 7133.05 | 7126.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:15:00 | 7280.00 | 7264.16 | 0.00 | ORB-long ORB[7235.00,7276.00] vol=5.4x ATR=10.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 12:25:00 | 7295.41 | 7273.13 | 0.00 | T1 1.5R @ 7295.41 |
| Stop hit — per-position SL triggered | 2026-04-21 12:35:00 | 7280.00 | 7273.52 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:35:00 | 7106.50 | 7150.81 | 0.00 | ORB-short ORB[7136.50,7199.00] vol=1.7x ATR=18.66 |
| Stop hit — per-position SL triggered | 2026-04-27 10:45:00 | 7125.16 | 7148.89 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 7232.00 | 7277.25 | 0.00 | ORB-short ORB[7246.00,7330.00] vol=1.5x ATR=19.07 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 7251.07 | 7275.91 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 7265.00 | 7341.43 | 0.00 | ORB-short ORB[7324.50,7392.00] vol=3.1x ATR=20.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:55:00 | 7234.13 | 7330.62 | 0.00 | T1 1.5R @ 7234.13 |
| Stop hit — per-position SL triggered | 2026-05-06 13:05:00 | 7265.00 | 7283.91 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 7358.00 | 7299.77 | 0.00 | ORB-long ORB[7228.00,7312.00] vol=2.3x ATR=15.65 |
| Stop hit — per-position SL triggered | 2026-05-08 12:05:00 | 7342.35 | 7314.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:00:00 | 7200.50 | 2026-02-09 14:05:00 | 7171.01 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-10 11:05:00 | 7289.50 | 2026-02-10 11:10:00 | 7275.39 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-18 11:05:00 | 7967.50 | 2026-02-18 12:30:00 | 7980.51 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-19 09:30:00 | 8120.00 | 2026-02-19 09:40:00 | 8100.42 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-10 10:40:00 | 7465.50 | 2026-03-10 11:25:00 | 7501.97 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-10 10:40:00 | 7465.50 | 2026-03-10 11:45:00 | 7465.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:10:00 | 7422.00 | 2026-03-11 12:00:00 | 7444.46 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-17 11:00:00 | 6944.50 | 2026-03-17 11:15:00 | 6917.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-27 09:55:00 | 6844.50 | 2026-03-27 10:15:00 | 6813.27 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-27 09:55:00 | 6844.50 | 2026-03-27 12:50:00 | 6831.50 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2026-04-10 09:45:00 | 7335.00 | 2026-04-10 10:05:00 | 7302.28 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-15 10:20:00 | 7113.00 | 2026-04-15 10:25:00 | 7133.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-21 11:15:00 | 7280.00 | 2026-04-21 12:25:00 | 7295.41 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2026-04-21 11:15:00 | 7280.00 | 2026-04-21 12:35:00 | 7280.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-27 10:35:00 | 7106.50 | 2026-04-27 10:45:00 | 7125.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-05 10:55:00 | 7232.00 | 2026-05-05 11:05:00 | 7251.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-05-06 10:50:00 | 7265.00 | 2026-05-06 10:55:00 | 7234.13 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-06 10:50:00 | 7265.00 | 2026-05-06 13:05:00 | 7265.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 11:00:00 | 7358.00 | 2026-05-08 12:05:00 | 7342.35 | STOP_HIT | 1.00 | -0.21% |
