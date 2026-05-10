# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 16101.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 6
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 0.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.01% | -0.1% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | -0.01% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.13% | 0.9% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.13% | 0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 7 | 38.9% | 1 | 11 | 6 | 0.05% | 0.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 13271.00 | 13289.95 | 0.00 | ORB-short ORB[13280.00,13439.00] vol=1.7x ATR=27.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 13230.05 | 13286.73 | 0.00 | T1 1.5R @ 13230.05 |
| Stop hit — per-position SL triggered | 2026-02-12 12:55:00 | 13271.00 | 13279.61 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 13208.00 | 13163.70 | 0.00 | ORB-long ORB[13025.00,13199.00] vol=1.5x ATR=37.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:45:00 | 13263.73 | 13192.81 | 0.00 | T1 1.5R @ 13263.73 |
| Target hit | 2026-02-17 12:25:00 | 13220.00 | 13224.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 13242.00 | 13421.03 | 0.00 | ORB-short ORB[13401.00,13590.00] vol=2.4x ATR=46.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 13172.23 | 13400.05 | 0.00 | T1 1.5R @ 13172.23 |
| Stop hit — per-position SL triggered | 2026-02-23 12:10:00 | 13242.00 | 13368.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 13462.00 | 13406.55 | 0.00 | ORB-long ORB[13276.00,13415.00] vol=1.6x ATR=30.13 |
| Stop hit — per-position SL triggered | 2026-02-25 11:25:00 | 13431.87 | 13414.86 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 13541.00 | 13510.46 | 0.00 | ORB-long ORB[13405.00,13540.00] vol=1.7x ATR=33.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:50:00 | 13591.79 | 13550.01 | 0.00 | T1 1.5R @ 13591.79 |
| Stop hit — per-position SL triggered | 2026-02-26 10:00:00 | 13541.00 | 13555.07 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 14097.00 | 14008.85 | 0.00 | ORB-long ORB[13880.00,14088.00] vol=2.1x ATR=54.43 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 14042.57 | 14034.49 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:55:00 | 13204.00 | 13325.30 | 0.00 | ORB-short ORB[13251.00,13445.00] vol=1.5x ATR=49.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:05:00 | 13130.10 | 13283.89 | 0.00 | T1 1.5R @ 13130.10 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 13204.00 | 13249.86 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 15051.00 | 14971.33 | 0.00 | ORB-long ORB[14851.00,15035.00] vol=1.9x ATR=48.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:10:00 | 15124.14 | 15025.39 | 0.00 | T1 1.5R @ 15124.14 |
| Stop hit — per-position SL triggered | 2026-04-21 11:55:00 | 15051.00 | 15056.79 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 15045.00 | 14990.98 | 0.00 | ORB-long ORB[14925.00,15036.00] vol=1.8x ATR=40.83 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 15004.17 | 14998.16 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 15684.00 | 15852.31 | 0.00 | ORB-short ORB[15811.00,15995.00] vol=2.0x ATR=75.11 |
| Stop hit — per-position SL triggered | 2026-04-24 09:35:00 | 15759.11 | 15840.49 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 15541.00 | 15452.26 | 0.00 | ORB-long ORB[15407.00,15516.00] vol=1.5x ATR=40.99 |
| Stop hit — per-position SL triggered | 2026-04-29 11:45:00 | 15500.01 | 15467.97 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 15544.00 | 15498.07 | 0.00 | ORB-long ORB[15407.00,15524.00] vol=2.1x ATR=49.01 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 15494.99 | 15515.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:10:00 | 13271.00 | 2026-02-12 11:30:00 | 13230.05 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-12 11:10:00 | 13271.00 | 2026-02-12 12:55:00 | 13271.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:30:00 | 13208.00 | 2026-02-17 09:45:00 | 13263.73 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-17 09:30:00 | 13208.00 | 2026-02-17 12:25:00 | 13220.00 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-02-23 10:55:00 | 13242.00 | 2026-02-23 11:15:00 | 13172.23 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-23 10:55:00 | 13242.00 | 2026-02-23 12:10:00 | 13242.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 11:00:00 | 13462.00 | 2026-02-25 11:25:00 | 13431.87 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 09:30:00 | 13541.00 | 2026-02-26 09:50:00 | 13591.79 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-26 09:30:00 | 13541.00 | 2026-02-26 10:00:00 | 13541.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:35:00 | 14097.00 | 2026-03-18 09:55:00 | 14042.57 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-20 10:55:00 | 13204.00 | 2026-03-20 12:05:00 | 13130.10 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-20 10:55:00 | 13204.00 | 2026-03-20 13:15:00 | 13204.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 15051.00 | 2026-04-21 10:10:00 | 15124.14 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-21 09:35:00 | 15051.00 | 2026-04-21 11:55:00 | 15051.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:30:00 | 15045.00 | 2026-04-22 09:40:00 | 15004.17 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 09:30:00 | 15684.00 | 2026-04-24 09:35:00 | 15759.11 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-29 11:10:00 | 15541.00 | 2026-04-29 11:45:00 | 15500.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-05 09:30:00 | 15544.00 | 2026-05-05 09:45:00 | 15494.99 | STOP_HIT | 1.00 | -0.32% |
