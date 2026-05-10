# Honeywell Automation India Ltd. (HONAUT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 30210.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 7
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 3.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.29% | -1.7% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.29% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.28% | 5.4% |
| SELL @ 2nd Alert (retest1) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.28% | 5.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 10 | 40.0% | 3 | 15 | 7 | 0.15% | 3.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:15:00 | 32450.00 | 32575.97 | 0.00 | ORB-short ORB[32490.00,32775.00] vol=1.6x ATR=79.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:15:00 | 32330.04 | 32427.97 | 0.00 | T1 1.5R @ 32330.04 |
| Target hit | 2026-02-10 15:20:00 | 32155.00 | 32253.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 31360.00 | 31538.10 | 0.00 | ORB-short ORB[31655.00,31905.00] vol=3.2x ATR=83.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:15:00 | 31234.72 | 31399.24 | 0.00 | T1 1.5R @ 31234.72 |
| Stop hit — per-position SL triggered | 2026-02-13 10:30:00 | 31360.00 | 31395.14 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 31585.00 | 31353.46 | 0.00 | ORB-long ORB[31200.00,31510.00] vol=2.8x ATR=81.33 |
| Stop hit — per-position SL triggered | 2026-02-16 10:50:00 | 31503.67 | 31368.42 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 31420.00 | 31349.48 | 0.00 | ORB-long ORB[31090.00,31350.00] vol=4.7x ATR=50.77 |
| Stop hit — per-position SL triggered | 2026-02-17 11:00:00 | 31369.23 | 31365.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:45:00 | 31145.00 | 31016.26 | 0.00 | ORB-long ORB[30850.00,31105.00] vol=2.6x ATR=87.32 |
| Stop hit — per-position SL triggered | 2026-02-20 09:50:00 | 31057.68 | 31021.67 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:00:00 | 31205.00 | 31319.44 | 0.00 | ORB-short ORB[31305.00,31605.00] vol=2.8x ATR=74.12 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 31279.12 | 31302.36 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:40:00 | 30940.00 | 31071.28 | 0.00 | ORB-short ORB[31055.00,31295.00] vol=4.3x ATR=62.19 |
| Stop hit — per-position SL triggered | 2026-02-27 15:20:00 | 30995.00 | 30985.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-03-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-02 10:45:00 | 30450.00 | 30680.41 | 0.00 | ORB-short ORB[30500.00,30820.00] vol=3.2x ATR=89.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:40:00 | 30316.03 | 30616.81 | 0.00 | T1 1.5R @ 30316.03 |
| Stop hit — per-position SL triggered | 2026-03-02 12:55:00 | 30450.00 | 30613.51 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 10:50:00 | 29490.00 | 29810.48 | 0.00 | ORB-short ORB[29750.00,29945.00] vol=2.9x ATR=101.23 |
| Stop hit — per-position SL triggered | 2026-03-09 12:05:00 | 29591.23 | 29726.29 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:40:00 | 29690.00 | 29478.83 | 0.00 | ORB-long ORB[29165.00,29525.00] vol=6.5x ATR=124.48 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 29565.52 | 29487.40 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:50:00 | 27090.00 | 27369.76 | 0.00 | ORB-short ORB[27405.00,27800.00] vol=1.6x ATR=97.68 |
| Stop hit — per-position SL triggered | 2026-03-24 11:20:00 | 27187.68 | 27265.75 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:05:00 | 28415.00 | 28113.87 | 0.00 | ORB-long ORB[27760.00,28155.00] vol=4.1x ATR=87.52 |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 28327.48 | 28121.43 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 32280.00 | 32401.54 | 0.00 | ORB-short ORB[32370.00,32695.00] vol=1.8x ATR=80.70 |
| Stop hit — per-position SL triggered | 2026-04-22 11:10:00 | 32360.70 | 32372.32 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 10:35:00 | 32765.00 | 32603.73 | 0.00 | ORB-long ORB[32445.00,32710.00] vol=4.7x ATR=93.82 |
| Stop hit — per-position SL triggered | 2026-04-23 10:50:00 | 32671.18 | 32608.34 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:20:00 | 32205.00 | 32445.29 | 0.00 | ORB-short ORB[32475.00,32880.00] vol=3.5x ATR=144.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:20:00 | 31987.78 | 32335.67 | 0.00 | T1 1.5R @ 31987.78 |
| Target hit | 2026-04-24 15:20:00 | 31460.00 | 31878.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 31205.00 | 31349.93 | 0.00 | ORB-short ORB[31330.00,31515.00] vol=2.3x ATR=82.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:45:00 | 31081.77 | 31266.01 | 0.00 | T1 1.5R @ 31081.77 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 31205.00 | 31259.31 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 31080.00 | 31253.15 | 0.00 | ORB-short ORB[31150.00,31550.00] vol=1.5x ATR=79.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:55:00 | 30961.12 | 31198.07 | 0.00 | T1 1.5R @ 30961.12 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 31080.00 | 31171.91 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:55:00 | 30995.00 | 31149.12 | 0.00 | ORB-short ORB[31035.00,31495.00] vol=2.0x ATR=82.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 13:35:00 | 30871.78 | 30999.12 | 0.00 | T1 1.5R @ 30871.78 |
| Target hit | 2026-05-05 15:20:00 | 30840.00 | 30881.22 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:15:00 | 32450.00 | 2026-02-10 11:15:00 | 32330.04 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-10 10:15:00 | 32450.00 | 2026-02-10 15:20:00 | 32155.00 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2026-02-13 09:35:00 | 31360.00 | 2026-02-13 10:15:00 | 31234.72 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-13 09:35:00 | 31360.00 | 2026-02-13 10:30:00 | 31360.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:45:00 | 31585.00 | 2026-02-16 10:50:00 | 31503.67 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-17 10:35:00 | 31420.00 | 2026-02-17 11:00:00 | 31369.23 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2026-02-20 09:45:00 | 31145.00 | 2026-02-20 09:50:00 | 31057.68 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-26 10:00:00 | 31205.00 | 2026-02-26 10:10:00 | 31279.12 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-27 10:40:00 | 30940.00 | 2026-02-27 15:20:00 | 30995.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-03-02 10:45:00 | 30450.00 | 2026-03-02 12:40:00 | 30316.03 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-02 10:45:00 | 30450.00 | 2026-03-02 12:55:00 | 30450.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-09 10:50:00 | 29490.00 | 2026-03-09 12:05:00 | 29591.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 09:40:00 | 29690.00 | 2026-03-17 09:55:00 | 29565.52 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-24 10:50:00 | 27090.00 | 2026-03-24 11:20:00 | 27187.68 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-10 11:05:00 | 28415.00 | 2026-04-10 11:15:00 | 28327.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-22 10:30:00 | 32280.00 | 2026-04-22 11:10:00 | 32360.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-23 10:35:00 | 32765.00 | 2026-04-23 10:50:00 | 32671.18 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-24 10:20:00 | 32205.00 | 2026-04-24 11:20:00 | 31987.78 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-04-24 10:20:00 | 32205.00 | 2026-04-24 15:20:00 | 31460.00 | TARGET_HIT | 0.50 | 2.31% |
| SELL | retest1 | 2026-04-29 10:20:00 | 31205.00 | 2026-04-29 10:45:00 | 31081.77 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-29 10:20:00 | 31205.00 | 2026-04-29 10:50:00 | 31205.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:10:00 | 31080.00 | 2026-04-30 10:55:00 | 30961.12 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-30 10:10:00 | 31080.00 | 2026-04-30 11:25:00 | 31080.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:55:00 | 30995.00 | 2026-05-05 13:35:00 | 30871.78 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-05-05 10:55:00 | 30995.00 | 2026-05-05 15:20:00 | 30840.00 | TARGET_HIT | 0.50 | 0.50% |
