# Polycab India Ltd. (POLYCAB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 9080.00
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 4
- **Avg / median % per leg:** -0.02% / 0.00%
- **Sum % (uncompounded):** -0.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.07% | 0.4% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.07% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.07% | -0.7% |
| SELL @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 1 | 6 | 2 | -0.07% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 5 | 33.3% | 1 | 10 | 4 | -0.02% | -0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:15:00 | 7656.00 | 7629.24 | 0.00 | ORB-long ORB[7536.00,7625.00] vol=2.7x ATR=14.82 |
| Stop hit — per-position SL triggered | 2026-02-16 13:45:00 | 7641.18 | 7640.62 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 7755.00 | 7781.85 | 0.00 | ORB-short ORB[7778.00,7848.50] vol=2.9x ATR=17.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:45:00 | 7728.84 | 7761.61 | 0.00 | T1 1.5R @ 7728.84 |
| Stop hit — per-position SL triggered | 2026-02-19 09:55:00 | 7755.00 | 7760.78 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:20:00 | 7737.50 | 7757.09 | 0.00 | ORB-short ORB[7743.00,7832.50] vol=1.5x ATR=27.77 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 7765.27 | 7745.37 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 8336.00 | 8390.10 | 0.00 | ORB-short ORB[8360.00,8475.00] vol=1.5x ATR=37.73 |
| Stop hit — per-position SL triggered | 2026-03-04 10:05:00 | 8373.73 | 8388.29 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:20:00 | 7314.50 | 7269.95 | 0.00 | ORB-long ORB[7176.50,7279.00] vol=2.2x ATR=27.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:10:00 | 7355.22 | 7293.52 | 0.00 | T1 1.5R @ 7355.22 |
| Stop hit — per-position SL triggered | 2026-03-18 12:05:00 | 7314.50 | 7316.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:40:00 | 7192.00 | 7249.74 | 0.00 | ORB-short ORB[7221.00,7288.00] vol=1.9x ATR=30.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:45:00 | 7146.01 | 7231.76 | 0.00 | T1 1.5R @ 7146.01 |
| Target hit | 2026-03-19 12:10:00 | 7185.00 | 7181.88 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2026-03-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 11:10:00 | 6895.00 | 6962.71 | 0.00 | ORB-short ORB[6935.00,7017.50] vol=1.7x ATR=21.57 |
| Stop hit — per-position SL triggered | 2026-03-30 11:35:00 | 6916.57 | 6954.74 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:30:00 | 7114.00 | 7022.71 | 0.00 | ORB-long ORB[6942.50,7035.50] vol=1.7x ATR=27.42 |
| Stop hit — per-position SL triggered | 2026-04-07 10:55:00 | 7086.58 | 7041.42 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 8020.00 | 7962.41 | 0.00 | ORB-long ORB[7897.00,7960.00] vol=2.4x ATR=22.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:15:00 | 8053.23 | 7990.78 | 0.00 | T1 1.5R @ 8053.23 |
| Stop hit — per-position SL triggered | 2026-04-22 14:05:00 | 8020.00 | 8004.65 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 8220.00 | 8279.68 | 0.00 | ORB-short ORB[8268.00,8338.50] vol=2.4x ATR=27.56 |
| Stop hit — per-position SL triggered | 2026-04-29 09:40:00 | 8247.56 | 8271.54 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:05:00 | 8001.00 | 8038.45 | 0.00 | ORB-short ORB[8040.50,8133.00] vol=1.5x ATR=21.47 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 8022.47 | 8034.40 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 11:15:00 | 7656.00 | 2026-02-16 13:45:00 | 7641.18 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-19 09:30:00 | 7755.00 | 2026-02-19 09:45:00 | 7728.84 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-19 09:30:00 | 7755.00 | 2026-02-19 09:55:00 | 7755.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-20 10:20:00 | 7737.50 | 2026-02-20 11:30:00 | 7765.27 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-04 09:55:00 | 8336.00 | 2026-03-04 10:05:00 | 8373.73 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-18 10:20:00 | 7314.50 | 2026-03-18 11:10:00 | 7355.22 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-18 10:20:00 | 7314.50 | 2026-03-18 12:05:00 | 7314.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 09:40:00 | 7192.00 | 2026-03-19 09:45:00 | 7146.01 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-03-19 09:40:00 | 7192.00 | 2026-03-19 12:10:00 | 7185.00 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2026-03-30 11:10:00 | 6895.00 | 2026-03-30 11:35:00 | 6916.57 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-07 10:30:00 | 7114.00 | 2026-04-07 10:55:00 | 7086.58 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-22 10:55:00 | 8020.00 | 2026-04-22 12:15:00 | 8053.23 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-22 10:55:00 | 8020.00 | 2026-04-22 14:05:00 | 8020.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:35:00 | 8220.00 | 2026-04-29 09:40:00 | 8247.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-30 11:05:00 | 8001.00 | 2026-04-30 11:25:00 | 8022.47 | STOP_HIT | 1.00 | -0.27% |
