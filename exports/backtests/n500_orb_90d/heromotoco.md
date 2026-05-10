# Hero MotoCorp Ltd. (HEROMOTOCO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5325.00
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
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 8
- **Target hits / Stop hits / Partials:** 7 / 8 / 9
- **Avg / median % per leg:** 0.40% / 0.45%
- **Sum % (uncompounded):** 9.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.04% | 0.5% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.04% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 11 | 91.7% | 5 | 1 | 6 | 0.77% | 9.2% |
| SELL @ 2nd Alert (retest1) | 12 | 11 | 91.7% | 5 | 1 | 6 | 0.77% | 9.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 16 | 66.7% | 7 | 8 | 9 | 0.40% | 9.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:05:00 | 5621.00 | 5659.88 | 0.00 | ORB-short ORB[5657.00,5724.50] vol=2.7x ATR=12.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:30:00 | 5601.72 | 5654.36 | 0.00 | T1 1.5R @ 5601.72 |
| Stop hit — per-position SL triggered | 2026-02-13 11:45:00 | 5621.00 | 5652.08 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:35:00 | 5528.50 | 5500.49 | 0.00 | ORB-long ORB[5446.50,5499.00] vol=1.5x ATR=14.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:50:00 | 5550.54 | 5517.68 | 0.00 | T1 1.5R @ 5550.54 |
| Target hit | 2026-02-17 15:20:00 | 5574.50 | 5534.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 5619.50 | 5604.59 | 0.00 | ORB-long ORB[5569.50,5615.00] vol=2.0x ATR=13.38 |
| Stop hit — per-position SL triggered | 2026-02-18 09:50:00 | 5606.12 | 5609.83 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 5710.50 | 5619.44 | 0.00 | ORB-long ORB[5500.50,5558.00] vol=1.5x ATR=17.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:00:00 | 5736.37 | 5642.49 | 0.00 | T1 1.5R @ 5736.37 |
| Target hit | 2026-02-25 13:00:00 | 5728.50 | 5730.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-03-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:45:00 | 5653.00 | 5581.93 | 0.00 | ORB-long ORB[5527.50,5609.50] vol=2.0x ATR=21.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 11:00:00 | 5684.86 | 5603.69 | 0.00 | T1 1.5R @ 5684.86 |
| Stop hit — per-position SL triggered | 2026-03-10 12:10:00 | 5653.00 | 5634.16 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 5688.50 | 5713.24 | 0.00 | ORB-short ORB[5700.50,5766.00] vol=2.0x ATR=17.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:40:00 | 5662.39 | 5701.86 | 0.00 | T1 1.5R @ 5662.39 |
| Target hit | 2026-03-11 15:20:00 | 5580.00 | 5627.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:40:00 | 5286.00 | 5329.94 | 0.00 | ORB-short ORB[5322.00,5377.50] vol=1.8x ATR=20.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:50:00 | 5255.92 | 5314.94 | 0.00 | T1 1.5R @ 5255.92 |
| Target hit | 2026-03-13 11:35:00 | 5247.00 | 5246.24 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2026-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:00:00 | 5344.50 | 5298.24 | 0.00 | ORB-long ORB[5216.00,5294.50] vol=1.5x ATR=18.57 |
| Stop hit — per-position SL triggered | 2026-03-20 11:50:00 | 5325.93 | 5309.02 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:00:00 | 5320.50 | 5298.08 | 0.00 | ORB-long ORB[5243.00,5313.50] vol=2.0x ATR=26.12 |
| Stop hit — per-position SL triggered | 2026-04-08 13:45:00 | 5294.38 | 5312.41 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 11:00:00 | 5282.50 | 5333.40 | 0.00 | ORB-short ORB[5295.00,5369.00] vol=1.9x ATR=20.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 12:30:00 | 5252.05 | 5308.40 | 0.00 | T1 1.5R @ 5252.05 |
| Target hit | 2026-04-13 15:20:00 | 5245.50 | 5273.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 5266.00 | 5304.33 | 0.00 | ORB-short ORB[5285.00,5349.00] vol=1.7x ATR=15.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:35:00 | 5242.35 | 5293.79 | 0.00 | T1 1.5R @ 5242.35 |
| Target hit | 2026-04-16 15:20:00 | 5167.50 | 5192.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:15:00 | 5095.50 | 5122.60 | 0.00 | ORB-short ORB[5102.00,5160.00] vol=1.8x ATR=14.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:55:00 | 5074.10 | 5112.24 | 0.00 | T1 1.5R @ 5074.10 |
| Target hit | 2026-04-23 15:20:00 | 5034.00 | 5048.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:25:00 | 5087.00 | 5081.27 | 0.00 | ORB-long ORB[5019.00,5083.00] vol=3.5x ATR=12.28 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 5074.72 | 5081.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 5222.00 | 5201.70 | 0.00 | ORB-long ORB[5175.00,5214.00] vol=1.7x ATR=21.12 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 5200.88 | 5202.87 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:20:00 | 5386.00 | 5349.74 | 0.00 | ORB-long ORB[5265.00,5341.00] vol=1.9x ATR=19.90 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 5366.10 | 5355.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 11:05:00 | 5621.00 | 2026-02-13 11:30:00 | 5601.72 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-13 11:05:00 | 5621.00 | 2026-02-13 11:45:00 | 5621.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:35:00 | 5528.50 | 2026-02-17 12:50:00 | 5550.54 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-17 10:35:00 | 5528.50 | 2026-02-17 15:20:00 | 5574.50 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2026-02-18 09:30:00 | 5619.50 | 2026-02-18 09:50:00 | 5606.12 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:40:00 | 5710.50 | 2026-02-25 11:00:00 | 5736.37 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-25 10:40:00 | 5710.50 | 2026-02-25 13:00:00 | 5728.50 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-03-10 10:45:00 | 5653.00 | 2026-03-10 11:00:00 | 5684.86 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-10 10:45:00 | 5653.00 | 2026-03-10 12:10:00 | 5653.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:25:00 | 5688.50 | 2026-03-11 10:40:00 | 5662.39 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-11 10:25:00 | 5688.50 | 2026-03-11 15:20:00 | 5580.00 | TARGET_HIT | 0.50 | 1.91% |
| SELL | retest1 | 2026-03-13 09:40:00 | 5286.00 | 2026-03-13 09:50:00 | 5255.92 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 09:40:00 | 5286.00 | 2026-03-13 11:35:00 | 5247.00 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-20 11:00:00 | 5344.50 | 2026-03-20 11:50:00 | 5325.93 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-08 10:00:00 | 5320.50 | 2026-04-08 13:45:00 | 5294.38 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-13 11:00:00 | 5282.50 | 2026-04-13 12:30:00 | 5252.05 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-13 11:00:00 | 5282.50 | 2026-04-13 15:20:00 | 5245.50 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-16 09:30:00 | 5266.00 | 2026-04-16 09:35:00 | 5242.35 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-04-16 09:30:00 | 5266.00 | 2026-04-16 15:20:00 | 5167.50 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2026-04-23 10:15:00 | 5095.50 | 2026-04-23 10:55:00 | 5074.10 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-23 10:15:00 | 5095.50 | 2026-04-23 15:20:00 | 5034.00 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2026-04-28 10:25:00 | 5087.00 | 2026-04-28 10:35:00 | 5074.72 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-04 09:35:00 | 5222.00 | 2026-05-04 09:50:00 | 5200.88 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-08 10:20:00 | 5386.00 | 2026-05-08 11:15:00 | 5366.10 | STOP_HIT | 1.00 | -0.37% |
