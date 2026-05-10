# Mangalore Refinery & Petrochemicals Ltd. (MRPL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 167.97
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 2
- **Avg / median % per leg:** 0.17% / -0.35%
- **Sum % (uncompounded):** 1.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.20% | -1.4% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.20% | -1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 1.50% | 3.0% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 1.50% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 3 | 33.3% | 1 | 6 | 2 | 0.17% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:45:00 | 196.31 | 194.94 | 0.00 | ORB-long ORB[193.70,195.77] vol=2.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-02-11 09:50:00 | 195.51 | 195.17 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:30:00 | 195.60 | 193.91 | 0.00 | ORB-long ORB[192.60,194.89] vol=2.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-18 10:40:00 | 194.78 | 194.04 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 195.69 | 193.82 | 0.00 | ORB-long ORB[192.50,194.95] vol=1.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-02-20 09:40:00 | 194.64 | 194.34 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 194.81 | 193.66 | 0.00 | ORB-long ORB[191.50,194.00] vol=4.7x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 193.94 | 193.82 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:10:00 | 178.19 | 176.99 | 0.00 | ORB-long ORB[175.83,177.96] vol=3.7x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:35:00 | 179.51 | 177.64 | 0.00 | T1 1.5R @ 179.51 |
| Stop hit — per-position SL triggered | 2026-04-16 11:20:00 | 178.19 | 177.80 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 178.42 | 176.79 | 0.00 | ORB-long ORB[175.30,177.84] vol=4.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-04-17 11:20:00 | 177.79 | 177.31 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:50:00 | 171.81 | 173.41 | 0.00 | ORB-short ORB[173.00,175.55] vol=1.5x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:55:00 | 170.64 | 173.18 | 0.00 | T1 1.5R @ 170.64 |
| Target hit | 2026-04-30 15:20:00 | 167.83 | 169.26 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:45:00 | 196.31 | 2026-02-11 09:50:00 | 195.51 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-02-18 10:30:00 | 195.60 | 2026-02-18 10:40:00 | 194.78 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-20 09:30:00 | 195.69 | 2026-02-20 09:40:00 | 194.64 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-02-24 09:35:00 | 194.81 | 2026-02-24 09:40:00 | 193.94 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-04-16 10:10:00 | 178.19 | 2026-04-16 10:35:00 | 179.51 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-04-16 10:10:00 | 178.19 | 2026-04-16 11:20:00 | 178.19 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 11:00:00 | 178.42 | 2026-04-17 11:20:00 | 177.79 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-30 09:50:00 | 171.81 | 2026-04-30 09:55:00 | 170.64 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-30 09:50:00 | 171.81 | 2026-04-30 15:20:00 | 167.83 | TARGET_HIT | 0.50 | 2.32% |
