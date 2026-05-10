# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 243.50
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 2
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 0.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.02% | 0.1% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.02% | 0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.11% | 0.4% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.11% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.07% | 0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:40:00 | 187.10 | 187.92 | 0.00 | ORB-short ORB[187.50,190.00] vol=2.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2026-02-13 10:45:00 | 187.99 | 188.03 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 187.33 | 185.80 | 0.00 | ORB-long ORB[184.51,186.44] vol=1.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-02-17 09:45:00 | 186.60 | 185.92 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:50:00 | 177.21 | 177.32 | 0.00 | ORB-short ORB[177.51,178.72] vol=3.0x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 176.12 | 177.10 | 0.00 | T1 1.5R @ 176.12 |
| Target hit | 2026-02-24 15:20:00 | 176.00 | 175.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 221.21 | 222.45 | 0.00 | ORB-short ORB[221.65,224.85] vol=1.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-04-17 09:35:00 | 222.07 | 222.00 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 246.01 | 243.06 | 0.00 | ORB-long ORB[239.90,243.39] vol=4.3x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:05:00 | 247.14 | 243.68 | 0.00 | T1 1.5R @ 247.14 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 246.01 | 243.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 10:40:00 | 187.10 | 2026-02-13 10:45:00 | 187.99 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-17 09:40:00 | 187.33 | 2026-02-17 09:45:00 | 186.60 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-24 09:50:00 | 177.21 | 2026-02-24 11:45:00 | 176.12 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-24 09:50:00 | 177.21 | 2026-02-24 15:20:00 | 176.00 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2026-04-17 09:30:00 | 221.21 | 2026-04-17 09:35:00 | 222.07 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-08 11:00:00 | 246.01 | 2026-05-08 11:05:00 | 247.14 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-05-08 11:00:00 | 246.01 | 2026-05-08 11:15:00 | 246.01 | STOP_HIT | 0.50 | 0.00% |
