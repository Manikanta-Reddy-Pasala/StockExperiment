# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 5115.00
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / Stop hits / Partials:** 2 / 6 / 4
- **Avg / median % per leg:** 0.49% / 0.44%
- **Sum % (uncompounded):** 5.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.06% | -0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.77% | 6.1% |
| SELL @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 2 | 3 | 3 | 0.77% | 6.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.49% | 5.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 5802.00 | 5845.17 | 0.00 | ORB-short ORB[5834.50,5900.00] vol=2.7x ATR=19.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 5772.12 | 5825.67 | 0.00 | T1 1.5R @ 5772.12 |
| Target hit | 2026-02-11 13:40:00 | 5752.50 | 5751.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 5499.50 | 5534.33 | 0.00 | ORB-short ORB[5520.50,5597.00] vol=1.7x ATR=19.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 5469.74 | 5507.52 | 0.00 | T1 1.5R @ 5469.74 |
| Target hit | 2026-02-19 15:20:00 | 5251.00 | 5356.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:15:00 | 4809.00 | 4825.91 | 0.00 | ORB-short ORB[4830.20,4895.00] vol=3.5x ATR=17.64 |
| Stop hit — per-position SL triggered | 2026-03-11 10:30:00 | 4826.64 | 4824.61 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:05:00 | 4596.00 | 4627.43 | 0.00 | ORB-short ORB[4597.00,4659.40] vol=2.0x ATR=17.31 |
| Stop hit — per-position SL triggered | 2026-03-19 12:25:00 | 4613.31 | 4618.68 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 10:55:00 | 5092.40 | 5028.25 | 0.00 | ORB-long ORB[4963.90,5034.90] vol=2.2x ATR=21.31 |
| Stop hit — per-position SL triggered | 2026-04-02 11:05:00 | 5071.09 | 5031.82 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 5113.80 | 5054.91 | 0.00 | ORB-long ORB[5011.00,5083.00] vol=1.9x ATR=15.62 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 5098.18 | 5057.96 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 4877.00 | 4839.48 | 0.00 | ORB-long ORB[4799.20,4852.00] vol=1.5x ATR=16.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 4901.48 | 4851.09 | 0.00 | T1 1.5R @ 4901.48 |
| Stop hit — per-position SL triggered | 2026-04-29 09:55:00 | 4877.00 | 4852.35 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:50:00 | 4936.40 | 4991.73 | 0.00 | ORB-short ORB[5004.10,5074.40] vol=1.6x ATR=14.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:25:00 | 4914.71 | 4979.84 | 0.00 | T1 1.5R @ 4914.71 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 4936.40 | 4978.83 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 5802.00 | 2026-02-11 09:45:00 | 5772.12 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-11 09:30:00 | 5802.00 | 2026-02-11 13:40:00 | 5752.50 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2026-02-19 09:35:00 | 5499.50 | 2026-02-19 09:50:00 | 5469.74 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-19 09:35:00 | 5499.50 | 2026-02-19 15:20:00 | 5251.00 | TARGET_HIT | 0.50 | 4.52% |
| SELL | retest1 | 2026-03-11 10:15:00 | 4809.00 | 2026-03-11 10:30:00 | 4826.64 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-19 11:05:00 | 4596.00 | 2026-03-19 12:25:00 | 4613.31 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-02 10:55:00 | 5092.40 | 2026-04-02 11:05:00 | 5071.09 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-23 11:00:00 | 5113.80 | 2026-04-23 11:10:00 | 5098.18 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-29 09:45:00 | 4877.00 | 2026-04-29 09:50:00 | 4901.48 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-29 09:45:00 | 4877.00 | 2026-04-29 09:55:00 | 4877.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-07 10:50:00 | 4936.40 | 2026-05-07 11:25:00 | 4914.71 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-07 10:50:00 | 4936.40 | 2026-05-07 11:30:00 | 4936.40 | STOP_HIT | 0.50 | 0.00% |
