# Vedanta Ltd. (VEDL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4416 bars)
- **Last close:** 297.00
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
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 4
- **Avg / median % per leg:** 0.48% / 0.49%
- **Sum % (uncompounded):** 4.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.29% | 1.1% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 2 | 2 | 0.29% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.64% | 3.2% |
| SELL @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.64% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 5 | 55.6% | 1 | 4 | 4 | 0.48% | 4.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:15:00 | 257.10 | 255.52 | 0.00 | ORB-long ORB[253.93,256.93] vol=1.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 12:10:00 | 258.36 | 256.29 | 0.00 | T1 1.5R @ 258.36 |
| Stop hit — per-position SL triggered | 2026-02-10 13:30:00 | 257.10 | 256.69 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 253.20 | 255.25 | 0.00 | ORB-short ORB[255.21,257.98] vol=1.6x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-02-19 12:20:00 | 253.81 | 254.81 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:05:00 | 270.62 | 271.84 | 0.00 | ORB-short ORB[271.54,275.28] vol=1.7x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 11:35:00 | 269.03 | 271.56 | 0.00 | T1 1.5R @ 269.03 |
| Stop hit — per-position SL triggered | 2026-04-08 12:25:00 | 270.62 | 271.40 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:30:00 | 293.43 | 294.84 | 0.00 | ORB-short ORB[293.63,297.75] vol=2.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:40:00 | 291.64 | 294.51 | 0.00 | T1 1.5R @ 291.64 |
| Target hit | 2026-04-21 15:20:00 | 286.85 | 290.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-04-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:45:00 | 275.84 | 272.26 | 0.00 | ORB-long ORB[269.87,272.58] vol=2.1x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:55:00 | 277.65 | 273.68 | 0.00 | T1 1.5R @ 277.65 |
| Stop hit — per-position SL triggered | 2026-04-27 10:55:00 | 275.84 | 274.90 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:15:00 | 257.10 | 2026-02-10 12:10:00 | 258.36 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-10 10:15:00 | 257.10 | 2026-02-10 13:30:00 | 257.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 253.20 | 2026-02-19 12:20:00 | 253.81 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-04-08 11:05:00 | 270.62 | 2026-04-08 11:35:00 | 269.03 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-08 11:05:00 | 270.62 | 2026-04-08 12:25:00 | 270.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 09:30:00 | 293.43 | 2026-04-21 09:40:00 | 291.64 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-21 09:30:00 | 293.43 | 2026-04-21 15:20:00 | 286.85 | TARGET_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2026-04-27 09:45:00 | 275.84 | 2026-04-27 09:55:00 | 277.65 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-27 09:45:00 | 275.84 | 2026-04-27 10:55:00 | 275.84 | STOP_HIT | 0.50 | 0.00% |
