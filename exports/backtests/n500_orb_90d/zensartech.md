# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 525.00
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 3
- **Avg / median % per leg:** -0.03% / 0.00%
- **Sum % (uncompounded):** -0.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | -0.01% | -0.0% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | -0.01% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.04% | -0.3% |
| SELL @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.04% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 3 | 27.3% | 0 | 8 | 3 | -0.03% | -0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:55:00 | 614.50 | 621.94 | 0.00 | ORB-short ORB[623.00,629.50] vol=1.9x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-02-11 10:45:00 | 616.76 | 619.57 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 575.15 | 580.70 | 0.00 | ORB-short ORB[579.45,587.20] vol=1.8x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:25:00 | 571.36 | 577.65 | 0.00 | T1 1.5R @ 571.36 |
| Stop hit — per-position SL triggered | 2026-02-18 13:05:00 | 575.15 | 573.68 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 09:30:00 | 555.85 | 558.14 | 0.00 | ORB-short ORB[556.00,561.10] vol=2.0x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:35:00 | 551.58 | 557.10 | 0.00 | T1 1.5R @ 551.58 |
| Stop hit — per-position SL triggered | 2026-03-05 09:40:00 | 555.85 | 556.79 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:10:00 | 564.60 | 561.34 | 0.00 | ORB-long ORB[557.50,564.00] vol=1.7x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:55:00 | 567.30 | 562.10 | 0.00 | T1 1.5R @ 567.30 |
| Stop hit — per-position SL triggered | 2026-03-12 12:05:00 | 564.60 | 562.26 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 555.05 | 559.29 | 0.00 | ORB-short ORB[556.50,562.30] vol=1.6x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 558.16 | 556.08 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:50:00 | 559.00 | 551.29 | 0.00 | ORB-long ORB[544.00,550.95] vol=3.1x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-04-06 11:10:00 | 556.23 | 552.96 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:40:00 | 546.70 | 551.98 | 0.00 | ORB-short ORB[551.90,556.00] vol=2.2x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-04-09 11:35:00 | 548.64 | 550.68 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:30:00 | 524.55 | 527.37 | 0.00 | ORB-short ORB[525.20,532.45] vol=1.9x ATR=2.41 |
| Stop hit — per-position SL triggered | 2026-04-29 09:40:00 | 526.96 | 526.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:55:00 | 614.50 | 2026-02-11 10:45:00 | 616.76 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 09:50:00 | 575.15 | 2026-02-18 10:25:00 | 571.36 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-18 09:50:00 | 575.15 | 2026-02-18 13:05:00 | 575.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 09:30:00 | 555.85 | 2026-03-05 09:35:00 | 551.58 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2026-03-05 09:30:00 | 555.85 | 2026-03-05 09:40:00 | 555.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-12 11:10:00 | 564.60 | 2026-03-12 11:55:00 | 567.30 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-03-12 11:10:00 | 564.60 | 2026-03-12 12:05:00 | 564.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:35:00 | 555.05 | 2026-03-13 11:25:00 | 558.16 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2026-04-06 10:50:00 | 559.00 | 2026-04-06 11:10:00 | 556.23 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-09 09:40:00 | 546.70 | 2026-04-09 11:35:00 | 548.64 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-29 09:30:00 | 524.55 | 2026-04-29 09:40:00 | 526.96 | STOP_HIT | 1.00 | -0.46% |
