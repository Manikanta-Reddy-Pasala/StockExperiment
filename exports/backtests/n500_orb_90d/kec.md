# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 597.80
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 4
- **Avg / median % per leg:** 0.37% / 0.44%
- **Sum % (uncompounded):** 5.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 6 | 60.0% | 3 | 4 | 3 | 0.56% | 5.6% |
| SELL @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 3 | 4 | 3 | 0.56% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.37% | 5.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 626.80 | 624.69 | 0.00 | ORB-long ORB[616.60,621.80] vol=1.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-02-09 11:30:00 | 623.87 | 624.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 609.75 | 605.71 | 0.00 | ORB-long ORB[601.15,605.70] vol=1.8x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:55:00 | 612.45 | 607.17 | 0.00 | T1 1.5R @ 612.45 |
| Stop hit — per-position SL triggered | 2026-02-17 10:35:00 | 609.75 | 608.05 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:15:00 | 600.10 | 605.62 | 0.00 | ORB-short ORB[606.25,614.75] vol=2.4x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 601.76 | 604.18 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 577.15 | 579.48 | 0.00 | ORB-short ORB[577.30,584.95] vol=1.5x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-02-24 09:35:00 | 579.02 | 579.43 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 597.90 | 594.99 | 0.00 | ORB-long ORB[591.50,596.05] vol=1.8x ATR=2.08 |
| Stop hit — per-position SL triggered | 2026-02-26 11:45:00 | 595.82 | 597.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:05:00 | 586.20 | 588.27 | 0.00 | ORB-short ORB[588.20,594.80] vol=1.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-27 11:10:00 | 587.42 | 588.14 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:00:00 | 530.60 | 533.98 | 0.00 | ORB-short ORB[535.00,540.30] vol=2.2x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:10:00 | 527.12 | 533.21 | 0.00 | T1 1.5R @ 527.12 |
| Target hit | 2026-03-23 15:20:00 | 514.05 | 523.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 570.10 | 573.60 | 0.00 | ORB-short ORB[570.60,578.40] vol=2.1x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 14:30:00 | 566.27 | 569.87 | 0.00 | T1 1.5R @ 566.27 |
| Target hit | 2026-04-15 15:20:00 | 565.00 | 569.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 572.30 | 575.68 | 0.00 | ORB-short ORB[572.50,577.70] vol=1.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-04-29 10:20:00 | 574.16 | 574.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 601.95 | 605.98 | 0.00 | ORB-short ORB[606.85,615.00] vol=1.6x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 15:05:00 | 597.76 | 603.95 | 0.00 | T1 1.5R @ 597.76 |
| Target hit | 2026-05-08 15:20:00 | 597.70 | 603.48 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:40:00 | 626.80 | 2026-02-09 11:30:00 | 623.87 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-17 09:40:00 | 609.75 | 2026-02-17 09:55:00 | 612.45 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-17 09:40:00 | 609.75 | 2026-02-17 10:35:00 | 609.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:15:00 | 600.10 | 2026-02-18 10:50:00 | 601.76 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-24 09:30:00 | 577.15 | 2026-02-24 09:35:00 | 579.02 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 09:40:00 | 597.90 | 2026-02-26 11:45:00 | 595.82 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 11:05:00 | 586.20 | 2026-02-27 11:10:00 | 587.42 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-23 10:00:00 | 530.60 | 2026-03-23 10:10:00 | 527.12 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-23 10:00:00 | 530.60 | 2026-03-23 15:20:00 | 514.05 | TARGET_HIT | 0.50 | 3.12% |
| SELL | retest1 | 2026-04-15 09:35:00 | 570.10 | 2026-04-15 14:30:00 | 566.27 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2026-04-15 09:35:00 | 570.10 | 2026-04-15 15:20:00 | 565.00 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2026-04-29 09:55:00 | 572.30 | 2026-04-29 10:20:00 | 574.16 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-08 10:50:00 | 601.95 | 2026-05-08 15:05:00 | 597.76 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-05-08 10:50:00 | 601.95 | 2026-05-08 15:20:00 | 597.70 | TARGET_HIT | 0.50 | 0.71% |
