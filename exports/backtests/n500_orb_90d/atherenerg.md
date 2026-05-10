# Ather Energy Ltd. (ATHERENERG)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 916.90
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 4
- **Avg / median % per leg:** 0.35% / 0.47%
- **Sum % (uncompounded):** 3.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.08% | 0.5% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.08% | 0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.75% | 3.0% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.75% | 3.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 5 | 50.0% | 1 | 5 | 4 | 0.35% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 739.25 | 734.69 | 0.00 | ORB-long ORB[725.45,734.95] vol=5.7x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:40:00 | 744.22 | 736.86 | 0.00 | T1 1.5R @ 744.22 |
| Stop hit — per-position SL triggered | 2026-02-19 09:55:00 | 739.25 | 740.70 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 720.60 | 724.45 | 0.00 | ORB-short ORB[721.70,728.95] vol=1.6x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:10:00 | 715.89 | 723.33 | 0.00 | T1 1.5R @ 715.89 |
| Target hit | 2026-02-20 15:20:00 | 707.05 | 717.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:25:00 | 721.55 | 710.54 | 0.00 | ORB-long ORB[700.05,708.50] vol=1.8x ATR=4.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:00:00 | 727.71 | 715.03 | 0.00 | T1 1.5R @ 727.71 |
| Stop hit — per-position SL triggered | 2026-02-25 11:20:00 | 721.55 | 716.46 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 692.40 | 695.75 | 0.00 | ORB-short ORB[694.30,703.20] vol=1.6x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 689.12 | 695.28 | 0.00 | T1 1.5R @ 689.12 |
| Stop hit — per-position SL triggered | 2026-03-05 14:40:00 | 692.40 | 689.71 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:35:00 | 710.75 | 707.30 | 0.00 | ORB-long ORB[699.65,710.00] vol=1.8x ATR=3.59 |
| Stop hit — per-position SL triggered | 2026-03-13 09:45:00 | 707.16 | 707.55 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-04-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:50:00 | 902.35 | 888.23 | 0.00 | ORB-long ORB[881.85,895.00] vol=1.7x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-04-21 11:20:00 | 897.37 | 891.79 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-19 09:35:00 | 739.25 | 2026-02-19 09:40:00 | 744.22 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-02-19 09:35:00 | 739.25 | 2026-02-19 09:55:00 | 739.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-20 10:45:00 | 720.60 | 2026-02-20 11:10:00 | 715.89 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-02-20 10:45:00 | 720.60 | 2026-02-20 15:20:00 | 707.05 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2026-02-25 10:25:00 | 721.55 | 2026-02-25 11:00:00 | 727.71 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2026-02-25 10:25:00 | 721.55 | 2026-02-25 11:20:00 | 721.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:55:00 | 692.40 | 2026-03-05 11:20:00 | 689.12 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-05 10:55:00 | 692.40 | 2026-03-05 14:40:00 | 692.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-13 09:35:00 | 710.75 | 2026-03-13 09:45:00 | 707.16 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2026-04-21 10:50:00 | 902.35 | 2026-04-21 11:20:00 | 897.37 | STOP_HIT | 1.00 | -0.55% |
