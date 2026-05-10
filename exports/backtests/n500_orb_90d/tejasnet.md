# Tejas Networks Ltd. (TEJASNET)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 515.50
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 2
- **Avg / median % per leg:** 0.48% / 0.78%
- **Sum % (uncompounded):** 2.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.48% | 2.4% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.48% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 1 | 2 | 2 | 0.48% | 2.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 450.05 | 444.47 | 0.00 | ORB-long ORB[440.50,447.00] vol=2.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:20:00 | 453.57 | 447.68 | 0.00 | T1 1.5R @ 453.57 |
| Stop hit — per-position SL triggered | 2026-04-10 10:45:00 | 450.05 | 451.56 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 423.00 | 414.87 | 0.00 | ORB-long ORB[412.05,415.95] vol=8.3x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-04-29 11:00:00 | 421.12 | 416.08 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-05-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:35:00 | 420.15 | 418.00 | 0.00 | ORB-long ORB[414.50,419.10] vol=3.9x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:10:00 | 423.76 | 419.41 | 0.00 | T1 1.5R @ 423.76 |
| Target hit | 2026-05-04 11:30:00 | 425.15 | 425.63 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-04-10 10:15:00 | 450.05 | 2026-04-10 10:20:00 | 453.57 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-04-10 10:15:00 | 450.05 | 2026-04-10 10:45:00 | 450.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:55:00 | 423.00 | 2026-04-29 11:00:00 | 421.12 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-05-04 09:35:00 | 420.15 | 2026-05-04 10:10:00 | 423.76 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2026-05-04 09:35:00 | 420.15 | 2026-05-04 11:30:00 | 425.15 | TARGET_HIT | 0.50 | 1.19% |
