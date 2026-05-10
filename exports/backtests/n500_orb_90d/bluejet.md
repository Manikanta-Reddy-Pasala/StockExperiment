# Blue Jet Healthcare Ltd. (BLUEJET)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 491.00
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.20% | 0.6% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.20% | 0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.54% | -0.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.54% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 3 | 1 | 0.02% | 0.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:50:00 | 375.00 | 365.68 | 0.00 | ORB-long ORB[359.00,363.70] vol=5.3x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:55:00 | 378.79 | 374.88 | 0.00 | T1 1.5R @ 378.79 |
| Stop hit — per-position SL triggered | 2026-02-19 10:10:00 | 375.00 | 378.35 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 409.10 | 413.63 | 0.00 | ORB-short ORB[412.85,418.20] vol=3.1x ATR=2.20 |
| Stop hit — per-position SL triggered | 2026-04-16 10:10:00 | 411.30 | 412.40 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-05-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:30:00 | 498.70 | 494.14 | 0.00 | ORB-long ORB[489.55,495.20] vol=2.9x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-05-08 10:40:00 | 496.70 | 495.05 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-19 09:50:00 | 375.00 | 2026-02-19 09:55:00 | 378.79 | PARTIAL | 0.50 | 1.01% |
| BUY | retest1 | 2026-02-19 09:50:00 | 375.00 | 2026-02-19 10:10:00 | 375.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:40:00 | 409.10 | 2026-04-16 10:10:00 | 411.30 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2026-05-08 10:30:00 | 498.70 | 2026-05-08 10:40:00 | 496.70 | STOP_HIT | 1.00 | -0.40% |
