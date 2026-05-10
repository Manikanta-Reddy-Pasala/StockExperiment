# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 494.80
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 2
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 1.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.32% | 1.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.32% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.06% | 0.2% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.06% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.22% | 1.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 429.00 | 431.73 | 0.00 | ORB-short ORB[429.45,435.80] vol=2.1x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 426.49 | 430.96 | 0.00 | T1 1.5R @ 426.49 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 429.00 | 430.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 454.05 | 448.59 | 0.00 | ORB-long ORB[443.20,449.00] vol=2.4x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:05:00 | 457.79 | 463.33 | 0.00 | T1 1.5R @ 457.79 |
| Target hit | 2026-03-13 10:15:00 | 463.95 | 469.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 480.10 | 484.12 | 0.00 | ORB-short ORB[483.10,488.40] vol=2.3x ATR=1.90 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 482.00 | 483.51 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 503.80 | 497.55 | 0.00 | ORB-long ORB[491.20,496.95] vol=5.1x ATR=2.51 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 501.29 | 498.63 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:35:00 | 482.15 | 481.00 | 0.00 | ORB-long ORB[477.85,482.05] vol=1.6x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-04-30 09:45:00 | 479.79 | 480.80 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-05-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:10:00 | 463.30 | 459.70 | 0.00 | ORB-long ORB[456.00,462.20] vol=1.8x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 461.34 | 459.76 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-05 10:50:00 | 429.00 | 2026-03-05 11:20:00 | 426.49 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-05 10:50:00 | 429.00 | 2026-03-05 11:25:00 | 429.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-13 10:00:00 | 454.05 | 2026-03-13 10:05:00 | 457.79 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2026-03-13 10:00:00 | 454.05 | 2026-03-13 10:15:00 | 463.95 | TARGET_HIT | 0.50 | 2.18% |
| SELL | retest1 | 2026-03-20 09:35:00 | 480.10 | 2026-03-20 09:50:00 | 482.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-28 09:40:00 | 503.80 | 2026-04-28 09:45:00 | 501.29 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-30 09:35:00 | 482.15 | 2026-04-30 09:45:00 | 479.79 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-05-06 10:10:00 | 463.30 | 2026-05-06 10:15:00 | 461.34 | STOP_HIT | 1.00 | -0.42% |
