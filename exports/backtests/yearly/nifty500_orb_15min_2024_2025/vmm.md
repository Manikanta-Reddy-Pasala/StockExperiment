# Vishal Mega Mart Ltd. (VMM)

## Backtest Summary

- **Window:** 2025-04-07 09:15:00 → 2026-05-08 15:25:00 (20038 bars)
- **Last close:** 124.00
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 3
- **Avg / median % per leg:** 0.76% / 0.54%
- **Sum % (uncompounded):** 4.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 1.00% | 4.0% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 1.00% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.28% | 0.6% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.28% | 0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 2 | 3 | 0.76% | 4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:40:00 | 106.11 | 105.66 | 0.00 | ORB-long ORB[104.95,105.89] vol=1.8x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:45:00 | 106.67 | 105.81 | 0.00 | T1 1.5R @ 106.67 |
| Stop hit — per-position SL triggered | 2025-04-15 09:50:00 | 106.11 | 105.83 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:05:00 | 110.01 | 109.16 | 0.00 | ORB-long ORB[108.15,109.75] vol=1.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 12:25:00 | 110.61 | 109.69 | 0.00 | T1 1.5R @ 110.61 |
| Target hit | 2025-04-21 15:20:00 | 113.25 | 111.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:00:00 | 110.90 | 112.26 | 0.00 | ORB-short ORB[112.70,113.50] vol=3.0x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:10:00 | 110.27 | 112.05 | 0.00 | T1 1.5R @ 110.27 |
| Stop hit — per-position SL triggered | 2025-04-23 10:20:00 | 110.90 | 111.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-04-15 09:40:00 | 106.11 | 2025-04-15 09:45:00 | 106.67 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-04-15 09:40:00 | 106.11 | 2025-04-15 09:50:00 | 106.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 10:05:00 | 110.01 | 2025-04-21 12:25:00 | 110.61 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-04-21 10:05:00 | 110.01 | 2025-04-21 15:20:00 | 113.25 | TARGET_HIT | 0.50 | 2.95% |
| SELL | retest1 | 2025-04-23 10:00:00 | 110.90 | 2025-04-23 10:10:00 | 110.27 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-23 10:00:00 | 110.90 | 2025-04-23 10:20:00 | 110.90 | STOP_HIT | 0.50 | 0.00% |
