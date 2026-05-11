# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2023-06-09 15:25:00 (1575 bars)
- **Last close:** 625.00
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 6 / 1
- **Avg / median % per leg:** -0.17% / -0.29%
- **Sum % (uncompounded):** -1.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.17% | -1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:40:00 | 631.35 | 626.38 | 0.00 | ORB-long ORB[623.25,629.00] vol=1.5x ATR=2.44 |
| Stop hit — per-position SL triggered | 2023-05-15 10:00:00 | 628.91 | 627.39 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:00:00 | 630.50 | 628.29 | 0.00 | ORB-long ORB[624.10,628.95] vol=2.2x ATR=1.49 |
| Stop hit — per-position SL triggered | 2023-05-16 10:10:00 | 629.01 | 628.46 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 09:45:00 | 639.00 | 633.63 | 0.00 | ORB-long ORB[627.00,633.55] vol=7.1x ATR=2.27 |
| Stop hit — per-position SL triggered | 2023-05-17 09:50:00 | 636.73 | 634.29 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:30:00 | 612.45 | 609.17 | 0.00 | ORB-long ORB[605.00,610.00] vol=3.3x ATR=1.87 |
| Stop hit — per-position SL triggered | 2023-05-29 09:35:00 | 610.58 | 609.39 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:10:00 | 610.80 | 607.33 | 0.00 | ORB-long ORB[601.30,609.15] vol=1.7x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-05-30 10:40:00 | 609.00 | 607.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:20:00 | 612.00 | 607.37 | 0.00 | ORB-long ORB[604.85,611.25] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 10:50:00 | 614.23 | 609.57 | 0.00 | T1 1.5R @ 614.23 |
| Stop hit — per-position SL triggered | 2023-05-31 15:00:00 | 612.00 | 612.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:40:00 | 631.35 | 2023-05-15 10:00:00 | 628.91 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-05-16 10:00:00 | 630.50 | 2023-05-16 10:10:00 | 629.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-17 09:45:00 | 639.00 | 2023-05-17 09:50:00 | 636.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-05-29 09:30:00 | 612.45 | 2023-05-29 09:35:00 | 610.58 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-05-30 10:10:00 | 610.80 | 2023-05-30 10:40:00 | 609.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-05-31 10:20:00 | 612.00 | 2023-05-31 10:50:00 | 614.23 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-05-31 10:20:00 | 612.00 | 2023-05-31 15:00:00 | 612.00 | STOP_HIT | 0.50 | 0.00% |
