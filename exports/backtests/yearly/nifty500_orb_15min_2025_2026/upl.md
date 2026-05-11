# UPL Ltd. (UPL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-06-09 15:25:00 (1575 bars)
- **Last close:** 639.35
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
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 1
- **Avg / median % per leg:** -0.14% / -0.22%
- **Sum % (uncompounded):** -1.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.10% | -0.6% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.10% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.23% | -0.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.23% | -0.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 1 | 11.1% | 0 | 8 | 1 | -0.14% | -1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 645.20 | 641.54 | 0.00 | ORB-long ORB[635.75,643.90] vol=2.0x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 643.21 | 641.78 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 649.95 | 647.47 | 0.00 | ORB-long ORB[643.75,648.50] vol=2.1x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-05-19 09:35:00 | 648.46 | 647.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:55:00 | 628.00 | 631.89 | 0.00 | ORB-short ORB[630.00,635.90] vol=1.6x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-05-27 10:45:00 | 629.62 | 630.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:40:00 | 636.60 | 634.46 | 0.00 | ORB-long ORB[631.45,635.45] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-05-28 09:50:00 | 635.20 | 635.12 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:30:00 | 626.25 | 628.54 | 0.00 | ORB-short ORB[627.10,633.95] vol=1.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-05-30 09:35:00 | 627.73 | 628.34 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:45:00 | 642.40 | 639.45 | 0.00 | ORB-long ORB[633.55,641.50] vol=1.6x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 12:10:00 | 644.73 | 641.34 | 0.00 | T1 1.5R @ 644.73 |
| Stop hit — per-position SL triggered | 2025-06-04 14:10:00 | 642.40 | 642.08 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:35:00 | 649.20 | 646.33 | 0.00 | ORB-long ORB[642.00,646.95] vol=4.0x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-06-05 09:40:00 | 647.83 | 646.43 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 11:15:00 | 644.25 | 646.38 | 0.00 | ORB-short ORB[645.50,649.80] vol=1.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-06-06 11:40:00 | 645.53 | 646.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:30:00 | 645.20 | 2025-05-15 09:35:00 | 643.21 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-05-19 09:30:00 | 649.95 | 2025-05-19 09:35:00 | 648.46 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-27 09:55:00 | 628.00 | 2025-05-27 10:45:00 | 629.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-28 09:40:00 | 636.60 | 2025-05-28 09:50:00 | 635.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-30 09:30:00 | 626.25 | 2025-05-30 09:35:00 | 627.73 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-04 10:45:00 | 642.40 | 2025-06-04 12:10:00 | 644.73 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-06-04 10:45:00 | 642.40 | 2025-06-04 14:10:00 | 642.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 09:35:00 | 649.20 | 2025-06-05 09:40:00 | 647.83 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-06 11:15:00 | 644.25 | 2025-06-06 11:40:00 | 645.53 | STOP_HIT | 1.00 | -0.20% |
