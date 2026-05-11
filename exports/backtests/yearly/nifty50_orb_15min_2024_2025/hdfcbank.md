# HDFCBANK (HDFCBANK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-06-07 15:25:00 (1446 bars)
- **Last close:** 787.50
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** 0.25% / -0.19%
- **Sum % (uncompounded):** 1.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.25% | 1.3% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.25% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.25% | 1.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:30:00 | 732.25 | 730.89 | 0.00 | ORB-long ORB[728.53,731.15] vol=2.2x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:00:00 | 734.34 | 731.21 | 0.00 | T1 1.5R @ 734.34 |
| Target hit | 2024-05-23 15:20:00 | 746.78 | 738.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 11:10:00 | 753.50 | 749.81 | 0.00 | ORB-long ORB[743.00,750.88] vol=3.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-05-24 11:55:00 | 752.09 | 750.29 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 11:10:00 | 761.28 | 749.02 | 0.00 | ORB-long ORB[744.00,754.00] vol=1.7x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-06-05 11:45:00 | 757.06 | 750.52 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:40:00 | 785.03 | 781.54 | 0.00 | ORB-long ORB[776.03,783.50] vol=1.5x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-06-07 11:25:00 | 783.02 | 782.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-23 10:30:00 | 732.25 | 2024-05-23 11:00:00 | 734.34 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-05-23 10:30:00 | 732.25 | 2024-05-23 15:20:00 | 746.78 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2024-05-24 11:10:00 | 753.50 | 2024-05-24 11:55:00 | 752.09 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-06-05 11:10:00 | 761.28 | 2024-06-05 11:45:00 | 757.06 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-06-07 10:40:00 | 785.03 | 2024-06-07 11:25:00 | 783.02 | STOP_HIT | 1.00 | -0.26% |
