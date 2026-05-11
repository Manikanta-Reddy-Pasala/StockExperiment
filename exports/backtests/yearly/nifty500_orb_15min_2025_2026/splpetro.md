# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2026-04-06 09:15:00 → 2026-05-08 15:25:00 (1725 bars)
- **Last close:** 738.40
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
- **Avg / median % per leg:** 0.31% / 0.00%
- **Sum % (uncompounded):** 2.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.43% | 2.1% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.43% | 2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.11% | 0.3% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.11% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.31% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 751.35 | 746.29 | 0.00 | ORB-long ORB[738.85,749.80] vol=3.1x ATR=3.54 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 747.81 | 746.71 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-04-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 754.30 | 740.19 | 0.00 | ORB-long ORB[730.40,741.65] vol=2.3x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:15:00 | 759.74 | 749.19 | 0.00 | T1 1.5R @ 759.74 |
| Target hit | 2026-04-15 15:20:00 | 775.80 | 764.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 779.35 | 775.12 | 0.00 | ORB-long ORB[769.95,778.00] vol=2.6x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-04-22 09:45:00 | 776.80 | 776.21 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 788.00 | 779.80 | 0.00 | ORB-long ORB[769.85,781.20] vol=4.2x ATR=5.06 |
| Stop hit — per-position SL triggered | 2026-04-28 09:40:00 | 782.94 | 785.11 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 749.00 | 757.03 | 0.00 | ORB-short ORB[753.00,761.15] vol=5.5x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:40:00 | 744.44 | 754.33 | 0.00 | T1 1.5R @ 744.44 |
| Stop hit — per-position SL triggered | 2026-05-04 11:00:00 | 749.00 | 753.58 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-05-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:35:00 | 732.25 | 736.39 | 0.00 | ORB-short ORB[733.00,743.40] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-05-05 11:05:00 | 734.36 | 736.20 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-04-10 09:45:00 | 751.35 | 2026-04-10 10:00:00 | 747.81 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-15 11:00:00 | 754.30 | 2026-04-15 11:15:00 | 759.74 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2026-04-15 11:00:00 | 754.30 | 2026-04-15 15:20:00 | 775.80 | TARGET_HIT | 0.50 | 2.85% |
| BUY | retest1 | 2026-04-22 09:35:00 | 779.35 | 2026-04-22 09:45:00 | 776.80 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-28 09:30:00 | 788.00 | 2026-04-28 09:40:00 | 782.94 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2026-05-04 10:30:00 | 749.00 | 2026-05-04 10:40:00 | 744.44 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-05-04 10:30:00 | 749.00 | 2026-05-04 11:00:00 | 749.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:35:00 | 732.25 | 2026-05-05 11:05:00 | 734.36 | STOP_HIT | 1.00 | -0.29% |
