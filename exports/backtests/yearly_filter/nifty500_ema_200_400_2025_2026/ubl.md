# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1419.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 1
- **Avg / median % per leg:** -0.04% / -1.32%
- **Sum % (uncompounded):** -0.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.72% | -13.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.72% | -13.7% |
| SELL (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.32% | 13.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.32% | 13.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.04% | -0.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 2000.00 | 2047.83 | 2047.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1990.20 | 2047.26 | 2047.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1985.20 | 1984.67 | 2005.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:00:00 | 1985.20 | 1984.67 | 2005.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1992.90 | 1984.13 | 2004.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1992.90 | 1984.13 | 2004.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1996.30 | 1984.41 | 2004.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:00:00 | 1994.30 | 1989.10 | 2005.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 2009.50 | 1989.44 | 2005.61 | SL hit (close>static) qty=1.00 sl=2006.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 1694.20 | 1603.59 | 1603.29 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 1551.80 | 1604.61 | 1604.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1541.50 | 1601.63 | 1603.22 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 15:15:00 | 2070.10 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-16 10:45:00 | 2054.50 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-05-16 13:00:00 | 2053.80 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-19 11:15:00 | 2060.10 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-06 13:30:00 | 2067.30 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-10 14:15:00 | 2064.20 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-10 14:45:00 | 2071.00 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-11 11:00:00 | 2065.00 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-18 14:00:00 | 1994.30 | 2025-07-18 15:15:00 | 2009.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1987.80 | 2025-07-28 13:15:00 | 2006.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1994.50 | 2025-08-18 11:15:00 | 1894.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1994.50 | 2025-09-01 12:15:00 | 1795.05 | TARGET_HIT | 0.50 | 10.00% |
