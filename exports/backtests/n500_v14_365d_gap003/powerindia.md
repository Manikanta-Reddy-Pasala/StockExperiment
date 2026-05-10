# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2024-07-09 09:15:00 → 2026-05-08 15:15:00 (3168 bars)
- **Last close:** 33960.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 10
- **Target hits / Stop hits / Partials:** 0 / 10 / 0
- **Avg / median % per leg:** -1.75% / -1.54%
- **Sum % (uncompounded):** -17.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.75% | -17.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.75% | -17.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.75% | -17.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 21865.00 | 18733.88 | 18725.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 22172.00 | 19779.36 | 19310.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 20560.00 | 20898.99 | 20110.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 20560.00 | 20898.99 | 20110.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 20100.00 | 20879.19 | 20112.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 20100.00 | 20879.19 | 20112.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 20080.00 | 20871.23 | 20112.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 19980.00 | 20862.37 | 20111.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 19800.00 | 20851.80 | 20110.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 19800.00 | 20851.80 | 20110.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 19355.00 | 20836.90 | 20106.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 19355.00 | 20836.90 | 20106.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| CROSSOVER_SKIP | 2025-12-24 13:15:00 | 18630.00 | 19671.87 | 19675.85 | min_gap filter: gap=0.021% < 0.030% |
| TREND_RESET | 2025-12-24 13:15:00 | 18630.00 | 19671.87 | 19675.85 | EMA inversion without crossover edge (EMA200=19671.87 EMA400=19675.85) — end cycle |
| CROSSOVER_SKIP | 2026-02-10 12:15:00 | 22483.00 | 18916.79 | 18911.70 | min_gap filter: gap=0.023% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-01 09:15:00 | 19241.00 | 2025-09-02 10:15:00 | 18900.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-09-01 13:15:00 | 19152.00 | 2025-09-02 10:15:00 | 18900.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-09-02 09:15:00 | 19115.00 | 2025-09-02 10:15:00 | 18900.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-03 10:45:00 | 19160.00 | 2025-09-04 13:15:00 | 19095.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-09-04 09:15:00 | 19319.00 | 2025-09-04 14:15:00 | 18839.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-10 09:30:00 | 19368.00 | 2025-09-19 12:15:00 | 19026.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-09-22 09:15:00 | 19240.00 | 2025-09-22 14:15:00 | 19090.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-22 14:45:00 | 19244.00 | 2025-09-23 10:15:00 | 18948.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-09-26 12:15:00 | 19701.00 | 2025-09-26 14:15:00 | 19069.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-09-26 12:45:00 | 19696.00 | 2025-09-26 14:15:00 | 19069.00 | STOP_HIT | 1.00 | -3.18% |
