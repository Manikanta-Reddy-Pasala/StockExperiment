# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 568.90
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -4.41% / -4.11%
- **Sum % (uncompounded):** -13.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.41% | -13.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.41% | -13.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.41% | -13.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 245.65 | 224.23 | 224.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 253.69 | 224.72 | 224.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 243.69 | 245.41 | 237.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 243.95 | 245.41 | 237.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 256.20 | 265.47 | 256.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 256.95 | 265.47 | 256.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 256.50 | 265.38 | 256.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 259.10 | 264.22 | 256.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 259.25 | 263.86 | 256.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 261.90 | 263.69 | 256.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 248.60 | 263.52 | 256.41 | SL hit (close<static) qty=1.00 sl=254.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 248.60 | 263.52 | 256.41 | SL hit (close<static) qty=1.00 sl=254.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 248.60 | 263.52 | 256.41 | SL hit (close<static) qty=1.00 sl=254.55 alert=retest2 |
| CROSSOVER_SKIP | 2025-08-14 13:15:00 | 239.80 | 251.54 | 251.56 | min_gap filter: gap=0.009% < 0.030% |
| TREND_RESET | 2025-08-14 13:15:00 | 239.80 | 251.54 | 251.56 | EMA inversion without crossover edge (EMA200=251.54 EMA400=251.56) — end cycle |
| CROSSOVER_SKIP | 2025-09-16 09:15:00 | 283.10 | 248.59 | 248.52 | min_gap filter: gap=0.023% < 0.030% |
| CROSSOVER_SKIP | 2026-03-30 13:15:00 | 458.75 | 516.63 | 516.76 | min_gap filter: gap=0.029% < 0.030% |
| CROSSOVER_SKIP | 2026-04-15 10:15:00 | 554.40 | 516.04 | 515.99 | min_gap filter: gap=0.009% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 13:00:00 | 259.10 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-07-30 11:30:00 | 259.25 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-07-30 15:00:00 | 261.90 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -5.08% |
