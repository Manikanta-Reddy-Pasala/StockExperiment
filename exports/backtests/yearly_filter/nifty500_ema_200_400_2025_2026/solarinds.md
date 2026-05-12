# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 16101.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 2 / 8 / 3
- **Avg / median % per leg:** 1.67% / -0.77%
- **Sum % (uncompounded):** 21.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 6 | 46.2% | 2 | 8 | 3 | 1.67% | 21.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 2 | 8 | 3 | 1.67% | 21.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 6 | 46.2% | 2 | 8 | 3 | 1.67% | 21.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 14535.00 | 15004.74 | 15006.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 14470.00 | 14999.42 | 15004.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 14575.00 | 14409.16 | 14639.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 14575.00 | 14409.16 | 14639.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 14503.00 | 14410.09 | 14638.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 14440.00 | 14410.36 | 14637.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 14481.00 | 14410.77 | 14631.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:30:00 | 14474.00 | 14411.44 | 14630.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 14:45:00 | 14493.00 | 14412.15 | 14629.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 14637.00 | 14416.47 | 14628.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 14635.00 | 14416.47 | 14628.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 14617.00 | 14418.47 | 14628.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:15:00 | 14632.00 | 14418.47 | 14628.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 14588.00 | 14420.16 | 14628.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 14571.00 | 14420.16 | 14628.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 14540.00 | 14421.37 | 14628.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 14862.00 | 14429.30 | 14629.02 | SL hit (close>static) qty=1.00 sl=14669.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 13241.00 | 13128.09 | 13127.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 13288.00 | 13132.72 | 13130.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 10:15:00 | 13825.00 | 13894.43 | 13596.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:30:00 | 13793.00 | 13894.43 | 13596.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 13329.00 | 13885.90 | 13610.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 13329.00 | 13885.90 | 13610.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 13296.00 | 13880.03 | 13609.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 13296.00 | 13880.03 | 13609.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 12529.00 | 13399.38 | 13402.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 12456.00 | 13365.29 | 13384.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 13330.00 | 13329.19 | 13364.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 13330.00 | 13329.19 | 13364.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 13290.00 | 13328.92 | 13364.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 13270.00 | 13328.92 | 13364.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 13340.00 | 13329.03 | 13364.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 13340.00 | 13329.03 | 13364.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 13382.00 | 13329.55 | 13364.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:45:00 | 13395.00 | 13329.55 | 13364.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 13408.00 | 13330.34 | 13364.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 13395.00 | 13330.34 | 13364.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 14114.00 | 13397.06 | 13395.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 14510.00 | 13475.13 | 13435.99 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-12 13:30:00 | 14440.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-09-15 12:30:00 | 14481.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-09-15 13:30:00 | 14474.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-09-15 14:45:00 | 14493.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-09-16 13:15:00 | 14571.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-09-16 13:45:00 | 14540.00 | 2025-09-17 09:15:00 | 14862.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-18 10:15:00 | 14571.00 | 2025-09-18 10:15:00 | 14683.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-22 11:00:00 | 14537.00 | 2025-09-26 13:15:00 | 13810.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:00:00 | 14537.00 | 2025-10-07 10:15:00 | 14180.00 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-11-12 14:15:00 | 14078.00 | 2025-11-24 12:15:00 | 13374.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 14080.00 | 2025-11-24 12:15:00 | 13376.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 14078.00 | 2025-12-08 09:15:00 | 12670.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 15:00:00 | 14080.00 | 2025-12-08 09:15:00 | 12672.00 | TARGET_HIT | 0.50 | 10.00% |
