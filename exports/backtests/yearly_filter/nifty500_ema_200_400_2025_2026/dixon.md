# Dixon Technologies (India) Ltd. (DIXON)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 10825.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -2.14% / -1.52%
- **Sum % (uncompounded):** -17.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.05% | -8.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.05% | -8.2% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.24% | -9.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.24% | -9.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.14% | -17.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 14490.00 | 15124.03 | 15125.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 14326.00 | 15116.09 | 15121.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 14823.00 | 14705.96 | 14879.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 14823.00 | 14705.96 | 14879.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 14823.00 | 14705.96 | 14879.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 14911.00 | 14705.96 | 14879.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 14982.00 | 14697.57 | 14865.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 14982.00 | 14697.57 | 14865.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 14912.00 | 14699.70 | 14865.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:15:00 | 14990.00 | 14699.70 | 14865.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 14990.00 | 14702.59 | 14866.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 14501.00 | 14702.59 | 14866.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 15180.00 | 14707.24 | 14858.31 | SL hit (close>static) qty=1.00 sl=15108.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 15809.00 | 14974.05 | 14974.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 15900.00 | 15007.08 | 14990.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 16130.00 | 16182.44 | 15764.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 14:00:00 | 16130.00 | 16182.44 | 15764.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 15870.00 | 16144.74 | 15782.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 15870.00 | 16144.74 | 15782.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 16890.00 | 17642.55 | 17052.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 16890.00 | 17642.55 | 17052.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 16776.00 | 17633.93 | 17051.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 16771.00 | 17633.93 | 17051.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 17025.00 | 17335.71 | 16975.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 17115.00 | 17329.93 | 16976.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 17108.00 | 17329.93 | 16976.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 16848.00 | 17305.60 | 16984.51 | SL hit (close<static) qty=1.00 sl=16956.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 15517.00 | 16809.61 | 16813.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 15456.00 | 16770.56 | 16793.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 11446.00 | 11356.58 | 12477.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:00:00 | 11446.00 | 11356.58 | 12477.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 10814.50 | 10433.76 | 10893.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 10789.00 | 10729.15 | 10958.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:00:00 | 10783.00 | 10729.15 | 10958.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 10777.00 | 10730.27 | 10957.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 10937.00 | 10733.94 | 10955.96 | SL hit (close>static) qty=1.00 sl=10930.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-01 09:15:00 | 14501.00 | 2025-07-02 14:15:00 | 15180.00 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2025-10-07 09:30:00 | 17115.00 | 2025-10-08 14:15:00 | 16848.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-10-07 10:15:00 | 17108.00 | 2025-10-08 14:15:00 | 16848.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-10-09 12:45:00 | 17093.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-10-09 14:00:00 | 17090.00 | 2025-10-14 09:15:00 | 16655.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-04-24 12:30:00 | 10789.00 | 2026-04-27 09:15:00 | 10937.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-04-24 13:00:00 | 10783.00 | 2026-04-27 09:15:00 | 10937.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-24 13:30:00 | 10777.00 | 2026-04-27 09:15:00 | 10937.00 | STOP_HIT | 1.00 | -1.48% |
