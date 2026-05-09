# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 10696.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 4 / 11 / 0
- **Avg / median % per leg:** 1.30% / -0.90%
- **Sum % (uncompounded):** 19.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 4 | 11 | 0 | 1.30% | 19.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 7 | 46.7% | 4 | 11 | 0 | 1.30% | 19.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 7 | 46.7% | 4 | 11 | 0 | 1.30% | 19.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 9048.50 | 9437.80 | 9439.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9428.24 | 9434.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.50 | 9292.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 9413.50 | 9187.50 | 9292.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 9429.00 | 9189.90 | 9293.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:45:00 | 9440.00 | 9189.90 | 9293.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 9770.00 | 9375.89 | 9374.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 9799.00 | 9398.74 | 9386.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 9490.00 | 9492.89 | 9442.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 9490.00 | 9492.89 | 9442.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9498.26 | 9447.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 9374.00 | 9498.26 | 9447.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 9678.00 | 9500.05 | 9448.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 9852.50 | 9505.89 | 9452.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-11 09:30:00 | 8790.50 | 2025-12-15 10:15:00 | 8900.00 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-11-11 11:30:00 | 8789.50 | 2025-12-17 14:15:00 | 8897.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-11-14 12:00:00 | 8810.50 | 2025-12-17 14:15:00 | 8897.00 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-11-24 10:15:00 | 9026.00 | 2025-12-17 14:15:00 | 8897.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-11-28 11:45:00 | 9004.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-12-03 10:30:00 | 9003.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-12-03 15:00:00 | 9003.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-12-10 13:45:00 | 8970.00 | 2025-12-18 09:15:00 | 8745.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-15 13:45:00 | 8969.00 | 2026-01-05 10:15:00 | 9669.55 | TARGET_HIT | 1.00 | 7.81% |
| BUY | retest2 | 2025-12-16 09:30:00 | 8966.50 | 2026-01-05 10:15:00 | 9668.45 | TARGET_HIT | 1.00 | 7.83% |
| BUY | retest2 | 2025-12-16 12:30:00 | 8968.00 | 2026-01-06 09:15:00 | 9691.55 | TARGET_HIT | 1.00 | 8.07% |
| BUY | retest2 | 2026-01-22 09:15:00 | 9285.50 | 2026-02-25 11:15:00 | 10138.70 | TARGET_HIT | 1.00 | 9.19% |
| BUY | retest2 | 2026-01-22 12:00:00 | 9217.00 | 2026-03-12 09:15:00 | 9134.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-03-12 11:45:00 | 9196.00 | 2026-03-13 09:15:00 | 8969.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-03-17 09:45:00 | 9236.50 | 2026-03-17 10:15:00 | 9145.00 | STOP_HIT | 1.00 | -0.99% |
