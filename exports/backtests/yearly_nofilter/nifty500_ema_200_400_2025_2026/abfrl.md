# Aditya Birla Fashion and Retail Ltd. (ABFRL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 66.15
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
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 0 / 9 / 2
- **Avg / median % per leg:** -0.17% / -2.42%
- **Sum % (uncompounded):** -1.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 0 | 9 | 2 | -0.17% | -1.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 4 | 36.4% | 0 | 9 | 2 | -0.17% | -1.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 4 | 36.4% | 0 | 9 | 2 | -0.17% | -1.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 278.25 | 262.39 | 262.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 281.15 | 263.98 | 263.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 89.95 | 265.67 | 264.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 90.45 | 262.21 | 262.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 90.00 | 260.49 | 261.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 79.94 | 77.73 | 95.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 79.94 | 77.73 | 95.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 89.38 | 80.20 | 91.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 91.27 | 80.20 | 91.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 90.55 | 80.57 | 91.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 89.55 | 81.31 | 91.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 89.65 | 81.31 | 91.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 88.90 | 83.23 | 90.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 89.34 | 83.30 | 90.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 90.71 | 83.37 | 90.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 90.54 | 83.37 | 90.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 90.74 | 83.44 | 90.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 90.32 | 83.44 | 90.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 89.99 | 83.58 | 90.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 91.82 | 84.21 | 90.79 | SL hit (close>static) qty=1.00 sl=91.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-10 10:30:00 | 89.55 | 2025-09-19 10:15:00 | 91.82 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-09-10 11:00:00 | 89.65 | 2025-09-19 10:15:00 | 91.82 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-09-17 09:45:00 | 88.90 | 2025-09-19 14:15:00 | 91.74 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-09-17 11:15:00 | 89.34 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-09-17 13:15:00 | 90.32 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-17 15:15:00 | 89.99 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-19 13:00:00 | 90.44 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-09-25 09:15:00 | 90.05 | 2025-09-26 11:15:00 | 85.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:30:00 | 89.11 | 2025-09-26 13:15:00 | 84.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 90.05 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-09-25 10:30:00 | 89.11 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 0.50 | 3.20% |
