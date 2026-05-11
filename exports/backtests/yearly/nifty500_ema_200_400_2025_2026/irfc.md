# Indian Railway Finance Corporation Ltd. (IRFC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 106.02
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 7
- **Target hits / Stop hits / Partials:** 5 / 7 / 5
- **Avg / median % per leg:** 3.64% / 5.00%
- **Sum % (uncompounded):** 61.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.90% | -3.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.90% | -3.6% |
| SELL (all) | 13 | 10 | 76.9% | 5 | 3 | 5 | 5.04% | 65.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 10 | 76.9% | 5 | 3 | 5 | 5.04% | 65.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 10 | 58.8% | 5 | 7 | 5 | 3.64% | 61.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 136.75 | 129.21 | 129.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 137.18 | 129.50 | 129.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 138.25 | 138.36 | 134.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:15:00 | 138.55 | 138.36 | 134.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 135.48 | 138.28 | 135.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 135.48 | 138.28 | 135.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 134.74 | 138.24 | 135.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 135.00 | 138.24 | 135.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 133.10 | 138.19 | 135.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 133.20 | 138.19 | 135.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 135.70 | 137.91 | 135.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 136.23 | 137.68 | 135.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 135.79 | 138.53 | 136.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 136.10 | 138.50 | 136.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 134.82 | 138.38 | 136.69 | SL hit (close<static) qty=1.00 sl=134.85 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 10:15:00 | 131.52 | 135.67 | 135.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 128.94 | 135.42 | 135.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 126.49 | 125.80 | 128.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:45:00 | 126.30 | 125.80 | 128.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 129.66 | 125.89 | 128.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 130.29 | 125.89 | 128.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 128.81 | 125.92 | 128.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 129.36 | 125.92 | 128.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 128.92 | 125.97 | 128.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 128.87 | 125.97 | 128.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 128.01 | 125.99 | 128.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:45:00 | 127.70 | 127.00 | 128.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 127.71 | 127.01 | 128.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 12:30:00 | 127.70 | 127.02 | 128.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 127.65 | 127.03 | 128.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 124.87 | 125.78 | 127.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 124.75 | 125.78 | 127.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 121.31 | 124.61 | 125.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 121.32 | 124.61 | 125.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 121.31 | 124.61 | 125.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 121.27 | 124.61 | 125.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-03 10:15:00 | 114.93 | 120.37 | 122.67 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 127.74 | 121.20 | 121.19 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 115.31 | 121.31 | 121.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 113.52 | 120.29 | 120.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 120.15 | 119.87 | 120.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 120.15 | 119.87 | 120.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 120.80 | 119.87 | 120.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 121.00 | 119.87 | 120.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 119.89 | 119.87 | 120.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 118.99 | 119.87 | 120.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 118.65 | 119.85 | 120.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:00:00 | 119.51 | 119.86 | 120.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 122.81 | 119.90 | 120.51 | SL hit (close>static) qty=1.00 sl=121.59 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 11:00:00 | 136.23 | 2025-07-14 09:15:00 | 134.82 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-11 11:30:00 | 135.79 | 2025-07-14 09:15:00 | 134.82 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-11 12:30:00 | 136.10 | 2025-07-14 09:15:00 | 134.82 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-16 11:00:00 | 135.76 | 2025-07-17 11:15:00 | 134.51 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-23 09:45:00 | 127.70 | 2025-11-06 09:15:00 | 121.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 10:30:00 | 127.71 | 2025-11-06 09:15:00 | 121.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 12:30:00 | 127.70 | 2025-11-06 09:15:00 | 121.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 127.65 | 2025-11-06 09:15:00 | 121.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:45:00 | 127.70 | 2025-12-03 10:15:00 | 114.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 10:30:00 | 127.71 | 2025-12-03 10:15:00 | 114.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 12:30:00 | 127.70 | 2025-12-03 10:15:00 | 114.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 127.65 | 2025-12-03 10:15:00 | 114.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-29 10:15:00 | 118.99 | 2026-02-01 09:15:00 | 122.81 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2026-01-30 09:15:00 | 118.65 | 2026-02-01 09:15:00 | 122.81 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-01-30 13:00:00 | 119.51 | 2026-02-01 09:15:00 | 122.81 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-02-01 12:00:00 | 118.99 | 2026-02-01 12:15:00 | 113.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:00:00 | 118.99 | 2026-02-25 09:15:00 | 107.09 | TARGET_HIT | 0.50 | 10.00% |
