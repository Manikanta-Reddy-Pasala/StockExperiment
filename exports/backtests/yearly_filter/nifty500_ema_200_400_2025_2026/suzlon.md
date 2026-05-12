# Suzlon Energy Ltd. (SUZLON)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 54.90
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
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 10 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 10
- **Target hits / Stop hits / Partials:** 3 / 10 / 3
- **Avg / median % per leg:** 1.20% / -1.50%
- **Sum % (uncompounded):** 19.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.11% | -12.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.11% | -12.3% |
| SELL (all) | 13 | 6 | 46.2% | 3 | 7 | 3 | 2.42% | 31.5% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.56% | -7.7% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 3 | 4 | 3 | 3.92% | 39.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.56% | -7.7% |
| retest2 (combined) | 13 | 6 | 46.2% | 3 | 7 | 3 | 2.07% | 26.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 60.85 | 56.32 | 56.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 62.10 | 56.91 | 56.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 64.03 | 64.15 | 61.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 12:00:00 | 64.03 | 64.15 | 61.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 64.12 | 65.52 | 64.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 64.12 | 65.52 | 64.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 63.86 | 65.50 | 64.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 63.95 | 65.50 | 64.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 63.24 | 65.48 | 64.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 63.24 | 65.48 | 64.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 63.53 | 64.61 | 63.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 63.53 | 64.61 | 63.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 62.69 | 64.59 | 63.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 62.69 | 64.59 | 63.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 63.35 | 64.56 | 63.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 63.35 | 64.56 | 63.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 63.57 | 64.55 | 63.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 63.08 | 64.55 | 63.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 63.29 | 64.45 | 63.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 63.29 | 64.45 | 63.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 63.35 | 64.44 | 63.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 63.56 | 64.44 | 63.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:15:00 | 63.69 | 64.42 | 63.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 63.41 | 64.39 | 63.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 60.94 | 64.35 | 63.87 | SL hit (close<static) qty=1.00 sl=62.94 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 58.30 | 63.44 | 63.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 57.38 | 62.16 | 62.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 59.29 | 59.22 | 60.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 13:30:00 | 59.07 | 59.22 | 60.61 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:15:00 | 59.14 | 59.22 | 60.58 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 15:15:00 | 59.05 | 59.21 | 60.55 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 60.60 | 59.26 | 60.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 60.60 | 59.26 | 60.52 | SL hit (close>ema400) qty=1.00 sl=60.52 alert=retest1 |

### Cycle 3 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 54.60 | 45.87 | 45.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 55.70 | 47.07 | 46.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-12 09:15:00 | 63.56 | 2025-08-13 09:15:00 | 60.94 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-08-12 12:15:00 | 63.69 | 2025-08-13 09:15:00 | 60.94 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2025-08-12 15:00:00 | 63.41 | 2025-08-13 09:15:00 | 60.94 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest1 | 2025-09-17 13:30:00 | 59.07 | 2025-09-19 15:15:00 | 60.60 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest1 | 2025-09-18 11:15:00 | 59.14 | 2025-09-19 15:15:00 | 60.60 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest1 | 2025-09-18 15:15:00 | 59.05 | 2025-09-19 15:15:00 | 60.60 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-11-10 12:45:00 | 57.15 | 2025-11-11 15:15:00 | 58.03 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-11-11 09:30:00 | 57.11 | 2025-11-11 15:15:00 | 58.03 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-11-11 10:15:00 | 57.17 | 2025-11-11 15:15:00 | 58.03 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-11-14 14:15:00 | 57.17 | 2025-11-14 14:15:00 | 57.82 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-18 15:15:00 | 56.80 | 2025-11-25 15:15:00 | 54.11 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-11-20 13:00:00 | 56.96 | 2025-11-28 13:15:00 | 53.96 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-11-20 14:30:00 | 56.85 | 2025-11-28 13:15:00 | 54.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 15:15:00 | 56.80 | 2025-12-04 12:15:00 | 51.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 13:00:00 | 56.96 | 2025-12-04 12:15:00 | 51.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 56.85 | 2025-12-04 12:15:00 | 51.16 | TARGET_HIT | 0.50 | 10.00% |
