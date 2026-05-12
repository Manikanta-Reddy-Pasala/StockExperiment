# IDFC First Bank Ltd. (IDFCFIRSTB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 71.19
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 19
- **Target hits / Stop hits / Partials:** 0 / 19 / 0
- **Avg / median % per leg:** -7.36% / -3.83%
- **Sum % (uncompounded):** -139.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -8.85% | -132.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -8.85% | -132.8% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.76% | -7.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.76% | -7.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 0 | 0.0% | 0 | 19 | 0 | -7.36% | -139.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 09:15:00 | 68.63 | 70.22 | 70.23 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 72.10 | 70.23 | 70.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 72.60 | 70.25 | 70.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 71.08 | 71.13 | 70.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 71.08 | 71.13 | 70.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 70.89 | 71.30 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 70.97 | 71.30 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 70.94 | 71.30 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 70.87 | 71.30 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 71.24 | 71.30 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 71.07 | 71.30 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 70.86 | 71.29 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 70.86 | 71.29 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 70.65 | 71.29 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 70.49 | 71.29 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 70.84 | 71.28 | 70.89 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 69.07 | 70.59 | 70.60 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 71.80 | 70.60 | 70.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 72.00 | 70.62 | 70.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 78.01 | 78.26 | 76.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 78.01 | 78.26 | 76.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 81.24 | 83.47 | 81.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 82.87 | 83.34 | 81.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:00:00 | 82.44 | 83.35 | 81.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:45:00 | 83.28 | 83.33 | 81.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 13:00:00 | 82.60 | 83.29 | 81.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 83.14 | 83.30 | 81.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 11:15:00 | 79.50 | 83.15 | 81.92 | SL hit (close<static) qty=1.00 sl=80.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 69.27 | 81.77 | 81.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 66.63 | 77.43 | 79.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 15:15:00 | 66.13 | 66.07 | 70.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 66.06 | 66.07 | 70.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 70.12 | 66.88 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:45:00 | 70.15 | 66.88 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 70.36 | 66.91 | 69.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 70.36 | 66.91 | 69.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 70.15 | 67.11 | 69.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 70.15 | 67.11 | 69.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 70.35 | 67.14 | 69.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 70.35 | 67.14 | 69.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 69.59 | 67.41 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 69.70 | 67.41 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 69.57 | 67.43 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 69.89 | 67.43 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 69.85 | 67.45 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 70.20 | 67.45 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 70.39 | 67.48 | 69.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 69.75 | 67.54 | 69.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:45:00 | 69.56 | 67.72 | 69.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:00:00 | 69.74 | 67.87 | 69.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:30:00 | 69.89 | 67.89 | 69.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 70.04 | 67.91 | 69.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 70.04 | 67.91 | 69.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-07 15:15:00 | 70.96 | 67.94 | 69.70 | SL hit (close>static) qty=1.00 sl=70.72 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-20 10:00:00 | 71.63 | 2025-08-21 14:15:00 | 70.03 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-01-22 09:15:00 | 82.87 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2026-01-27 12:00:00 | 82.44 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2026-01-27 14:45:00 | 83.28 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest2 | 2026-01-28 13:00:00 | 82.60 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2026-02-03 12:30:00 | 84.54 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-02-03 13:45:00 | 84.64 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-02-04 12:00:00 | 84.52 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-02-04 12:30:00 | 84.66 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2026-02-16 14:45:00 | 82.81 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.35% |
| BUY | retest2 | 2026-02-17 12:30:00 | 82.94 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.48% |
| BUY | retest2 | 2026-02-17 13:30:00 | 82.95 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.49% |
| BUY | retest2 | 2026-02-17 14:00:00 | 82.83 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.37% |
| BUY | retest2 | 2026-02-20 09:30:00 | 83.51 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -17.05% |
| BUY | retest2 | 2026-02-20 10:45:00 | 83.22 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.76% |
| SELL | retest2 | 2026-05-04 12:00:00 | 69.75 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-05-06 09:45:00 | 69.56 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-05-07 13:00:00 | 69.74 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-05-07 13:30:00 | 69.89 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -1.53% |
