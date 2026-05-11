# NBCC (India) Ltd. (NBCC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 101.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 31
- **Target hits / Stop hits / Partials:** 1 / 31 / 0
- **Avg / median % per leg:** -1.55% / -1.68%
- **Sum % (uncompounded):** -49.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 1 | 4.2% | 1 | 23 | 0 | -1.32% | -31.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 1 | 4.2% | 1 | 23 | 0 | -1.32% | -31.6% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.23% | -17.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.23% | -17.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 1 | 3.1% | 1 | 31 | 0 | -1.55% | -49.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 105.76 | 111.40 | 111.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 103.92 | 111.27 | 111.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 105.80 | 105.25 | 107.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 105.80 | 105.25 | 107.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 107.62 | 105.30 | 107.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 107.50 | 105.30 | 107.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 108.90 | 105.33 | 107.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 108.90 | 105.33 | 107.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 108.15 | 105.46 | 107.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 109.14 | 105.46 | 107.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 107.22 | 105.50 | 107.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 107.75 | 105.50 | 107.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 107.25 | 105.52 | 107.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 107.25 | 105.52 | 107.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 108.72 | 105.58 | 107.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 108.72 | 105.58 | 107.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 108.40 | 105.61 | 107.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 108.85 | 105.61 | 107.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 109.35 | 105.76 | 107.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 109.31 | 105.76 | 107.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 109.35 | 105.79 | 107.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 108.97 | 105.86 | 107.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 108.95 | 105.89 | 107.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 109.65 | 105.96 | 107.53 | SL hit (close>static) qty=1.00 sl=109.56 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 11:15:00 | 112.15 | 108.57 | 108.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 112.45 | 108.61 | 108.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 109.37 | 109.76 | 109.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 13:00:00 | 109.37 | 109.76 | 109.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 109.44 | 109.77 | 109.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 109.55 | 109.77 | 109.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 110.72 | 109.77 | 109.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 111.00 | 109.77 | 109.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 111.27 | 110.17 | 109.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 111.33 | 110.72 | 109.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:15:00 | 110.89 | 112.41 | 111.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 111.55 | 112.41 | 111.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 112.60 | 112.41 | 111.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 11:45:00 | 112.24 | 112.47 | 111.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 112.13 | 112.50 | 111.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:30:00 | 112.04 | 112.50 | 111.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 112.00 | 112.47 | 111.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 112.00 | 112.47 | 111.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 109.03 | 112.44 | 111.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 109.03 | 112.44 | 111.25 | SL hit (close<static) qty=1.00 sl=109.23 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 103.75 | 113.33 | 113.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 102.61 | 113.13 | 113.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 88.06 | 86.51 | 92.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 09:45:00 | 87.92 | 86.51 | 92.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 91.49 | 87.33 | 91.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:45:00 | 90.90 | 90.63 | 92.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:45:00 | 90.95 | 90.64 | 92.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 91.07 | 90.64 | 92.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 94.42 | 90.95 | 92.43 | SL hit (close>static) qty=1.00 sl=93.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-11 12:30:00 | 113.64 | 2025-07-25 09:15:00 | 112.33 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-14 09:30:00 | 113.72 | 2025-07-25 09:15:00 | 112.33 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-15 09:15:00 | 114.25 | 2025-07-25 09:15:00 | 112.33 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-18 13:30:00 | 113.64 | 2025-07-25 09:15:00 | 112.33 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-21 10:15:00 | 113.68 | 2025-07-25 10:15:00 | 110.90 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-07-24 11:30:00 | 113.43 | 2025-07-25 10:15:00 | 110.90 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-07-24 13:15:00 | 113.40 | 2025-07-25 10:15:00 | 110.90 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-07-24 13:45:00 | 113.44 | 2025-07-25 10:15:00 | 110.90 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-16 13:00:00 | 108.97 | 2025-09-16 15:15:00 | 109.65 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-16 13:45:00 | 108.95 | 2025-09-16 15:15:00 | 109.65 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-25 15:00:00 | 109.00 | 2025-10-01 15:15:00 | 109.99 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-26 09:15:00 | 108.12 | 2025-10-01 15:15:00 | 109.99 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-10-01 13:30:00 | 108.35 | 2025-10-03 09:15:00 | 111.16 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-10-15 11:15:00 | 111.00 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-17 13:45:00 | 111.27 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-29 09:15:00 | 111.33 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-11-07 10:15:00 | 110.89 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-11-07 11:15:00 | 112.60 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-11-11 11:45:00 | 112.24 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-11-12 15:00:00 | 112.13 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-11-13 09:30:00 | 112.04 | 2025-11-13 14:15:00 | 109.03 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-11-14 09:15:00 | 109.60 | 2025-12-08 13:15:00 | 108.19 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-14 10:45:00 | 109.47 | 2025-12-08 13:15:00 | 108.19 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-12-12 09:30:00 | 109.43 | 2025-12-18 09:15:00 | 107.78 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-12 14:00:00 | 109.34 | 2025-12-18 09:15:00 | 107.78 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-12-19 14:15:00 | 112.75 | 2025-12-29 09:15:00 | 124.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-08 12:45:00 | 112.63 | 2026-01-08 15:15:00 | 111.68 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-08 13:30:00 | 112.91 | 2026-01-08 15:15:00 | 111.68 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-01-08 14:30:00 | 112.71 | 2026-01-08 15:15:00 | 111.68 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-30 10:45:00 | 90.90 | 2026-05-06 09:15:00 | 94.42 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2026-04-30 11:45:00 | 90.95 | 2026-05-06 09:15:00 | 94.42 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2026-04-30 12:15:00 | 91.07 | 2026-05-06 09:15:00 | 94.42 | STOP_HIT | 1.00 | -3.68% |
