# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 403.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 210 |
| ALERT1 | 150 |
| ALERT2 | 149 |
| ALERT2_SKIP | 64 |
| ALERT3 | 433 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 150 |
| PARTIAL | 23 |
| TARGET_HIT | 10 |
| STOP_HIT | 145 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 178 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 77 / 101
- **Target hits / Stop hits / Partials:** 10 / 145 / 23
- **Avg / median % per leg:** 0.64% / -0.65%
- **Sum % (uncompounded):** 113.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 84 | 24 | 28.6% | 8 | 74 | 2 | 0.17% | 14.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.29% | 16.0% |
| BUY @ 3rd Alert (retest2) | 77 | 20 | 26.0% | 7 | 70 | 0 | -0.02% | -1.6% |
| SELL (all) | 94 | 53 | 56.4% | 2 | 71 | 21 | 1.05% | 99.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 94 | 53 | 56.4% | 2 | 71 | 21 | 1.05% | 99.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 4 | 2 | 2.29% | 16.0% |
| retest2 (combined) | 171 | 73 | 42.7% | 9 | 141 | 21 | 0.57% | 97.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 11:15:00 | 80.85 | 80.91 | 80.92 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 82.00 | 80.93 | 80.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 09:15:00 | 82.70 | 81.91 | 81.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 13:15:00 | 81.70 | 82.08 | 81.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 13:15:00 | 81.70 | 82.08 | 81.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 81.70 | 82.08 | 81.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 14:00:00 | 81.70 | 82.08 | 81.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 81.65 | 81.99 | 81.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 15:15:00 | 81.65 | 81.99 | 81.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 15:15:00 | 81.65 | 81.92 | 81.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 09:15:00 | 81.00 | 81.92 | 81.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 80.10 | 81.56 | 81.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 14:15:00 | 79.35 | 79.88 | 80.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 80.15 | 79.82 | 80.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-23 10:00:00 | 80.15 | 79.82 | 80.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 80.35 | 79.93 | 80.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:15:00 | 80.50 | 79.93 | 80.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 80.35 | 80.01 | 80.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:45:00 | 80.45 | 80.01 | 80.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 80.90 | 80.19 | 80.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:00:00 | 80.90 | 80.19 | 80.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 80.80 | 80.31 | 80.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:30:00 | 81.00 | 80.31 | 80.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 80.35 | 80.35 | 80.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 80.50 | 80.35 | 80.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 80.70 | 80.42 | 80.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:30:00 | 80.80 | 80.42 | 80.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 80.65 | 80.47 | 80.46 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 11:15:00 | 80.10 | 80.39 | 80.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 79.95 | 80.30 | 80.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 80.35 | 80.23 | 80.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 80.35 | 80.23 | 80.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 80.35 | 80.23 | 80.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:45:00 | 80.25 | 80.23 | 80.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 80.95 | 80.37 | 80.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:00:00 | 80.95 | 80.37 | 80.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 11:15:00 | 80.65 | 80.43 | 80.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 13:15:00 | 81.30 | 80.65 | 80.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 79.45 | 80.73 | 80.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 79.45 | 80.73 | 80.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 79.45 | 80.73 | 80.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 79.45 | 80.73 | 80.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 10:15:00 | 78.75 | 80.33 | 80.44 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 11:15:00 | 83.05 | 80.34 | 80.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 84.80 | 83.39 | 82.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 83.10 | 84.52 | 83.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 83.10 | 84.52 | 83.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 83.10 | 84.52 | 83.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:00:00 | 83.10 | 84.52 | 83.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 82.45 | 84.11 | 83.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:00:00 | 82.45 | 84.11 | 83.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 84.00 | 83.76 | 83.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 14:30:00 | 83.45 | 83.76 | 83.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 83.50 | 83.71 | 83.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 09:15:00 | 84.15 | 83.71 | 83.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 09:45:00 | 84.25 | 83.90 | 83.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 09:15:00 | 84.75 | 84.17 | 83.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 10:45:00 | 84.35 | 84.38 | 84.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 85.10 | 84.52 | 84.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 14:00:00 | 85.45 | 84.72 | 84.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 14:30:00 | 85.25 | 85.59 | 85.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 15:00:00 | 85.30 | 85.59 | 85.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 09:15:00 | 84.25 | 85.21 | 85.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 84.25 | 85.21 | 85.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 12:15:00 | 83.55 | 84.56 | 84.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 10:15:00 | 84.25 | 84.14 | 84.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-15 11:00:00 | 84.25 | 84.14 | 84.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 84.45 | 84.21 | 84.53 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 85.65 | 84.64 | 84.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 13:15:00 | 86.70 | 85.41 | 85.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 14:15:00 | 87.30 | 87.35 | 86.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 15:00:00 | 87.30 | 87.35 | 86.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 86.75 | 87.25 | 86.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:30:00 | 86.45 | 87.25 | 86.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 87.15 | 87.23 | 86.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 86.50 | 87.23 | 86.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 87.10 | 87.56 | 87.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 11:00:00 | 87.10 | 87.56 | 87.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 87.70 | 87.59 | 87.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:15:00 | 87.80 | 87.59 | 87.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:45:00 | 87.80 | 87.65 | 87.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 15:15:00 | 86.85 | 87.43 | 87.26 | SL hit (close<static) qty=1.00 sl=87.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 86.80 | 87.10 | 87.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 85.90 | 86.71 | 86.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 84.95 | 84.25 | 84.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 84.95 | 84.25 | 84.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 84.95 | 84.25 | 84.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 84.95 | 84.25 | 84.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 84.75 | 84.35 | 84.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:45:00 | 84.40 | 84.42 | 84.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 14:00:00 | 84.50 | 84.44 | 84.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 12:00:00 | 84.50 | 84.57 | 84.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 12:15:00 | 85.20 | 84.70 | 84.73 | SL hit (close>static) qty=1.00 sl=85.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 13:15:00 | 85.10 | 84.78 | 84.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 87.00 | 85.34 | 85.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 87.45 | 87.88 | 87.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 87.45 | 87.88 | 87.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 87.45 | 87.88 | 87.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 87.45 | 87.88 | 87.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 87.15 | 87.78 | 87.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 12:45:00 | 87.35 | 87.78 | 87.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 86.90 | 87.60 | 87.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:00:00 | 86.90 | 87.60 | 87.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 87.00 | 87.48 | 87.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:30:00 | 86.85 | 87.48 | 87.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 90.30 | 92.35 | 91.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 90.05 | 92.35 | 91.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 90.70 | 92.02 | 91.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 13:45:00 | 91.15 | 91.64 | 91.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 12:15:00 | 90.80 | 91.24 | 91.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 12:15:00 | 90.80 | 91.24 | 91.25 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 92.65 | 91.52 | 91.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 10:15:00 | 93.55 | 91.92 | 91.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 15:15:00 | 94.10 | 94.24 | 93.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:15:00 | 95.70 | 94.24 | 93.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 93.20 | 93.98 | 93.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-13 11:15:00 | 93.20 | 93.98 | 93.54 | SL hit (close<ema400) qty=1.00 sl=93.54 alert=retest1 |

### Cycle 15 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 91.85 | 93.18 | 93.26 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 13:15:00 | 93.25 | 92.79 | 92.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 94.35 | 93.18 | 92.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 15:15:00 | 94.70 | 94.82 | 94.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 09:15:00 | 96.05 | 94.82 | 94.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 96.45 | 95.75 | 95.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 13:00:00 | 96.75 | 95.84 | 95.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 13:45:00 | 97.05 | 96.00 | 95.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 14:45:00 | 97.20 | 96.46 | 95.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-01 09:15:00 | 106.43 | 104.38 | 103.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 101.05 | 103.16 | 103.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 99.60 | 102.45 | 103.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 100.80 | 100.56 | 101.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 100.80 | 100.56 | 101.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 100.85 | 100.71 | 101.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:15:00 | 100.65 | 100.71 | 101.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 11:15:00 | 102.70 | 101.12 | 101.51 | SL hit (close>static) qty=1.00 sl=101.95 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 11:15:00 | 99.70 | 99.02 | 98.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 14:15:00 | 101.20 | 99.76 | 99.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 12:15:00 | 100.45 | 100.60 | 100.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 12:45:00 | 100.35 | 100.60 | 100.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 99.70 | 100.42 | 99.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:00:00 | 99.70 | 100.42 | 99.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 99.90 | 100.32 | 99.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:30:00 | 99.85 | 100.32 | 99.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 99.95 | 100.25 | 99.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 100.35 | 100.25 | 99.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 10:15:00 | 99.70 | 101.17 | 101.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 10:15:00 | 99.70 | 101.17 | 101.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 11:15:00 | 98.70 | 100.68 | 100.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 100.95 | 100.23 | 100.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 100.95 | 100.23 | 100.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 100.95 | 100.23 | 100.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:45:00 | 101.45 | 100.23 | 100.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 100.70 | 100.32 | 100.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 10:30:00 | 101.25 | 100.32 | 100.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 99.85 | 100.23 | 100.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 09:30:00 | 99.35 | 99.84 | 100.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 14:15:00 | 100.75 | 99.46 | 99.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 100.75 | 99.46 | 99.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 103.00 | 100.38 | 99.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 12:15:00 | 109.40 | 109.52 | 107.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 13:00:00 | 109.40 | 109.52 | 107.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 107.90 | 108.96 | 107.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 108.85 | 108.96 | 107.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 107.70 | 108.70 | 107.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 106.15 | 108.70 | 107.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 106.25 | 108.21 | 107.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 106.25 | 108.21 | 107.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 105.85 | 107.74 | 107.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:00:00 | 105.85 | 107.74 | 107.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 105.40 | 106.92 | 107.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 105.10 | 106.31 | 106.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 107.35 | 106.52 | 106.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 107.35 | 106.52 | 106.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 107.35 | 106.52 | 106.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:15:00 | 108.15 | 106.52 | 106.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 110.30 | 107.28 | 107.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 12:15:00 | 112.40 | 109.82 | 108.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 13:15:00 | 138.20 | 138.78 | 134.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 13:30:00 | 137.60 | 138.78 | 134.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 136.35 | 138.18 | 135.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:45:00 | 136.00 | 138.18 | 135.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 135.55 | 137.66 | 135.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:00:00 | 135.55 | 137.66 | 135.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 135.70 | 137.26 | 135.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:30:00 | 134.80 | 137.26 | 135.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 136.25 | 137.06 | 135.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 135.55 | 137.06 | 135.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 135.15 | 136.68 | 135.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 13:45:00 | 135.20 | 136.68 | 135.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 135.70 | 136.48 | 135.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 09:15:00 | 136.45 | 136.25 | 135.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 133.35 | 139.54 | 139.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 133.35 | 139.54 | 139.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 129.70 | 136.52 | 138.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 13:15:00 | 128.40 | 128.37 | 131.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 14:15:00 | 129.65 | 128.37 | 131.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 129.75 | 128.77 | 131.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 131.00 | 128.77 | 131.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 131.40 | 129.29 | 131.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:00:00 | 131.40 | 129.29 | 131.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 133.15 | 130.07 | 131.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:45:00 | 133.60 | 130.07 | 131.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 130.30 | 130.11 | 131.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:45:00 | 129.85 | 130.10 | 131.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 15:00:00 | 129.90 | 130.06 | 131.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 11:15:00 | 130.00 | 130.25 | 130.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 13:15:00 | 123.36 | 125.62 | 127.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 13:15:00 | 123.41 | 125.62 | 127.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 13:15:00 | 123.50 | 125.62 | 127.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-21 10:15:00 | 125.65 | 124.75 | 126.17 | SL hit (close>ema200) qty=0.50 sl=124.75 alert=retest2 |

### Cycle 24 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 125.35 | 125.10 | 125.09 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 11:15:00 | 124.65 | 125.02 | 125.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 12:15:00 | 123.95 | 124.80 | 124.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 13:15:00 | 124.85 | 124.81 | 124.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 14:00:00 | 124.85 | 124.81 | 124.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 124.35 | 124.72 | 124.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 123.85 | 124.63 | 124.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:45:00 | 124.10 | 124.33 | 124.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 13:15:00 | 126.20 | 124.69 | 124.73 | SL hit (close>static) qty=1.00 sl=125.15 alert=retest2 |

### Cycle 26 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 126.85 | 125.12 | 124.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 128.15 | 126.03 | 125.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 126.25 | 127.69 | 126.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 126.25 | 127.69 | 126.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 126.25 | 127.69 | 126.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 126.25 | 127.69 | 126.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 127.90 | 127.73 | 126.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 129.65 | 127.73 | 126.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 12:15:00 | 126.30 | 128.99 | 129.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 126.30 | 128.99 | 129.18 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 128.80 | 128.48 | 128.44 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 126.45 | 128.08 | 128.26 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 129.40 | 127.80 | 127.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 14:15:00 | 131.30 | 128.75 | 128.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 129.60 | 130.21 | 129.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 14:00:00 | 129.60 | 130.21 | 129.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 128.35 | 130.28 | 130.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:00:00 | 128.35 | 130.28 | 130.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 10:15:00 | 128.40 | 129.90 | 129.91 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 14:15:00 | 131.30 | 129.44 | 129.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 133.10 | 130.37 | 129.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 13:15:00 | 130.90 | 131.26 | 130.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 13:15:00 | 130.90 | 131.26 | 130.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 130.90 | 131.26 | 130.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:00:00 | 130.90 | 131.26 | 130.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 131.75 | 131.93 | 131.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 131.75 | 131.93 | 131.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 130.35 | 131.61 | 131.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:00:00 | 130.35 | 131.61 | 131.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 130.75 | 131.44 | 131.03 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 128.90 | 130.55 | 130.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 128.15 | 130.07 | 130.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 13:15:00 | 129.60 | 129.54 | 130.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 14:00:00 | 129.60 | 129.54 | 130.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 128.25 | 129.17 | 129.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:30:00 | 129.25 | 129.17 | 129.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 119.05 | 117.56 | 118.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 119.05 | 117.56 | 118.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 119.35 | 117.92 | 118.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 119.45 | 117.92 | 118.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 119.05 | 118.15 | 118.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:30:00 | 119.35 | 118.15 | 118.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 14:15:00 | 117.55 | 118.00 | 118.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 14:30:00 | 118.10 | 118.00 | 118.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 118.60 | 118.11 | 118.52 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 119.95 | 118.92 | 118.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 14:15:00 | 120.45 | 119.43 | 119.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 119.60 | 120.05 | 119.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 11:15:00 | 119.60 | 120.05 | 119.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 119.60 | 120.05 | 119.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:00:00 | 119.60 | 120.05 | 119.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 120.25 | 120.09 | 119.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 13:30:00 | 120.70 | 120.37 | 119.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 12:15:00 | 127.20 | 128.15 | 128.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 12:15:00 | 127.20 | 128.15 | 128.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 14:15:00 | 125.55 | 127.47 | 127.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 11:15:00 | 126.75 | 126.13 | 127.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-09 12:00:00 | 126.75 | 126.13 | 127.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 125.90 | 126.09 | 126.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 14:45:00 | 125.55 | 125.99 | 126.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:30:00 | 124.95 | 125.67 | 126.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 12:15:00 | 129.10 | 126.56 | 126.68 | SL hit (close>static) qty=1.00 sl=127.40 alert=retest2 |

### Cycle 36 — BUY (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 13:15:00 | 128.80 | 127.01 | 126.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 09:15:00 | 134.90 | 129.34 | 128.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 15:15:00 | 138.40 | 138.44 | 136.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 09:15:00 | 138.60 | 138.44 | 136.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 137.85 | 140.31 | 139.64 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 11:15:00 | 137.55 | 139.20 | 139.21 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 09:15:00 | 140.50 | 139.29 | 139.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 14:15:00 | 141.05 | 140.12 | 139.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 170.20 | 170.48 | 166.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 15:00:00 | 170.20 | 170.48 | 166.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 178.40 | 179.85 | 178.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 178.40 | 179.85 | 178.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 176.25 | 179.13 | 177.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 12:00:00 | 176.25 | 179.13 | 177.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 175.45 | 178.39 | 177.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 175.45 | 178.39 | 177.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 177.25 | 178.83 | 178.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:00:00 | 177.25 | 178.83 | 178.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 178.60 | 178.78 | 178.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 11:15:00 | 178.90 | 178.78 | 178.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 13:15:00 | 178.70 | 178.70 | 178.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 15:15:00 | 177.40 | 178.26 | 178.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 177.40 | 178.26 | 178.31 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 10:15:00 | 180.70 | 178.74 | 178.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 183.20 | 179.99 | 179.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 14:15:00 | 181.05 | 181.27 | 180.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 15:00:00 | 181.05 | 181.27 | 180.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 181.35 | 182.14 | 181.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 181.40 | 182.14 | 181.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 182.65 | 182.24 | 181.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:15:00 | 183.00 | 182.24 | 181.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 11:00:00 | 182.95 | 182.38 | 181.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 12:00:00 | 183.00 | 182.50 | 181.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 14:45:00 | 187.00 | 183.52 | 182.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 182.90 | 183.87 | 182.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 182.90 | 183.87 | 182.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 184.75 | 184.04 | 182.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 12:30:00 | 186.10 | 184.55 | 183.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 13:45:00 | 186.20 | 184.77 | 183.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 14:15:00 | 186.00 | 184.77 | 183.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 188.00 | 184.82 | 183.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 184.40 | 185.73 | 184.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 184.40 | 185.73 | 184.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 176.50 | 183.88 | 183.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 176.50 | 183.88 | 183.96 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 11:15:00 | 182.90 | 180.18 | 180.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 184.85 | 182.06 | 181.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 181.40 | 182.17 | 181.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 12:15:00 | 181.40 | 182.17 | 181.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 181.40 | 182.17 | 181.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 180.85 | 182.17 | 181.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 182.05 | 182.15 | 181.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 181.35 | 182.15 | 181.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 181.30 | 181.93 | 181.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 09:15:00 | 182.70 | 181.93 | 181.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-01 09:15:00 | 200.97 | 194.54 | 190.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 14:15:00 | 194.30 | 196.87 | 196.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 13:15:00 | 193.75 | 195.46 | 195.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 14:15:00 | 195.95 | 195.56 | 195.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 14:15:00 | 195.95 | 195.56 | 195.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 195.95 | 195.56 | 195.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 15:00:00 | 195.95 | 195.56 | 195.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 15:15:00 | 195.50 | 195.55 | 195.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:15:00 | 194.80 | 195.55 | 195.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 09:15:00 | 199.70 | 196.38 | 196.20 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 195.70 | 196.25 | 196.27 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 197.05 | 196.41 | 196.34 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 15:15:00 | 195.85 | 196.37 | 196.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 10:15:00 | 194.40 | 195.81 | 196.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 199.35 | 195.82 | 195.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 14:15:00 | 199.35 | 195.82 | 195.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 199.35 | 195.82 | 195.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 199.35 | 195.82 | 195.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 15:15:00 | 199.90 | 196.64 | 196.29 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 196.40 | 197.10 | 197.17 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 09:15:00 | 200.00 | 197.55 | 197.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-17 09:15:00 | 210.95 | 203.63 | 201.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 12:15:00 | 205.60 | 205.98 | 203.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 13:00:00 | 205.60 | 205.98 | 203.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 206.10 | 206.26 | 203.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 14:30:00 | 205.50 | 206.26 | 203.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 201.95 | 205.77 | 204.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 199.95 | 205.77 | 204.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 207.65 | 206.14 | 204.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 12:00:00 | 208.80 | 206.67 | 204.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:00:00 | 209.45 | 207.40 | 205.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:30:00 | 209.45 | 208.36 | 206.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 12:15:00 | 209.25 | 214.59 | 215.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 209.25 | 214.59 | 215.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 206.70 | 213.01 | 214.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 208.65 | 208.21 | 210.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 13:45:00 | 207.90 | 208.21 | 210.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 213.90 | 209.84 | 210.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:45:00 | 215.30 | 209.84 | 210.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 11:15:00 | 216.40 | 212.21 | 211.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 13:15:00 | 218.10 | 214.19 | 212.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 10:15:00 | 226.90 | 228.01 | 224.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 11:00:00 | 226.90 | 228.01 | 224.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 227.90 | 228.40 | 226.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 227.90 | 228.40 | 226.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 225.50 | 227.82 | 226.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 223.75 | 227.82 | 226.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 225.85 | 227.43 | 226.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:30:00 | 223.95 | 227.43 | 226.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 235.60 | 236.30 | 233.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 233.45 | 236.30 | 233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 232.65 | 235.36 | 233.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 09:15:00 | 236.45 | 232.97 | 232.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 11:15:00 | 230.60 | 232.61 | 232.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 11:15:00 | 230.60 | 232.61 | 232.72 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 11:15:00 | 235.45 | 232.61 | 232.52 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 12:15:00 | 231.80 | 232.45 | 232.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 230.60 | 231.93 | 232.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 227.35 | 226.83 | 228.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 14:15:00 | 227.35 | 226.83 | 228.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 227.35 | 226.83 | 228.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 15:00:00 | 227.35 | 226.83 | 228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 218.50 | 225.35 | 227.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 11:45:00 | 216.95 | 222.73 | 226.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 15:00:00 | 215.55 | 220.87 | 224.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 12:15:00 | 206.10 | 214.70 | 219.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 14:15:00 | 204.77 | 213.54 | 218.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-14 09:15:00 | 214.80 | 213.64 | 217.51 | SL hit (close>ema200) qty=0.50 sl=213.64 alert=retest2 |

### Cycle 56 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 223.80 | 219.79 | 219.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 228.30 | 221.49 | 220.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 226.95 | 226.97 | 224.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 12:45:00 | 226.50 | 226.97 | 224.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 225.35 | 226.56 | 225.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 228.80 | 226.56 | 225.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 15:15:00 | 226.05 | 227.09 | 226.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 10:45:00 | 226.50 | 228.85 | 227.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 12:30:00 | 226.25 | 227.93 | 227.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 223.80 | 227.11 | 227.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 223.80 | 227.11 | 227.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 222.90 | 226.26 | 226.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 225.80 | 225.60 | 226.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 225.80 | 225.60 | 226.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 225.80 | 225.60 | 226.43 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 229.25 | 226.81 | 226.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 231.85 | 227.82 | 227.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 14:15:00 | 229.70 | 230.04 | 228.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 15:00:00 | 229.70 | 230.04 | 228.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 229.10 | 229.85 | 228.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 230.10 | 229.85 | 228.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 227.80 | 229.44 | 228.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 226.75 | 229.44 | 228.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 228.35 | 229.22 | 228.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:30:00 | 227.15 | 229.22 | 228.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 228.30 | 229.04 | 228.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:45:00 | 227.70 | 229.04 | 228.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 230.40 | 229.31 | 228.74 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 09:15:00 | 225.30 | 228.12 | 228.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 11:15:00 | 224.00 | 226.77 | 227.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 228.15 | 225.44 | 226.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 228.15 | 225.44 | 226.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 228.15 | 225.44 | 226.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:45:00 | 227.10 | 225.44 | 226.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 225.30 | 225.41 | 226.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:30:00 | 227.75 | 225.41 | 226.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 221.20 | 224.49 | 225.77 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 228.05 | 226.15 | 225.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 15:15:00 | 229.15 | 226.75 | 226.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 257.35 | 258.63 | 248.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:00:00 | 257.35 | 258.63 | 248.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 253.00 | 258.68 | 253.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 253.00 | 258.68 | 253.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 250.80 | 257.11 | 253.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 250.80 | 257.11 | 253.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 11:15:00 | 252.20 | 256.12 | 253.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 14:15:00 | 254.20 | 254.73 | 253.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 09:15:00 | 252.00 | 255.24 | 255.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 252.00 | 255.24 | 255.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 246.65 | 253.52 | 254.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 15:15:00 | 217.20 | 216.96 | 224.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-18 09:15:00 | 219.10 | 216.96 | 224.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 223.40 | 219.35 | 223.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:00:00 | 223.40 | 219.35 | 223.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 223.90 | 220.26 | 223.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:00:00 | 223.90 | 220.26 | 223.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 224.40 | 221.09 | 223.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 14:30:00 | 225.55 | 221.09 | 223.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 223.00 | 221.64 | 222.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 13:00:00 | 223.00 | 221.64 | 222.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 220.30 | 221.37 | 222.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 13:30:00 | 222.70 | 221.37 | 222.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 216.70 | 219.62 | 221.37 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 15:15:00 | 224.85 | 222.25 | 221.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 234.25 | 224.65 | 223.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 10:15:00 | 251.30 | 251.32 | 248.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 10:30:00 | 250.75 | 251.32 | 248.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 251.30 | 251.69 | 249.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:30:00 | 250.55 | 251.69 | 249.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 250.35 | 251.42 | 250.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:45:00 | 250.55 | 251.42 | 250.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 250.85 | 251.31 | 250.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:45:00 | 250.35 | 251.31 | 250.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 248.15 | 251.27 | 250.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 248.15 | 251.27 | 250.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 250.80 | 251.17 | 250.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 13:30:00 | 251.05 | 251.01 | 250.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 14:30:00 | 251.05 | 251.07 | 250.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 10:15:00 | 251.65 | 250.94 | 250.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 14:15:00 | 256.15 | 259.78 | 260.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 256.15 | 259.78 | 260.13 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 260.50 | 259.59 | 259.52 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 252.25 | 258.07 | 258.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 250.70 | 255.40 | 257.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 255.20 | 255.01 | 256.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 12:45:00 | 255.10 | 255.01 | 256.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 255.50 | 254.92 | 256.18 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 258.55 | 256.81 | 256.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 260.75 | 257.60 | 257.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 276.05 | 276.86 | 272.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 10:00:00 | 276.05 | 276.86 | 272.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 283.80 | 300.02 | 295.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 283.80 | 300.02 | 295.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 293.00 | 298.62 | 295.06 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 15:15:00 | 288.80 | 293.07 | 293.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 287.35 | 291.93 | 292.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 287.35 | 284.29 | 287.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 287.35 | 284.29 | 287.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 287.35 | 284.29 | 287.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 287.35 | 284.29 | 287.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 286.60 | 284.75 | 287.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 282.15 | 286.28 | 287.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 09:15:00 | 268.04 | 277.35 | 281.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 277.00 | 274.33 | 277.06 | SL hit (close>ema200) qty=0.50 sl=274.33 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 283.20 | 279.24 | 278.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 286.40 | 281.17 | 279.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 290.85 | 291.06 | 288.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:15:00 | 289.60 | 291.06 | 288.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 288.50 | 290.55 | 288.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 289.20 | 290.55 | 288.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 295.55 | 291.55 | 289.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:30:00 | 297.30 | 294.78 | 292.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 14:15:00 | 297.20 | 294.78 | 292.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 14:15:00 | 304.80 | 305.36 | 305.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 304.80 | 305.36 | 305.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 15:15:00 | 304.10 | 305.11 | 305.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 305.00 | 304.86 | 305.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 10:15:00 | 305.00 | 304.86 | 305.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 305.00 | 304.86 | 305.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 305.00 | 304.86 | 305.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 11:15:00 | 307.35 | 305.36 | 305.33 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 299.15 | 304.58 | 305.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 294.10 | 299.34 | 301.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 293.80 | 293.57 | 296.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:00:00 | 293.80 | 293.57 | 296.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 294.95 | 293.85 | 296.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 295.80 | 293.85 | 296.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 293.15 | 292.90 | 294.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 294.30 | 292.90 | 294.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 289.70 | 292.04 | 293.68 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 298.75 | 294.24 | 294.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 299.60 | 295.31 | 294.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 293.35 | 304.95 | 301.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 293.35 | 304.95 | 301.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 293.35 | 304.95 | 301.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 288.55 | 304.95 | 301.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 264.75 | 296.91 | 298.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 235.55 | 284.64 | 292.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 254.65 | 252.60 | 266.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 254.65 | 252.60 | 266.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 288.00 | 260.32 | 266.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 288.85 | 260.32 | 266.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 281.45 | 272.18 | 271.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 287.35 | 282.96 | 278.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 284.25 | 284.50 | 281.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 294.50 | 284.56 | 281.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 12:15:00 | 309.23 | 303.16 | 298.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 299.50 | 304.38 | 302.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 299.50 | 304.38 | 302.78 | SL hit (close<ema200) qty=0.50 sl=304.38 alert=retest1 |

### Cycle 75 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 299.45 | 301.85 | 301.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 296.70 | 300.82 | 301.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 296.50 | 295.15 | 297.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 296.50 | 295.15 | 297.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 296.50 | 295.15 | 297.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:45:00 | 298.90 | 295.15 | 297.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 297.70 | 295.66 | 297.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 297.40 | 295.66 | 297.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 297.15 | 295.96 | 297.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:45:00 | 297.15 | 295.96 | 297.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 294.35 | 295.64 | 297.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 292.35 | 295.43 | 296.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 298.30 | 296.01 | 295.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 298.30 | 296.01 | 295.73 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 293.25 | 295.66 | 295.72 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 297.75 | 296.08 | 295.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 298.20 | 296.50 | 296.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 300.45 | 300.99 | 299.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 300.45 | 300.99 | 299.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 300.00 | 300.83 | 299.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 15:00:00 | 302.45 | 300.30 | 299.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 11:15:00 | 296.90 | 299.80 | 299.58 | SL hit (close<static) qty=1.00 sl=298.50 alert=retest2 |

### Cycle 79 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 295.35 | 298.91 | 299.20 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 301.60 | 299.17 | 299.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 312.20 | 302.51 | 300.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 323.80 | 327.12 | 323.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 323.80 | 327.12 | 323.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 323.80 | 327.12 | 323.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 323.80 | 327.12 | 323.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 322.55 | 326.21 | 323.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 319.00 | 326.21 | 323.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 326.55 | 326.27 | 323.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 331.80 | 326.85 | 324.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 15:15:00 | 325.50 | 327.83 | 328.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 325.50 | 327.83 | 328.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 10:15:00 | 324.25 | 325.70 | 326.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 301.40 | 299.69 | 306.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 301.40 | 299.69 | 306.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 304.10 | 302.22 | 305.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 305.30 | 302.22 | 305.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 305.55 | 302.89 | 305.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 305.55 | 302.89 | 305.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 305.00 | 303.31 | 305.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 302.80 | 303.31 | 305.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 301.05 | 302.86 | 305.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 293.50 | 304.53 | 305.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 15:15:00 | 307.75 | 305.58 | 305.76 | SL hit (close>static) qty=1.00 sl=307.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 312.45 | 306.95 | 306.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 314.80 | 311.18 | 309.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 318.50 | 321.73 | 319.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 318.50 | 321.73 | 319.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 318.50 | 321.73 | 319.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 318.50 | 321.73 | 319.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 317.15 | 320.82 | 319.01 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 315.10 | 318.13 | 318.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 12:15:00 | 314.55 | 317.41 | 317.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 318.90 | 316.64 | 317.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 318.90 | 316.64 | 317.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 318.90 | 316.64 | 317.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 318.90 | 316.64 | 317.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 315.25 | 316.36 | 317.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:45:00 | 311.95 | 315.05 | 316.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 296.35 | 302.54 | 307.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 299.05 | 294.53 | 299.79 | SL hit (close>ema200) qty=0.50 sl=294.53 alert=retest2 |

### Cycle 84 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 304.80 | 298.64 | 298.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 305.60 | 300.03 | 298.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 300.00 | 300.48 | 299.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 13:15:00 | 300.00 | 300.48 | 299.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 300.00 | 300.48 | 299.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:30:00 | 301.10 | 300.48 | 299.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 297.60 | 299.90 | 299.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 297.60 | 299.90 | 299.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 297.50 | 299.42 | 299.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 303.35 | 299.42 | 299.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:00:00 | 298.00 | 300.86 | 300.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 14:15:00 | 299.00 | 300.11 | 300.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 299.00 | 300.11 | 300.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 10:15:00 | 297.20 | 299.00 | 299.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 291.25 | 291.04 | 293.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 291.25 | 291.04 | 293.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 291.25 | 291.04 | 293.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 290.25 | 291.04 | 293.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 296.85 | 292.85 | 293.38 | SL hit (close>static) qty=1.00 sl=295.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 298.60 | 294.52 | 294.08 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 292.75 | 294.23 | 294.27 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 13:15:00 | 295.80 | 294.44 | 294.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 296.90 | 295.43 | 294.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 10:15:00 | 298.90 | 299.13 | 297.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 11:00:00 | 298.90 | 299.13 | 297.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 295.55 | 298.43 | 297.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 295.55 | 298.43 | 297.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 295.80 | 297.91 | 297.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 296.70 | 297.91 | 297.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 297.65 | 297.96 | 297.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 297.65 | 297.96 | 297.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 296.80 | 297.73 | 297.74 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 298.40 | 297.71 | 297.70 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 10:15:00 | 297.50 | 297.66 | 297.68 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 298.20 | 297.77 | 297.73 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 297.15 | 297.65 | 297.68 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 298.15 | 297.78 | 297.73 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 296.80 | 297.55 | 297.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 12:15:00 | 296.05 | 297.00 | 297.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 291.50 | 291.31 | 293.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 291.30 | 291.31 | 293.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 290.00 | 291.05 | 293.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 289.40 | 290.51 | 291.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:15:00 | 274.93 | 279.97 | 283.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-09 09:15:00 | 260.46 | 266.29 | 271.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 96 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 266.40 | 264.06 | 264.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 267.35 | 265.10 | 264.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 265.25 | 267.54 | 266.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 265.25 | 267.54 | 266.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 265.25 | 267.54 | 266.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 265.25 | 267.54 | 266.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 266.15 | 267.26 | 266.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:00:00 | 267.00 | 267.21 | 266.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 267.40 | 266.62 | 266.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 264.85 | 266.43 | 266.39 | SL hit (close<static) qty=1.00 sl=264.90 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 264.25 | 265.99 | 266.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 259.20 | 264.36 | 265.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 261.65 | 258.68 | 261.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 261.65 | 258.68 | 261.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 261.65 | 258.68 | 261.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 261.65 | 258.68 | 261.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 261.95 | 259.33 | 261.18 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 266.00 | 262.91 | 262.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 09:15:00 | 269.65 | 264.79 | 263.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 280.40 | 280.94 | 277.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:45:00 | 280.75 | 280.94 | 277.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 279.75 | 280.46 | 278.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:45:00 | 279.00 | 280.46 | 278.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 277.00 | 283.91 | 282.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 275.85 | 283.91 | 282.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 277.65 | 282.66 | 282.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 11:30:00 | 278.05 | 282.09 | 281.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 279.45 | 281.64 | 281.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 279.45 | 281.64 | 281.68 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 284.60 | 281.60 | 281.59 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 279.90 | 281.40 | 281.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 275.60 | 280.24 | 281.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 262.30 | 260.23 | 264.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 262.30 | 260.23 | 264.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 265.40 | 261.26 | 264.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 266.40 | 261.26 | 264.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 263.00 | 261.61 | 264.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 265.25 | 261.61 | 264.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 264.95 | 262.60 | 264.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 264.95 | 262.60 | 264.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 265.20 | 263.12 | 264.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 265.20 | 263.12 | 264.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 266.55 | 263.81 | 264.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 265.00 | 263.81 | 264.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 264.35 | 264.25 | 265.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 267.20 | 264.84 | 265.22 | SL hit (close>static) qty=1.00 sl=266.95 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 270.50 | 266.04 | 265.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 272.95 | 270.02 | 268.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 270.00 | 270.25 | 268.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:45:00 | 270.30 | 270.25 | 268.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 268.80 | 270.02 | 268.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 268.80 | 270.02 | 268.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 269.80 | 269.98 | 268.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 271.35 | 270.20 | 269.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 270.60 | 270.20 | 269.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 268.35 | 269.82 | 269.25 | SL hit (close<static) qty=1.00 sl=268.50 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 268.00 | 268.99 | 268.99 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 269.25 | 269.04 | 269.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 271.40 | 269.84 | 269.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 270.10 | 270.50 | 269.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 270.10 | 270.50 | 269.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 270.10 | 270.50 | 269.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 270.20 | 270.50 | 269.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 269.85 | 270.37 | 269.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:45:00 | 269.35 | 270.37 | 269.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 270.05 | 270.30 | 269.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 268.85 | 270.30 | 269.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 269.80 | 270.20 | 269.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 269.75 | 270.20 | 269.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 269.60 | 270.08 | 269.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 265.35 | 270.08 | 269.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 258.85 | 267.84 | 268.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 255.80 | 261.76 | 265.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 12:15:00 | 229.00 | 227.44 | 232.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 13:00:00 | 229.00 | 227.44 | 232.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 220.80 | 219.50 | 222.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 220.80 | 219.50 | 222.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 231.35 | 221.87 | 223.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 238.50 | 221.87 | 223.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 231.10 | 223.72 | 223.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 232.05 | 223.72 | 223.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 09:15:00 | 230.85 | 225.15 | 224.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 235.50 | 230.15 | 227.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 236.95 | 237.38 | 233.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 236.95 | 237.38 | 233.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 237.30 | 237.16 | 233.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:45:00 | 239.10 | 237.65 | 235.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:30:00 | 239.25 | 237.97 | 235.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 239.25 | 237.97 | 235.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 242.13 | 238.97 | 236.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 233.94 | 238.45 | 236.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 233.94 | 238.45 | 236.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 232.10 | 237.18 | 236.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 10:15:00 | 232.10 | 237.18 | 236.24 | SL hit (close<static) qty=1.00 sl=232.85 alert=retest2 |

### Cycle 107 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 232.66 | 235.49 | 235.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 230.49 | 232.95 | 234.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 234.08 | 232.96 | 233.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 234.08 | 232.96 | 233.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 234.08 | 232.96 | 233.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 234.31 | 232.96 | 233.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 236.14 | 233.60 | 234.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 236.14 | 233.60 | 234.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 235.05 | 233.89 | 234.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 240.93 | 233.89 | 234.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 244.75 | 236.06 | 235.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 247.77 | 238.40 | 236.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 245.49 | 245.73 | 242.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 245.49 | 245.73 | 242.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 243.77 | 244.95 | 243.18 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 238.44 | 241.77 | 242.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 238.00 | 240.64 | 241.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 242.09 | 240.42 | 241.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 242.09 | 240.42 | 241.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 242.09 | 240.42 | 241.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 242.09 | 240.42 | 241.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 240.40 | 240.42 | 241.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 241.99 | 240.42 | 241.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 241.30 | 240.59 | 241.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 241.30 | 240.59 | 241.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 239.20 | 240.31 | 240.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 238.39 | 240.12 | 240.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 226.47 | 231.80 | 235.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 225.90 | 225.82 | 230.11 | SL hit (close>ema200) qty=0.50 sl=225.82 alert=retest2 |

### Cycle 110 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 231.00 | 227.21 | 226.77 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 219.60 | 225.46 | 226.18 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 15:15:00 | 228.00 | 226.55 | 226.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 229.02 | 227.05 | 226.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 243.15 | 243.51 | 240.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 243.53 | 243.51 | 240.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 249.15 | 250.46 | 248.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 249.15 | 250.46 | 248.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 248.75 | 250.24 | 248.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 248.75 | 250.24 | 248.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 249.20 | 250.03 | 248.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 248.65 | 250.03 | 248.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 249.25 | 249.87 | 249.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:45:00 | 249.10 | 249.87 | 249.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 249.35 | 249.67 | 249.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:45:00 | 249.15 | 249.67 | 249.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 249.75 | 249.69 | 249.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:45:00 | 249.15 | 249.69 | 249.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 251.30 | 252.01 | 251.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 251.30 | 252.01 | 251.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 250.65 | 251.74 | 251.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 250.45 | 251.74 | 251.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 251.40 | 251.67 | 251.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 250.90 | 251.67 | 251.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 251.00 | 251.54 | 251.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 251.15 | 251.54 | 251.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 251.70 | 251.57 | 251.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 252.45 | 251.57 | 251.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 250.50 | 251.36 | 251.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 250.50 | 251.36 | 251.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 250.45 | 251.17 | 251.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 249.60 | 251.17 | 251.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 250.20 | 250.98 | 250.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 248.65 | 250.98 | 250.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 251.60 | 251.07 | 250.99 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 249.00 | 250.69 | 250.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 15:15:00 | 248.95 | 249.76 | 250.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 09:15:00 | 250.15 | 249.84 | 250.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 250.15 | 249.84 | 250.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 250.15 | 249.84 | 250.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 252.35 | 249.84 | 250.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 249.15 | 249.70 | 250.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 247.70 | 249.69 | 250.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 252.80 | 250.31 | 250.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 252.80 | 250.31 | 250.27 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 248.60 | 250.25 | 250.33 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 252.75 | 250.47 | 250.32 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 246.00 | 250.07 | 250.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 13:15:00 | 245.50 | 248.49 | 249.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 243.70 | 243.30 | 245.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 243.70 | 243.30 | 245.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 243.70 | 243.30 | 245.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 244.30 | 243.30 | 245.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 244.70 | 243.66 | 244.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 244.70 | 243.66 | 244.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 246.10 | 244.14 | 245.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 246.10 | 244.14 | 245.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 247.50 | 244.82 | 245.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:45:00 | 247.75 | 244.82 | 245.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 248.85 | 245.97 | 245.73 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 242.40 | 245.22 | 245.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 15:15:00 | 242.25 | 244.62 | 245.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 240.35 | 239.60 | 241.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 12:00:00 | 240.35 | 239.60 | 241.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 241.80 | 240.31 | 241.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:30:00 | 241.35 | 240.31 | 241.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 242.55 | 240.76 | 241.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 241.90 | 240.76 | 241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 241.95 | 241.00 | 241.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 240.50 | 241.00 | 241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 239.50 | 240.70 | 241.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 243.25 | 240.70 | 241.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 242.45 | 241.05 | 241.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 242.45 | 241.05 | 241.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 244.05 | 241.65 | 241.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 244.05 | 241.65 | 241.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 240.55 | 238.90 | 240.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 240.55 | 238.90 | 240.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 241.10 | 239.34 | 240.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:30:00 | 240.50 | 239.34 | 240.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 242.15 | 239.90 | 240.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:00:00 | 242.15 | 239.90 | 240.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 13:15:00 | 242.80 | 240.78 | 240.72 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 239.35 | 240.95 | 240.98 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 242.35 | 241.00 | 240.90 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 10:15:00 | 239.05 | 240.80 | 240.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 14:15:00 | 235.50 | 238.95 | 239.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 231.50 | 229.59 | 232.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 231.50 | 229.59 | 232.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 232.38 | 230.39 | 231.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:00:00 | 232.38 | 230.39 | 231.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 233.06 | 230.93 | 231.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:00:00 | 233.06 | 230.93 | 231.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 233.30 | 231.40 | 231.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:45:00 | 234.00 | 231.40 | 231.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 230.17 | 231.36 | 231.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 232.07 | 231.36 | 231.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 232.70 | 231.12 | 231.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:45:00 | 232.60 | 231.12 | 231.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 232.51 | 231.40 | 231.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:15:00 | 232.90 | 231.40 | 231.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 232.90 | 231.70 | 231.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:15:00 | 235.07 | 231.70 | 231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 233.99 | 232.16 | 231.91 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 230.13 | 231.59 | 231.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 226.50 | 230.32 | 231.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 225.71 | 223.07 | 225.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 13:15:00 | 225.71 | 223.07 | 225.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 225.71 | 223.07 | 225.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 225.71 | 223.07 | 225.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 224.95 | 223.45 | 225.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 221.57 | 223.65 | 225.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 210.49 | 217.09 | 219.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 199.41 | 207.17 | 212.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 207.95 | 202.90 | 202.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 210.29 | 204.38 | 203.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 213.80 | 215.40 | 213.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 213.80 | 215.40 | 213.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 212.21 | 214.76 | 212.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 212.21 | 214.76 | 212.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 213.98 | 214.61 | 213.02 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 209.20 | 212.15 | 212.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 205.78 | 210.88 | 211.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 206.08 | 205.60 | 208.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 206.08 | 205.60 | 208.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 207.22 | 205.78 | 207.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 207.22 | 205.78 | 207.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 207.00 | 206.02 | 207.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 206.93 | 206.02 | 207.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 203.72 | 205.61 | 206.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 201.97 | 204.12 | 205.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 191.87 | 195.23 | 199.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 195.00 | 191.71 | 195.01 | SL hit (close>ema200) qty=0.50 sl=191.71 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 199.92 | 196.29 | 196.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 200.50 | 197.13 | 196.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 195.64 | 197.38 | 196.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 195.64 | 197.38 | 196.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 195.64 | 197.38 | 196.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 195.64 | 197.38 | 196.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 196.47 | 197.20 | 196.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 195.34 | 197.20 | 196.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 196.20 | 197.00 | 196.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 199.55 | 197.00 | 196.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 205.43 | 207.14 | 203.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 206.25 | 207.14 | 203.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 199.03 | 205.51 | 203.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 199.03 | 205.51 | 203.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 202.41 | 204.89 | 203.17 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 192.17 | 200.93 | 201.67 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 205.73 | 200.82 | 200.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 208.10 | 204.92 | 202.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 207.60 | 207.82 | 205.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:30:00 | 208.50 | 207.82 | 205.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 206.10 | 207.39 | 205.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 206.47 | 207.39 | 205.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 205.44 | 207.00 | 205.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 205.44 | 207.00 | 205.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 204.50 | 206.50 | 205.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 204.60 | 206.50 | 205.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 204.77 | 206.16 | 205.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 204.75 | 206.16 | 205.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 205.38 | 205.54 | 205.37 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 203.75 | 205.19 | 205.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 201.86 | 204.33 | 204.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 09:15:00 | 204.83 | 204.00 | 204.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 204.83 | 204.00 | 204.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 204.83 | 204.00 | 204.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 12:45:00 | 202.88 | 203.88 | 204.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:30:00 | 203.01 | 203.52 | 204.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 202.09 | 203.52 | 204.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 15:00:00 | 203.08 | 203.43 | 204.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 196.78 | 201.94 | 203.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 195.50 | 200.85 | 202.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 192.74 | 197.07 | 199.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 192.86 | 197.07 | 199.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 191.99 | 197.07 | 199.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 192.93 | 197.07 | 199.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:30:00 | 194.19 | 197.07 | 199.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 10:15:00 | 197.95 | 197.24 | 199.54 | SL hit (close>ema200) qty=0.50 sl=197.24 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 204.20 | 200.48 | 200.16 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 195.24 | 199.75 | 200.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 193.73 | 198.55 | 199.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 193.80 | 193.18 | 195.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 193.80 | 193.18 | 195.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 188.12 | 192.23 | 194.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 186.48 | 191.08 | 193.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 185.74 | 190.08 | 193.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:00:00 | 186.42 | 187.81 | 191.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 194.60 | 189.41 | 190.81 | SL hit (close>static) qty=1.00 sl=194.50 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 195.32 | 192.06 | 191.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 199.22 | 193.96 | 192.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 197.70 | 198.05 | 195.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:15:00 | 197.79 | 198.05 | 195.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 197.97 | 197.74 | 196.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:30:00 | 199.44 | 197.92 | 196.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 193.37 | 196.49 | 196.12 | SL hit (close<static) qty=1.00 sl=195.59 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 192.92 | 195.77 | 195.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 191.33 | 193.96 | 194.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 193.56 | 191.86 | 193.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 13:15:00 | 193.56 | 191.86 | 193.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 193.56 | 191.86 | 193.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 193.56 | 191.86 | 193.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 194.65 | 192.42 | 193.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:30:00 | 195.45 | 192.42 | 193.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 194.00 | 192.73 | 193.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 192.28 | 192.73 | 193.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 184.36 | 180.85 | 183.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 184.36 | 180.85 | 183.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 185.49 | 181.78 | 183.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 185.49 | 181.78 | 183.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 185.66 | 182.55 | 184.01 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 189.06 | 184.84 | 184.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 190.45 | 186.89 | 185.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 197.19 | 197.87 | 195.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 199.83 | 197.87 | 195.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 196.80 | 197.59 | 196.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 197.79 | 197.59 | 196.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 196.17 | 197.21 | 196.42 | SL hit (close<static) qty=1.00 sl=196.20 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 193.83 | 195.89 | 195.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 192.35 | 195.18 | 195.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 194.09 | 193.79 | 194.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 12:00:00 | 194.09 | 193.79 | 194.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 193.60 | 193.56 | 194.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 193.44 | 193.56 | 194.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 194.99 | 193.85 | 194.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 193.65 | 193.85 | 194.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 191.45 | 193.37 | 194.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 190.80 | 192.90 | 193.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 195.95 | 193.77 | 193.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 195.95 | 193.77 | 193.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 198.76 | 195.62 | 194.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 203.50 | 204.56 | 202.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 203.50 | 204.56 | 202.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 213.67 | 216.59 | 214.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 213.67 | 216.59 | 214.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 213.20 | 215.91 | 214.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 214.49 | 215.91 | 214.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 212.75 | 215.28 | 214.33 | SL hit (close<static) qty=1.00 sl=212.85 alert=retest2 |

### Cycle 139 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 211.42 | 213.50 | 213.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 210.49 | 212.90 | 213.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 11:15:00 | 213.63 | 212.45 | 212.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 11:15:00 | 213.63 | 212.45 | 212.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 213.63 | 212.45 | 212.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 213.63 | 212.45 | 212.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 213.51 | 212.67 | 213.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 13:30:00 | 212.78 | 212.58 | 212.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 215.99 | 213.39 | 213.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 215.99 | 213.39 | 213.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 220.31 | 215.28 | 214.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 15:15:00 | 215.15 | 216.06 | 215.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 09:15:00 | 213.00 | 216.06 | 215.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 213.10 | 215.46 | 214.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:15:00 | 211.44 | 215.46 | 214.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 210.58 | 214.49 | 214.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 208.10 | 211.83 | 213.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 211.74 | 211.55 | 212.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:30:00 | 211.93 | 211.55 | 212.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 211.17 | 211.48 | 212.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 212.05 | 211.48 | 212.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 212.70 | 211.72 | 212.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 212.70 | 211.72 | 212.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 212.50 | 211.88 | 212.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 213.34 | 211.88 | 212.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 211.99 | 211.90 | 212.43 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 215.50 | 213.04 | 212.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 218.43 | 214.83 | 213.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 215.45 | 215.75 | 214.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 215.45 | 215.75 | 214.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 215.54 | 216.25 | 215.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 215.55 | 216.25 | 215.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 214.63 | 215.93 | 215.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 213.20 | 215.93 | 215.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 214.40 | 215.62 | 215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 205.27 | 215.62 | 215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 201.15 | 212.73 | 213.80 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 212.40 | 211.02 | 210.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 215.87 | 212.21 | 211.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 213.32 | 213.82 | 212.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 213.32 | 213.82 | 212.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 212.85 | 213.52 | 212.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 217.15 | 213.52 | 212.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 222.71 | 228.23 | 228.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 222.71 | 228.23 | 228.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 11:15:00 | 220.49 | 225.52 | 226.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 227.42 | 224.08 | 225.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 227.42 | 224.08 | 225.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 227.42 | 224.08 | 225.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 227.42 | 224.08 | 225.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 227.40 | 224.75 | 225.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 227.40 | 224.75 | 225.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 228.19 | 226.46 | 226.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 230.68 | 227.30 | 226.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 230.00 | 230.83 | 229.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 230.00 | 230.83 | 229.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 230.00 | 230.83 | 229.54 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 227.13 | 228.93 | 229.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 225.74 | 228.29 | 228.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 230.36 | 228.70 | 228.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 230.36 | 228.70 | 228.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 230.36 | 228.70 | 228.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 230.36 | 228.70 | 228.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 228.26 | 228.61 | 228.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 227.60 | 228.61 | 228.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:00:00 | 226.85 | 228.26 | 228.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 229.42 | 228.10 | 227.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 229.42 | 228.10 | 227.96 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 224.90 | 227.46 | 227.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 223.83 | 226.36 | 227.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 222.32 | 221.49 | 223.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 222.32 | 221.49 | 223.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 225.16 | 222.66 | 223.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 225.16 | 222.66 | 223.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 224.80 | 223.09 | 223.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 225.08 | 223.09 | 223.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 226.90 | 218.97 | 219.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 226.15 | 218.97 | 219.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 229.90 | 221.16 | 220.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 230.95 | 223.12 | 221.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 235.63 | 236.92 | 232.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 235.63 | 236.92 | 232.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 245.00 | 248.69 | 246.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 245.00 | 248.69 | 246.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 244.70 | 247.89 | 246.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 244.70 | 247.89 | 246.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 247.24 | 246.59 | 246.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 246.63 | 246.59 | 246.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 247.70 | 247.16 | 246.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 246.40 | 247.16 | 246.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 246.52 | 247.03 | 246.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 246.52 | 247.03 | 246.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 244.68 | 246.56 | 246.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 244.06 | 246.56 | 246.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 244.65 | 246.18 | 246.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 243.23 | 246.18 | 246.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 246.20 | 246.75 | 246.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 246.20 | 246.75 | 246.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 248.55 | 247.11 | 246.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 248.90 | 247.76 | 247.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 250.01 | 247.90 | 247.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 262.52 | 263.58 | 263.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 262.52 | 263.58 | 263.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 259.77 | 262.82 | 263.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 261.35 | 261.28 | 262.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:45:00 | 261.10 | 261.28 | 262.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 261.85 | 261.39 | 262.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 262.15 | 261.39 | 262.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 257.65 | 260.62 | 261.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 256.30 | 259.09 | 260.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 255.80 | 259.09 | 260.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 255.70 | 254.12 | 254.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 258.75 | 255.65 | 255.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 258.75 | 255.65 | 255.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 260.65 | 257.94 | 256.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 15:15:00 | 260.20 | 260.26 | 258.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:15:00 | 258.05 | 260.26 | 258.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 258.75 | 259.96 | 258.70 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 257.20 | 258.21 | 258.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 255.55 | 257.68 | 258.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 254.10 | 253.74 | 255.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 254.10 | 253.74 | 255.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 254.65 | 253.26 | 254.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 254.65 | 253.26 | 254.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 255.50 | 253.71 | 254.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 255.50 | 253.71 | 254.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 254.70 | 253.91 | 254.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 254.00 | 253.91 | 254.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 254.20 | 253.96 | 254.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 253.75 | 253.23 | 253.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 253.10 | 250.64 | 250.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 253.10 | 250.64 | 250.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 254.95 | 251.91 | 251.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 264.40 | 264.99 | 263.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 264.40 | 264.99 | 263.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 263.90 | 264.77 | 263.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 266.40 | 264.77 | 263.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:30:00 | 264.85 | 265.23 | 264.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 264.45 | 264.66 | 264.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 262.00 | 264.25 | 264.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 262.00 | 264.25 | 264.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 261.75 | 263.75 | 264.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 261.65 | 259.73 | 260.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 261.65 | 259.73 | 260.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 261.65 | 259.73 | 260.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 262.15 | 259.73 | 260.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 262.60 | 260.31 | 261.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 262.60 | 260.31 | 261.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 259.55 | 260.24 | 260.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 258.40 | 260.01 | 260.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 258.30 | 258.21 | 258.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 263.50 | 259.27 | 258.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 263.50 | 259.27 | 258.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 264.30 | 260.28 | 259.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 261.30 | 261.43 | 260.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 10:15:00 | 260.85 | 261.31 | 260.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 260.85 | 261.31 | 260.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 260.85 | 261.31 | 260.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 260.50 | 261.20 | 260.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 260.60 | 261.20 | 260.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 261.20 | 261.20 | 260.68 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 258.40 | 260.13 | 260.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 12:15:00 | 256.85 | 257.59 | 258.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 14:15:00 | 257.65 | 257.47 | 258.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 257.65 | 257.47 | 258.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 255.60 | 257.14 | 257.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:00:00 | 254.60 | 256.63 | 257.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:00:00 | 254.60 | 255.46 | 256.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 254.00 | 254.88 | 256.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 15:15:00 | 254.35 | 253.52 | 253.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 254.35 | 253.52 | 253.46 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 252.95 | 253.41 | 253.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 252.05 | 253.14 | 253.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 13:15:00 | 250.50 | 250.34 | 251.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:00:00 | 250.50 | 250.34 | 251.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 251.30 | 250.53 | 251.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 251.30 | 250.53 | 251.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 250.55 | 250.53 | 251.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 250.90 | 250.53 | 251.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 249.15 | 250.26 | 251.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 247.45 | 249.95 | 250.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 235.08 | 238.28 | 241.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 239.10 | 237.84 | 240.73 | SL hit (close>ema200) qty=0.50 sl=237.84 alert=retest2 |

### Cycle 160 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 241.64 | 237.59 | 237.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 244.89 | 239.05 | 237.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 240.50 | 244.13 | 241.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 240.50 | 244.13 | 241.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 240.50 | 244.13 | 241.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 240.50 | 244.13 | 241.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 239.75 | 243.26 | 241.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 239.75 | 243.26 | 241.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 239.13 | 242.43 | 241.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 239.44 | 242.43 | 241.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 225.01 | 237.78 | 239.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 223.62 | 231.35 | 235.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 224.15 | 222.73 | 225.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:00:00 | 224.15 | 222.73 | 225.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 224.60 | 223.43 | 224.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:45:00 | 225.27 | 223.43 | 224.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 222.87 | 223.32 | 224.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 221.74 | 223.49 | 224.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 222.58 | 223.55 | 224.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 222.23 | 223.20 | 223.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 222.34 | 221.97 | 222.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 219.72 | 218.75 | 219.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:45:00 | 219.17 | 218.75 | 219.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 219.90 | 218.98 | 219.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 220.47 | 218.98 | 219.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 220.29 | 219.24 | 219.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 220.76 | 220.21 | 220.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 220.76 | 220.21 | 220.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 221.85 | 220.62 | 220.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 220.00 | 220.78 | 220.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 220.00 | 220.78 | 220.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 220.00 | 220.78 | 220.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 220.00 | 220.78 | 220.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 219.79 | 220.58 | 220.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 219.35 | 220.58 | 220.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 218.72 | 220.21 | 220.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 218.49 | 219.87 | 220.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 14:15:00 | 218.84 | 218.76 | 219.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 218.84 | 218.76 | 219.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 218.84 | 218.76 | 219.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 217.43 | 218.25 | 218.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:15:00 | 217.45 | 218.25 | 218.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 217.30 | 217.93 | 218.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 206.56 | 209.83 | 212.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 206.58 | 209.83 | 212.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 206.44 | 209.83 | 212.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 210.44 | 209.95 | 212.16 | SL hit (close>ema200) qty=0.50 sl=209.95 alert=retest2 |

### Cycle 164 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 212.70 | 211.54 | 211.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 214.50 | 212.13 | 211.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 214.99 | 215.12 | 214.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 214.99 | 215.12 | 214.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 216.09 | 215.32 | 214.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 214.66 | 215.32 | 214.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 215.01 | 215.89 | 215.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 214.91 | 215.89 | 215.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 214.11 | 215.53 | 214.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:15:00 | 213.78 | 215.53 | 214.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 212.07 | 214.84 | 214.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 212.28 | 214.84 | 214.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 210.85 | 214.04 | 214.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 210.08 | 212.12 | 213.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 212.15 | 211.92 | 212.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 212.15 | 211.92 | 212.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 212.15 | 211.92 | 212.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 212.15 | 211.92 | 212.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 212.90 | 212.19 | 212.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 215.15 | 212.19 | 212.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 216.25 | 213.00 | 213.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 216.25 | 213.00 | 213.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 216.39 | 213.68 | 213.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 218.00 | 214.54 | 213.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 215.88 | 216.01 | 215.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 216.11 | 216.01 | 215.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 216.18 | 216.04 | 215.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 215.39 | 216.04 | 215.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 215.24 | 215.88 | 215.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 215.24 | 215.88 | 215.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 215.38 | 215.78 | 215.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 215.22 | 215.78 | 215.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 214.65 | 215.56 | 215.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 214.65 | 215.56 | 215.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 214.60 | 215.36 | 215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:15:00 | 214.95 | 215.36 | 215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 216.95 | 215.62 | 215.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 217.51 | 215.99 | 215.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 10:15:00 | 239.26 | 235.37 | 233.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 236.00 | 237.72 | 237.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 234.15 | 235.91 | 236.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 233.92 | 232.46 | 233.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 10:15:00 | 233.92 | 232.46 | 233.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 233.92 | 232.46 | 233.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 233.92 | 232.46 | 233.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 232.27 | 232.42 | 233.59 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 236.70 | 234.08 | 234.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 15:15:00 | 238.85 | 237.01 | 235.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 243.93 | 244.61 | 243.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 243.93 | 244.61 | 243.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 243.93 | 244.61 | 243.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 243.85 | 244.61 | 243.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 243.38 | 244.36 | 243.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 243.23 | 244.36 | 243.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 243.66 | 244.22 | 243.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 245.00 | 244.38 | 243.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:30:00 | 244.51 | 244.48 | 243.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 240.70 | 243.66 | 243.46 | SL hit (close<static) qty=1.00 sl=243.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 240.44 | 243.01 | 243.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 238.68 | 241.76 | 242.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 239.45 | 239.42 | 240.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 240.67 | 239.42 | 240.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 240.64 | 239.67 | 240.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 240.64 | 239.67 | 240.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 240.39 | 239.81 | 240.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 241.00 | 239.81 | 240.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 240.80 | 240.01 | 240.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 240.80 | 240.01 | 240.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 241.75 | 240.36 | 240.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 241.75 | 240.36 | 240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 240.65 | 240.42 | 240.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:15:00 | 240.42 | 240.42 | 240.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 239.30 | 240.19 | 240.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 236.30 | 235.66 | 235.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 236.30 | 235.66 | 235.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 236.75 | 235.88 | 235.71 | Break + close above crossover candle high |

### Cycle 171 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 232.53 | 235.43 | 235.57 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 237.65 | 234.92 | 234.76 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 234.85 | 235.18 | 235.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 231.81 | 234.51 | 234.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 235.01 | 233.67 | 234.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 235.01 | 233.67 | 234.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 235.01 | 233.67 | 234.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 235.85 | 233.67 | 234.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 233.69 | 233.68 | 234.27 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 235.60 | 234.66 | 234.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 237.50 | 235.58 | 235.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 12:15:00 | 263.10 | 263.53 | 258.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 263.10 | 263.53 | 258.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 261.65 | 265.22 | 263.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 261.65 | 265.22 | 263.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 260.45 | 264.26 | 263.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 260.20 | 264.26 | 263.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 259.95 | 262.55 | 262.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 257.10 | 261.16 | 261.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 261.25 | 261.17 | 261.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 260.50 | 261.17 | 261.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 262.65 | 261.47 | 261.95 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 264.70 | 262.50 | 262.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 267.55 | 263.93 | 263.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 285.90 | 286.43 | 282.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 281.95 | 284.99 | 282.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 281.95 | 284.99 | 282.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 281.95 | 284.99 | 282.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 282.00 | 284.39 | 282.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 282.00 | 284.39 | 282.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 283.05 | 283.61 | 282.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:00:00 | 285.10 | 283.17 | 282.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 285.80 | 283.49 | 282.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:30:00 | 284.65 | 284.76 | 283.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 283.60 | 286.56 | 286.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 283.60 | 286.56 | 286.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 279.90 | 282.73 | 284.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 282.05 | 280.71 | 282.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 282.05 | 280.71 | 282.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 282.05 | 280.71 | 282.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 282.05 | 280.71 | 282.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 282.50 | 281.07 | 282.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 284.40 | 281.07 | 282.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 282.15 | 281.28 | 282.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 281.60 | 281.35 | 282.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 282.95 | 281.78 | 282.47 | SL hit (close>static) qty=1.00 sl=282.85 alert=retest2 |

### Cycle 178 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 286.45 | 283.42 | 283.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 289.00 | 284.53 | 283.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 15:15:00 | 290.20 | 290.62 | 289.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 293.55 | 290.62 | 289.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 291.50 | 291.64 | 290.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 290.65 | 291.64 | 290.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 290.65 | 291.44 | 290.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 290.80 | 291.44 | 290.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 290.85 | 291.33 | 290.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 288.25 | 290.71 | 290.40 | SL hit (close<ema400) qty=1.00 sl=290.40 alert=retest1 |

### Cycle 179 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 288.25 | 289.91 | 290.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 285.40 | 288.68 | 289.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 277.75 | 276.46 | 278.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 277.75 | 276.46 | 278.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 277.75 | 276.46 | 278.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 274.95 | 276.44 | 278.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 276.95 | 276.02 | 275.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 276.95 | 276.02 | 275.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 277.90 | 276.61 | 276.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 276.30 | 276.55 | 276.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 11:15:00 | 276.30 | 276.55 | 276.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 276.30 | 276.55 | 276.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 276.30 | 276.55 | 276.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 276.20 | 276.48 | 276.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:30:00 | 275.55 | 276.48 | 276.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 275.20 | 276.22 | 276.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 275.20 | 276.22 | 276.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 275.35 | 276.05 | 276.05 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 276.90 | 276.14 | 276.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 282.35 | 277.76 | 276.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 282.70 | 283.21 | 280.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:00:00 | 282.70 | 283.21 | 280.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 279.05 | 282.09 | 281.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 279.05 | 282.09 | 281.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 280.20 | 281.71 | 281.29 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 279.15 | 280.77 | 280.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 278.65 | 280.35 | 280.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 278.50 | 278.14 | 279.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 278.50 | 278.14 | 279.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 274.50 | 274.96 | 276.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 273.90 | 274.69 | 276.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 273.95 | 274.38 | 275.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 278.25 | 276.23 | 276.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 278.25 | 276.23 | 276.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 280.00 | 276.99 | 276.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 281.60 | 281.81 | 280.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 284.00 | 281.81 | 280.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 281.10 | 282.43 | 281.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 281.10 | 282.43 | 281.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 279.60 | 281.86 | 280.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 279.60 | 281.86 | 280.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 278.10 | 281.11 | 280.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 278.10 | 281.11 | 280.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 282.45 | 282.88 | 281.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 282.65 | 282.88 | 281.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 280.55 | 282.42 | 281.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 283.40 | 282.78 | 282.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 279.60 | 281.92 | 281.75 | SL hit (close<static) qty=1.00 sl=280.20 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 279.95 | 281.52 | 281.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 279.00 | 280.54 | 281.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 279.80 | 279.72 | 280.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 279.80 | 279.72 | 280.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 279.80 | 279.72 | 280.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 280.50 | 279.72 | 280.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 283.20 | 280.43 | 280.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 283.20 | 280.43 | 280.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 283.20 | 280.98 | 280.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 284.15 | 281.95 | 281.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 296.80 | 297.73 | 294.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 296.80 | 297.73 | 294.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 296.15 | 299.03 | 297.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 296.15 | 299.03 | 297.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 296.95 | 298.61 | 297.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 295.50 | 298.61 | 297.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 296.45 | 298.18 | 297.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 298.40 | 298.18 | 297.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 294.75 | 300.33 | 299.57 | SL hit (close<static) qty=1.00 sl=296.40 alert=retest2 |

### Cycle 187 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 295.05 | 298.42 | 298.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 272.05 | 292.46 | 295.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 267.85 | 266.27 | 270.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:45:00 | 267.85 | 266.27 | 270.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 271.55 | 267.32 | 270.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 271.95 | 267.32 | 270.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 269.60 | 267.78 | 270.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 271.85 | 267.78 | 270.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 268.10 | 268.05 | 269.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 268.00 | 268.05 | 269.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 268.30 | 266.95 | 268.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 268.30 | 266.95 | 268.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 267.65 | 267.09 | 268.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:30:00 | 265.05 | 265.19 | 267.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 251.80 | 261.35 | 264.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 255.05 | 253.44 | 257.29 | SL hit (close>ema200) qty=0.50 sl=253.44 alert=retest2 |

### Cycle 188 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 254.80 | 249.84 | 249.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 259.85 | 252.54 | 250.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 255.10 | 261.26 | 259.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 255.10 | 261.26 | 259.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 255.10 | 261.26 | 259.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 255.10 | 261.26 | 259.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 256.80 | 260.37 | 258.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 255.45 | 260.37 | 258.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 252.25 | 257.67 | 257.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 250.70 | 255.02 | 256.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 254.45 | 254.37 | 255.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 254.50 | 254.37 | 255.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 258.85 | 255.27 | 256.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 258.85 | 255.27 | 256.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 258.40 | 255.89 | 256.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 264.10 | 255.89 | 256.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 261.75 | 257.07 | 256.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 273.00 | 269.23 | 265.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 269.50 | 270.13 | 266.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 269.50 | 270.13 | 266.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 265.05 | 268.46 | 267.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 265.05 | 268.46 | 267.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 265.40 | 267.85 | 267.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 264.50 | 267.85 | 267.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 264.50 | 266.85 | 266.91 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 272.30 | 267.70 | 267.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 274.05 | 268.97 | 267.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 261.15 | 271.52 | 271.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 261.15 | 271.52 | 271.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 261.15 | 271.52 | 271.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 260.30 | 271.52 | 271.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 260.35 | 269.29 | 270.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 259.25 | 265.90 | 268.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 09:15:00 | 263.55 | 263.25 | 266.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 09:30:00 | 264.35 | 263.25 | 266.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 260.50 | 259.84 | 262.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:45:00 | 260.00 | 259.84 | 262.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 260.10 | 258.37 | 259.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 260.10 | 258.37 | 259.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 260.60 | 258.82 | 260.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:45:00 | 261.25 | 258.82 | 260.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 262.40 | 260.10 | 260.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 261.35 | 260.10 | 260.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 262.90 | 261.04 | 260.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 264.00 | 261.63 | 261.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 262.60 | 262.65 | 261.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:15:00 | 264.30 | 262.65 | 261.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 263.10 | 263.50 | 262.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:15:00 | 261.00 | 263.50 | 262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 261.00 | 263.00 | 262.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 261.00 | 263.00 | 262.59 | SL hit (close<ema400) qty=1.00 sl=262.59 alert=retest1 |

### Cycle 195 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 256.25 | 261.65 | 262.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 253.90 | 257.92 | 259.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 257.80 | 256.46 | 258.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 257.80 | 256.46 | 258.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 257.80 | 256.46 | 258.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 257.80 | 256.46 | 258.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 256.40 | 256.44 | 258.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 258.60 | 256.44 | 258.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 257.90 | 256.74 | 258.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 258.25 | 256.74 | 258.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 257.15 | 256.82 | 258.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 257.15 | 256.82 | 258.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 258.90 | 257.38 | 258.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 260.50 | 257.38 | 258.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 256.75 | 257.25 | 257.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 255.80 | 257.02 | 257.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 255.70 | 256.96 | 257.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 255.30 | 256.96 | 257.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:45:00 | 255.85 | 256.96 | 257.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 260.20 | 257.61 | 257.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-24 13:15:00 | 260.20 | 257.61 | 257.66 | SL hit (close>static) qty=1.00 sl=259.10 alert=retest2 |

### Cycle 196 — BUY (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 14:15:00 | 261.95 | 258.48 | 258.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 264.10 | 260.18 | 258.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 260.95 | 261.02 | 259.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 260.95 | 261.02 | 259.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 260.80 | 261.42 | 260.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 259.90 | 261.42 | 260.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 260.65 | 261.26 | 260.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:15:00 | 261.25 | 261.26 | 260.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 262.45 | 261.50 | 260.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 263.70 | 262.16 | 261.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 259.40 | 262.94 | 262.85 | SL hit (close<static) qty=1.00 sl=260.40 alert=retest2 |

### Cycle 197 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 260.90 | 262.53 | 262.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 253.95 | 260.69 | 261.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 252.40 | 252.01 | 255.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 252.40 | 252.01 | 255.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 253.60 | 252.98 | 255.14 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 262.85 | 257.30 | 256.66 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 249.70 | 256.00 | 256.76 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 258.30 | 256.22 | 256.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 259.95 | 257.35 | 256.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 257.60 | 257.95 | 257.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 257.60 | 257.95 | 257.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 256.75 | 257.71 | 257.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 256.75 | 257.71 | 257.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 255.20 | 257.21 | 256.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 255.20 | 257.21 | 256.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 255.80 | 256.93 | 256.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 253.75 | 256.93 | 256.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 255.10 | 256.56 | 256.66 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 263.70 | 257.99 | 257.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 269.55 | 260.30 | 258.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 262.65 | 264.29 | 261.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 262.65 | 264.29 | 261.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 262.65 | 264.29 | 261.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 262.65 | 264.29 | 261.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 261.80 | 263.79 | 261.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 262.00 | 263.79 | 261.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 262.80 | 263.59 | 261.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:15:00 | 261.20 | 263.59 | 261.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 258.20 | 262.51 | 261.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 258.20 | 262.51 | 261.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 258.30 | 261.67 | 261.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 258.30 | 261.67 | 261.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 258.25 | 260.56 | 260.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 254.80 | 259.41 | 260.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 255.85 | 253.67 | 255.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 13:15:00 | 255.85 | 253.67 | 255.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 255.85 | 253.67 | 255.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:45:00 | 255.40 | 253.67 | 255.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 256.70 | 254.28 | 255.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 257.05 | 254.28 | 255.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 255.40 | 254.50 | 255.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 258.80 | 254.50 | 255.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 258.70 | 255.34 | 255.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 260.40 | 255.34 | 255.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 259.00 | 256.07 | 256.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 259.75 | 257.26 | 256.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 253.60 | 257.58 | 257.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 253.60 | 257.58 | 257.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 253.60 | 257.58 | 257.12 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 254.45 | 256.45 | 256.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 253.15 | 255.79 | 256.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 263.25 | 255.79 | 255.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 263.25 | 255.79 | 255.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 263.25 | 255.79 | 255.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 263.25 | 255.79 | 255.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 264.80 | 257.59 | 256.76 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 251.45 | 257.80 | 257.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 250.00 | 255.11 | 256.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 254.80 | 254.06 | 255.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 254.80 | 254.06 | 255.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 254.80 | 254.06 | 255.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 253.75 | 254.06 | 255.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 258.40 | 256.34 | 256.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 258.40 | 256.34 | 256.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 266.10 | 258.52 | 257.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 262.20 | 262.30 | 259.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:45:00 | 262.40 | 262.30 | 259.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 257.40 | 261.30 | 259.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 257.40 | 261.30 | 259.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 257.10 | 260.46 | 259.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 256.50 | 260.46 | 259.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 256.40 | 258.96 | 259.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 254.70 | 257.69 | 258.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 252.68 | 251.04 | 253.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 252.68 | 251.04 | 253.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 252.68 | 251.04 | 253.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 254.42 | 251.04 | 253.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 252.86 | 251.40 | 253.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 253.07 | 251.40 | 253.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 255.50 | 252.22 | 253.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 255.50 | 252.22 | 253.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 256.71 | 253.12 | 254.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 256.80 | 253.12 | 254.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 251.71 | 252.95 | 253.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 243.69 | 252.64 | 253.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 253.91 | 248.94 | 248.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 253.91 | 248.94 | 248.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 262.81 | 252.44 | 250.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 15:15:00 | 287.00 | 287.40 | 281.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:15:00 | 294.13 | 287.40 | 281.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 14:15:00 | 308.84 | 297.00 | 291.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-20 09:15:00 | 323.54 | 315.04 | 305.80 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-07 09:15:00 | 84.15 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2023-06-07 09:45:00 | 84.25 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2023-06-08 09:15:00 | 84.75 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-06-09 10:45:00 | 84.35 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-06-09 14:00:00 | 85.45 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2023-06-13 14:30:00 | 85.25 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-06-13 15:00:00 | 85.30 | 2023-06-14 09:15:00 | 84.25 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-06-21 12:15:00 | 87.80 | 2023-06-21 15:15:00 | 86.85 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-06-21 12:45:00 | 87.80 | 2023-06-21 15:15:00 | 86.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-06-27 12:45:00 | 84.40 | 2023-06-28 12:15:00 | 85.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-06-27 14:00:00 | 84.50 | 2023-06-28 12:15:00 | 85.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-06-28 12:00:00 | 84.50 | 2023-06-28 12:15:00 | 85.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-07-07 13:45:00 | 91.15 | 2023-07-10 12:15:00 | 90.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-07-13 09:15:00 | 95.70 | 2023-07-13 11:15:00 | 93.20 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2023-07-24 13:00:00 | 96.75 | 2023-08-01 09:15:00 | 106.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-24 13:45:00 | 97.05 | 2023-08-01 09:15:00 | 106.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-24 14:45:00 | 97.20 | 2023-08-01 10:15:00 | 106.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-04 10:15:00 | 100.65 | 2023-08-04 11:15:00 | 102.70 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-08-04 13:15:00 | 100.35 | 2023-08-07 09:15:00 | 95.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-04 13:15:00 | 100.35 | 2023-08-08 09:15:00 | 99.90 | STOP_HIT | 0.50 | 0.45% |
| BUY | retest2 | 2023-08-11 09:15:00 | 100.35 | 2023-08-16 10:15:00 | 99.70 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-08-18 09:30:00 | 99.35 | 2023-08-21 14:15:00 | 100.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2023-09-07 09:15:00 | 136.45 | 2023-09-12 09:15:00 | 133.35 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2023-09-14 13:45:00 | 129.85 | 2023-09-20 13:15:00 | 123.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 15:00:00 | 129.90 | 2023-09-20 13:15:00 | 123.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 11:15:00 | 130.00 | 2023-09-20 13:15:00 | 123.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 13:45:00 | 129.85 | 2023-09-21 10:15:00 | 125.65 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2023-09-14 15:00:00 | 129.90 | 2023-09-21 10:15:00 | 125.65 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2023-09-15 11:15:00 | 130.00 | 2023-09-21 10:15:00 | 125.65 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2023-09-27 09:15:00 | 123.85 | 2023-09-27 13:15:00 | 126.20 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2023-09-27 09:45:00 | 124.10 | 2023-09-27 13:15:00 | 126.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-09-29 09:15:00 | 129.65 | 2023-10-04 12:15:00 | 126.30 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-10-31 13:30:00 | 120.70 | 2023-11-08 12:15:00 | 127.20 | STOP_HIT | 1.00 | 5.39% |
| SELL | retest2 | 2023-11-09 14:45:00 | 125.55 | 2023-11-10 12:15:00 | 129.10 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2023-11-10 09:30:00 | 124.95 | 2023-11-10 12:15:00 | 129.10 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2023-12-12 11:15:00 | 178.90 | 2023-12-12 15:15:00 | 177.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-12-12 13:15:00 | 178.70 | 2023-12-12 15:15:00 | 177.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-12-18 10:15:00 | 183.00 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2023-12-18 11:00:00 | 182.95 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2023-12-18 12:00:00 | 183.00 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2023-12-18 14:45:00 | 187.00 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -5.61% |
| BUY | retest2 | 2023-12-19 12:30:00 | 186.10 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2023-12-19 13:45:00 | 186.20 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest2 | 2023-12-19 14:15:00 | 186.00 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -5.11% |
| BUY | retest2 | 2023-12-20 09:15:00 | 188.00 | 2023-12-20 13:15:00 | 176.50 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest2 | 2023-12-28 09:15:00 | 182.70 | 2024-01-01 09:15:00 | 200.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-18 12:00:00 | 208.80 | 2024-01-23 12:15:00 | 209.25 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-01-18 14:00:00 | 209.45 | 2024-01-23 12:15:00 | 209.25 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-01-18 14:30:00 | 209.45 | 2024-01-23 12:15:00 | 209.25 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-02-07 09:15:00 | 236.45 | 2024-02-07 11:15:00 | 230.60 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-02-12 11:45:00 | 216.95 | 2024-02-13 12:15:00 | 206.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-12 15:00:00 | 215.55 | 2024-02-13 14:15:00 | 204.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-12 11:45:00 | 216.95 | 2024-02-14 09:15:00 | 214.80 | STOP_HIT | 0.50 | 0.99% |
| SELL | retest2 | 2024-02-12 15:00:00 | 215.55 | 2024-02-14 09:15:00 | 214.80 | STOP_HIT | 0.50 | 0.35% |
| BUY | retest2 | 2024-02-19 09:15:00 | 228.80 | 2024-02-21 13:15:00 | 223.80 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-02-19 15:15:00 | 226.05 | 2024-02-21 13:15:00 | 223.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-02-21 10:45:00 | 226.50 | 2024-02-21 13:15:00 | 223.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-02-21 12:30:00 | 226.25 | 2024-02-21 13:15:00 | 223.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-03-06 14:15:00 | 254.20 | 2024-03-12 09:15:00 | 252.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-04-04 13:30:00 | 251.05 | 2024-04-15 14:15:00 | 256.15 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2024-04-04 14:30:00 | 251.05 | 2024-04-15 14:15:00 | 256.15 | STOP_HIT | 1.00 | 2.03% |
| BUY | retest2 | 2024-04-05 10:15:00 | 251.65 | 2024-04-15 14:15:00 | 256.15 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2024-05-09 09:15:00 | 282.15 | 2024-05-10 09:15:00 | 268.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 09:15:00 | 282.15 | 2024-05-13 11:15:00 | 277.00 | STOP_HIT | 0.50 | 1.83% |
| BUY | retest2 | 2024-05-17 13:30:00 | 297.30 | 2024-05-23 14:15:00 | 304.80 | STOP_HIT | 1.00 | 2.52% |
| BUY | retest2 | 2024-05-17 14:15:00 | 297.20 | 2024-05-23 14:15:00 | 304.80 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest1 | 2024-06-11 09:15:00 | 294.50 | 2024-06-14 12:15:00 | 309.23 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-11 09:15:00 | 294.50 | 2024-06-19 09:15:00 | 299.50 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-06-24 09:15:00 | 292.35 | 2024-06-27 09:15:00 | 298.30 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-07-01 15:00:00 | 302.45 | 2024-07-02 11:15:00 | 296.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-11 09:15:00 | 331.80 | 2024-07-12 15:15:00 | 325.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-07-23 12:15:00 | 293.50 | 2024-07-23 15:15:00 | 307.75 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2024-08-01 11:45:00 | 311.95 | 2024-08-05 09:15:00 | 296.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 11:45:00 | 311.95 | 2024-08-06 09:15:00 | 299.05 | STOP_HIT | 0.50 | 4.14% |
| BUY | retest2 | 2024-08-09 09:15:00 | 303.35 | 2024-08-12 14:15:00 | 299.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-08-12 10:00:00 | 298.00 | 2024-08-12 14:15:00 | 299.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-08-16 10:15:00 | 290.25 | 2024-08-16 14:15:00 | 296.85 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-09-02 09:30:00 | 289.40 | 2024-09-05 09:15:00 | 274.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:30:00 | 289.40 | 2024-09-09 09:15:00 | 260.46 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-09-17 12:00:00 | 267.00 | 2024-09-18 12:15:00 | 264.85 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-18 10:00:00 | 267.40 | 2024-09-18 12:15:00 | 264.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-09-30 11:30:00 | 278.05 | 2024-09-30 13:15:00 | 279.45 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-10-09 09:15:00 | 265.00 | 2024-10-09 10:15:00 | 267.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-10-09 10:15:00 | 264.35 | 2024-10-09 10:15:00 | 267.20 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-10-09 14:00:00 | 265.10 | 2024-10-10 09:15:00 | 270.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-10-14 09:30:00 | 271.35 | 2024-10-14 12:15:00 | 268.35 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-10-14 10:30:00 | 270.60 | 2024-10-14 12:15:00 | 268.35 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-10-31 13:45:00 | 239.10 | 2024-11-04 10:15:00 | 232.10 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-10-31 14:30:00 | 239.25 | 2024-11-04 10:15:00 | 232.10 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-10-31 15:00:00 | 239.25 | 2024-11-04 10:15:00 | 232.10 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2024-11-01 18:00:00 | 242.13 | 2024-11-04 10:15:00 | 232.10 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2024-11-11 15:15:00 | 238.39 | 2024-11-13 09:15:00 | 226.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:15:00 | 238.39 | 2024-11-14 09:15:00 | 225.90 | STOP_HIT | 0.50 | 5.24% |
| SELL | retest2 | 2024-12-10 09:15:00 | 247.70 | 2024-12-10 09:15:00 | 252.80 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-01-08 09:15:00 | 221.57 | 2025-01-10 09:15:00 | 210.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 221.57 | 2025-01-13 09:15:00 | 199.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 201.97 | 2025-01-28 09:15:00 | 191.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 201.97 | 2025-01-29 09:15:00 | 195.00 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-02-10 12:45:00 | 202.88 | 2025-02-12 09:15:00 | 192.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 13:30:00 | 203.01 | 2025-02-12 09:15:00 | 192.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 202.09 | 2025-02-12 09:15:00 | 191.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 15:00:00 | 203.08 | 2025-02-12 09:15:00 | 192.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 12:45:00 | 202.88 | 2025-02-12 10:15:00 | 197.95 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-02-10 13:30:00 | 203.01 | 2025-02-12 10:15:00 | 197.95 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2025-02-10 14:00:00 | 202.09 | 2025-02-12 10:15:00 | 197.95 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2025-02-10 15:00:00 | 203.08 | 2025-02-12 10:15:00 | 197.95 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-02-11 11:15:00 | 195.50 | 2025-02-13 09:15:00 | 204.44 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-02-12 09:30:00 | 194.19 | 2025-02-13 09:15:00 | 204.44 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-02-18 11:00:00 | 186.48 | 2025-02-19 11:15:00 | 194.60 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-02-18 11:45:00 | 185.74 | 2025-02-19 11:15:00 | 194.60 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-02-18 15:00:00 | 186.42 | 2025-02-19 11:15:00 | 194.60 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-02-21 12:30:00 | 199.44 | 2025-02-24 09:15:00 | 193.37 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-03-10 09:15:00 | 197.79 | 2025-03-10 10:15:00 | 196.17 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-03-10 10:30:00 | 197.22 | 2025-03-10 11:15:00 | 194.70 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-03-12 10:45:00 | 190.80 | 2025-03-13 10:15:00 | 195.95 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-03-26 09:15:00 | 214.49 | 2025-03-26 09:15:00 | 212.75 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-03-26 11:15:00 | 213.78 | 2025-03-26 12:15:00 | 212.81 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-03-27 13:30:00 | 212.78 | 2025-03-27 15:15:00 | 215.99 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-04-15 09:15:00 | 217.15 | 2025-04-25 09:15:00 | 222.71 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest2 | 2025-05-02 11:15:00 | 227.60 | 2025-05-05 15:15:00 | 229.42 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-02 12:00:00 | 226.85 | 2025-05-05 15:15:00 | 229.42 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-05-22 09:30:00 | 248.90 | 2025-05-30 13:15:00 | 262.52 | STOP_HIT | 1.00 | 5.47% |
| BUY | retest2 | 2025-05-23 10:45:00 | 250.01 | 2025-05-30 13:15:00 | 262.52 | STOP_HIT | 1.00 | 5.00% |
| SELL | retest2 | 2025-06-03 11:30:00 | 256.30 | 2025-06-09 09:15:00 | 258.75 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-03 12:00:00 | 255.80 | 2025-06-09 09:15:00 | 258.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-06 10:30:00 | 255.70 | 2025-06-09 09:15:00 | 258.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-06-16 15:15:00 | 254.00 | 2025-06-23 10:15:00 | 253.10 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-06-17 11:30:00 | 254.20 | 2025-06-23 10:15:00 | 253.10 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-06-18 10:15:00 | 253.75 | 2025-06-23 10:15:00 | 253.10 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-06-30 09:15:00 | 266.40 | 2025-07-02 09:15:00 | 262.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-01 10:30:00 | 264.85 | 2025-07-02 09:15:00 | 262.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-01 13:15:00 | 264.45 | 2025-07-02 09:15:00 | 262.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-07 09:15:00 | 258.40 | 2025-07-09 10:15:00 | 263.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-09 09:30:00 | 258.30 | 2025-07-09 10:15:00 | 263.50 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-16 11:00:00 | 254.60 | 2025-07-21 15:15:00 | 254.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-16 15:00:00 | 254.60 | 2025-07-21 15:15:00 | 254.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-17 09:30:00 | 254.00 | 2025-07-21 15:15:00 | 254.35 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-07-25 09:30:00 | 247.45 | 2025-07-29 09:15:00 | 235.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 247.45 | 2025-07-29 12:15:00 | 239.10 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2025-08-13 11:45:00 | 221.74 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-08-14 09:45:00 | 222.58 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-08-14 10:30:00 | 222.23 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-08-18 09:30:00 | 222.34 | 2025-08-20 14:15:00 | 220.76 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-08-25 11:30:00 | 217.43 | 2025-08-29 09:15:00 | 206.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:15:00 | 217.45 | 2025-08-29 09:15:00 | 206.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:00:00 | 217.30 | 2025-08-29 09:15:00 | 206.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 11:30:00 | 217.43 | 2025-08-29 10:15:00 | 210.44 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-08-25 12:15:00 | 217.45 | 2025-08-29 10:15:00 | 210.44 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2025-08-25 14:00:00 | 217.30 | 2025-08-29 10:15:00 | 210.44 | STOP_HIT | 0.50 | 3.16% |
| BUY | retest2 | 2025-09-10 11:00:00 | 217.51 | 2025-09-19 10:15:00 | 239.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-07 13:00:00 | 245.00 | 2025-10-08 10:15:00 | 240.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-07 14:30:00 | 244.51 | 2025-10-08 10:15:00 | 240.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-10 14:15:00 | 240.42 | 2025-10-16 15:15:00 | 236.30 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2025-10-10 15:00:00 | 239.30 | 2025-10-16 15:15:00 | 236.30 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-11-17 10:00:00 | 285.10 | 2025-11-21 09:15:00 | 283.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-11-17 12:15:00 | 285.80 | 2025-11-21 09:15:00 | 283.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-18 10:30:00 | 284.65 | 2025-11-21 09:15:00 | 283.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-11-25 13:00:00 | 281.60 | 2025-11-25 14:15:00 | 282.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-12-01 09:15:00 | 293.55 | 2025-12-02 10:15:00 | 288.25 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-12-08 09:45:00 | 274.95 | 2025-12-09 15:15:00 | 276.95 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-19 10:45:00 | 273.90 | 2025-12-22 11:15:00 | 278.25 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-12-19 13:00:00 | 273.95 | 2025-12-22 11:15:00 | 278.25 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-12-29 09:30:00 | 283.40 | 2025-12-29 11:15:00 | 279.60 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-01-07 09:15:00 | 298.40 | 2026-01-08 10:15:00 | 294.75 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-19 11:30:00 | 265.05 | 2026-01-20 09:15:00 | 251.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:30:00 | 265.05 | 2026-01-21 11:15:00 | 255.05 | STOP_HIT | 0.50 | 3.77% |
| BUY | retest1 | 2026-02-18 09:15:00 | 264.30 | 2026-02-18 15:15:00 | 261.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-02-23 11:45:00 | 255.80 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-24 09:30:00 | 255.70 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-02-24 10:00:00 | 255.30 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-02-24 12:45:00 | 255.85 | 2026-02-24 13:15:00 | 260.20 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-02-26 14:30:00 | 263.70 | 2026-03-02 11:15:00 | 259.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-03-24 10:15:00 | 253.75 | 2026-03-24 14:15:00 | 258.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-04-02 09:15:00 | 243.69 | 2026-04-07 14:15:00 | 253.91 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2026-04-15 09:15:00 | 294.13 | 2026-04-16 14:15:00 | 308.84 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-15 09:15:00 | 294.13 | 2026-04-20 09:15:00 | 323.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-30 09:30:00 | 346.42 | 2026-05-04 11:15:00 | 381.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 11:00:00 | 347.65 | 2026-05-04 11:15:00 | 382.42 | TARGET_HIT | 1.00 | 10.00% |
