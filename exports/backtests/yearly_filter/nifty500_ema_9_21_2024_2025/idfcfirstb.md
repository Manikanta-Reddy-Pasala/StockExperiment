# IDFC First Bank Ltd. (IDFCFIRSTB)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 71.19
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 104 |
| ALERT2 | 103 |
| ALERT2_SKIP | 50 |
| ALERT3 | 263 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 116 |
| PARTIAL | 9 |
| TARGET_HIT | 0 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 108
- **Target hits / Stop hits / Partials:** 0 / 127 / 9
- **Avg / median % per leg:** -0.28% / -0.80%
- **Sum % (uncompounded):** -38.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 6 | 10.9% | 0 | 55 | 0 | -0.56% | -30.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.42% | -1.7% |
| BUY @ 3rd Alert (retest2) | 51 | 6 | 11.8% | 0 | 51 | 0 | -0.57% | -29.2% |
| SELL (all) | 81 | 22 | 27.2% | 0 | 72 | 9 | -0.09% | -7.4% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.43% | -10.0% |
| SELL @ 3rd Alert (retest2) | 74 | 22 | 29.7% | 0 | 65 | 9 | 0.04% | 2.6% |
| retest1 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.06% | -11.7% |
| retest2 (combined) | 125 | 28 | 22.4% | 0 | 116 | 9 | -0.21% | -26.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 77.45 | 76.89 | 76.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 77.65 | 77.21 | 77.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 77.05 | 77.25 | 77.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 77.05 | 77.25 | 77.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 77.05 | 77.25 | 77.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 77.05 | 77.25 | 77.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 77.00 | 77.20 | 77.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:15:00 | 76.85 | 77.20 | 77.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 77.10 | 77.18 | 77.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 77.05 | 77.18 | 77.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 77.00 | 77.14 | 77.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 77.00 | 77.14 | 77.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 76.80 | 77.08 | 77.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:00:00 | 76.80 | 77.08 | 77.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 77.00 | 77.06 | 77.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:15:00 | 76.85 | 77.06 | 77.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 15:15:00 | 76.85 | 77.02 | 77.02 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 77.30 | 77.07 | 77.05 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 76.90 | 77.01 | 77.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 76.75 | 76.96 | 77.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 77.05 | 76.98 | 77.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 77.05 | 76.98 | 77.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 77.05 | 76.98 | 77.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 77.10 | 76.98 | 77.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 77.05 | 76.99 | 77.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 77.30 | 76.99 | 77.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 77.20 | 77.03 | 77.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 77.60 | 77.39 | 77.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 15:15:00 | 77.40 | 77.43 | 77.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 15:15:00 | 77.40 | 77.43 | 77.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 77.40 | 77.43 | 77.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 77.40 | 77.43 | 77.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 77.15 | 77.38 | 77.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 77.25 | 77.38 | 77.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 77.00 | 77.30 | 77.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 77.00 | 77.30 | 77.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 77.00 | 77.22 | 77.24 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 77.75 | 77.29 | 77.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 78.05 | 77.52 | 77.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 77.75 | 77.88 | 77.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 77.75 | 77.88 | 77.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 77.75 | 77.88 | 77.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:30:00 | 77.85 | 77.88 | 77.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 77.65 | 77.84 | 77.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 77.85 | 77.84 | 77.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 77.90 | 77.85 | 77.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 12:00:00 | 78.20 | 77.93 | 77.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:45:00 | 78.20 | 78.41 | 78.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 14:30:00 | 78.20 | 78.28 | 78.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:30:00 | 78.35 | 78.17 | 78.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 10:15:00 | 77.75 | 78.09 | 78.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 77.75 | 78.09 | 78.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 77.15 | 77.71 | 77.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 10:15:00 | 77.55 | 77.53 | 77.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 11:00:00 | 77.55 | 77.53 | 77.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 77.20 | 77.46 | 77.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:30:00 | 77.10 | 77.52 | 77.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 78.10 | 77.64 | 77.72 | SL hit (close>static) qty=1.00 sl=77.75 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 78.25 | 77.47 | 77.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 78.60 | 77.69 | 77.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 76.30 | 77.64 | 77.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 76.30 | 77.64 | 77.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 76.30 | 77.64 | 77.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 75.45 | 77.64 | 77.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 72.05 | 76.52 | 77.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 71.80 | 75.58 | 76.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 74.30 | 73.81 | 75.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 74.30 | 73.81 | 75.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 75.20 | 74.09 | 75.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 75.20 | 74.09 | 75.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 76.20 | 74.51 | 75.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 76.20 | 74.51 | 75.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 77.20 | 75.05 | 75.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 77.20 | 75.05 | 75.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 77.10 | 75.82 | 75.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 77.70 | 76.45 | 76.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 77.35 | 77.62 | 77.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 10:15:00 | 77.35 | 77.62 | 77.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 77.35 | 77.62 | 77.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 77.34 | 77.62 | 77.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 77.64 | 77.60 | 77.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 77.45 | 77.60 | 77.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 77.25 | 77.53 | 77.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:00:00 | 77.74 | 77.57 | 77.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 77.66 | 77.62 | 77.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:45:00 | 77.74 | 77.65 | 77.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 77.74 | 77.62 | 77.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 78.07 | 77.71 | 77.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:15:00 | 78.32 | 77.71 | 77.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 15:15:00 | 77.47 | 77.62 | 77.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 77.47 | 77.62 | 77.63 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 77.93 | 77.68 | 77.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 10:15:00 | 78.20 | 77.79 | 77.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 12:15:00 | 77.78 | 77.80 | 77.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 12:30:00 | 77.82 | 77.80 | 77.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 77.66 | 77.77 | 77.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:00:00 | 77.66 | 77.77 | 77.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 78.04 | 77.83 | 77.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:15:00 | 78.12 | 77.83 | 77.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 82.54 | 82.86 | 82.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 82.54 | 82.86 | 82.87 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 83.13 | 82.92 | 82.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 83.19 | 82.97 | 82.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 82.93 | 82.96 | 82.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 82.93 | 82.96 | 82.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 82.93 | 82.96 | 82.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 82.93 | 82.96 | 82.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 82.80 | 82.93 | 82.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:00:00 | 82.80 | 82.93 | 82.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 82.85 | 82.91 | 82.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 82.85 | 82.91 | 82.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 82.90 | 82.91 | 82.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 82.42 | 82.91 | 82.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 83.30 | 82.99 | 82.94 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 82.30 | 82.92 | 82.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 09:15:00 | 81.67 | 82.26 | 82.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 10:15:00 | 79.82 | 79.54 | 80.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 11:00:00 | 79.82 | 79.54 | 80.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 80.54 | 79.74 | 80.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:45:00 | 80.51 | 79.74 | 80.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 80.50 | 79.90 | 80.47 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 81.16 | 80.60 | 80.59 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 80.00 | 80.78 | 80.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 79.81 | 80.36 | 80.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 78.76 | 78.55 | 79.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:00:00 | 78.76 | 78.55 | 79.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 78.99 | 78.50 | 78.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 79.16 | 78.50 | 78.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 78.99 | 78.60 | 78.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:45:00 | 79.11 | 78.60 | 78.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 78.62 | 78.60 | 78.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 78.40 | 78.59 | 78.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 13:30:00 | 78.50 | 78.56 | 78.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:00:00 | 78.42 | 78.56 | 78.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:15:00 | 74.48 | 74.88 | 75.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:15:00 | 74.58 | 74.88 | 75.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:15:00 | 74.50 | 74.88 | 75.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 74.85 | 74.67 | 75.03 | SL hit (close>ema200) qty=0.50 sl=74.67 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 75.92 | 75.12 | 75.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 09:15:00 | 76.10 | 75.71 | 75.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 13:15:00 | 75.66 | 75.77 | 75.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 14:00:00 | 75.66 | 75.77 | 75.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 75.97 | 75.81 | 75.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 15:15:00 | 76.13 | 75.81 | 75.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:00:00 | 76.07 | 75.95 | 75.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 75.46 | 75.78 | 75.69 | SL hit (close<static) qty=1.00 sl=75.57 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 74.44 | 75.41 | 75.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 72.66 | 74.33 | 74.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 72.80 | 72.69 | 73.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 72.32 | 72.63 | 73.28 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:00:00 | 72.17 | 72.63 | 73.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 72.02 | 72.26 | 72.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 73.20 | 72.49 | 72.60 | SL hit (close>ema400) qty=1.00 sl=72.60 alert=retest1 |

### Cycle 21 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 73.07 | 72.72 | 72.69 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 72.01 | 72.65 | 72.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 71.79 | 72.25 | 72.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 10:15:00 | 72.25 | 72.19 | 72.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:45:00 | 72.19 | 72.19 | 72.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 71.22 | 71.05 | 71.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 70.99 | 71.05 | 71.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 71.75 | 71.56 | 71.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 71.75 | 71.56 | 71.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 72.65 | 72.01 | 71.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 12:15:00 | 74.85 | 75.02 | 74.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 13:00:00 | 74.85 | 75.02 | 74.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 74.42 | 74.85 | 74.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 74.42 | 74.85 | 74.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 74.45 | 74.77 | 74.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 74.65 | 74.77 | 74.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 74.48 | 74.72 | 74.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 12:15:00 | 74.11 | 74.55 | 74.43 | SL hit (close<static) qty=1.00 sl=74.14 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 74.22 | 74.36 | 74.37 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 74.61 | 74.41 | 74.39 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 73.88 | 74.38 | 74.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 73.49 | 74.14 | 74.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 74.25 | 73.74 | 73.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 74.25 | 73.74 | 73.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 74.25 | 73.74 | 73.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 74.11 | 73.74 | 73.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 74.53 | 73.89 | 74.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 74.67 | 73.89 | 74.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 75.32 | 74.28 | 74.18 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 74.50 | 74.74 | 74.76 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 74.81 | 74.77 | 74.76 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 74.13 | 74.81 | 74.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 73.60 | 74.20 | 74.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 73.81 | 73.79 | 74.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 14:00:00 | 73.81 | 73.79 | 74.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 72.88 | 73.61 | 74.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:30:00 | 73.38 | 73.61 | 74.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 72.45 | 72.18 | 72.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 72.45 | 72.18 | 72.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 72.80 | 72.30 | 72.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 72.80 | 72.30 | 72.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 72.84 | 72.41 | 72.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 73.14 | 72.41 | 72.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 73.39 | 72.73 | 72.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 11:15:00 | 73.77 | 72.94 | 72.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 15:15:00 | 73.60 | 73.65 | 73.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:15:00 | 73.43 | 73.65 | 73.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 73.17 | 73.55 | 73.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 73.17 | 73.55 | 73.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 73.40 | 73.52 | 73.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:45:00 | 73.09 | 73.52 | 73.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 73.53 | 73.52 | 73.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 73.63 | 73.43 | 73.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 72.53 | 73.25 | 73.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 72.53 | 73.25 | 73.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 72.10 | 72.66 | 72.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 73.88 | 72.84 | 72.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 73.88 | 72.84 | 72.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 73.88 | 72.84 | 72.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 73.88 | 72.84 | 72.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 74.00 | 73.07 | 73.03 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 73.05 | 73.53 | 73.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 72.68 | 73.36 | 73.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 73.07 | 73.06 | 73.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 11:15:00 | 73.08 | 73.06 | 73.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 73.22 | 73.10 | 73.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 73.22 | 73.10 | 73.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 73.15 | 73.11 | 73.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:45:00 | 72.90 | 73.07 | 73.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 14:15:00 | 73.54 | 73.16 | 73.23 | SL hit (close>static) qty=1.00 sl=73.25 alert=retest2 |

### Cycle 35 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 74.42 | 73.41 | 73.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 09:15:00 | 74.92 | 74.31 | 73.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 12:15:00 | 74.73 | 74.79 | 74.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:00:00 | 74.73 | 74.79 | 74.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 74.49 | 74.67 | 74.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:45:00 | 74.60 | 74.67 | 74.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 74.30 | 74.60 | 74.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 74.81 | 74.62 | 74.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 10:45:00 | 74.66 | 74.62 | 74.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 13:15:00 | 74.06 | 74.44 | 74.35 | SL hit (close<static) qty=1.00 sl=74.09 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 73.48 | 74.25 | 74.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 72.64 | 73.79 | 74.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 72.73 | 72.52 | 73.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:45:00 | 72.78 | 72.52 | 73.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 71.75 | 72.08 | 72.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 70.98 | 71.85 | 72.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:30:00 | 70.90 | 71.76 | 72.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 12:30:00 | 71.32 | 71.70 | 72.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 13:30:00 | 71.27 | 71.66 | 72.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 72.13 | 71.75 | 72.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 15:00:00 | 72.13 | 71.75 | 72.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 72.40 | 71.88 | 72.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 72.82 | 72.07 | 72.26 | SL hit (close>static) qty=1.00 sl=72.81 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 73.02 | 72.41 | 72.39 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 15:15:00 | 72.25 | 72.54 | 72.57 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 73.21 | 72.68 | 72.63 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 72.05 | 72.78 | 72.79 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 72.90 | 72.72 | 72.71 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 10:15:00 | 72.66 | 72.71 | 72.71 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 72.83 | 72.73 | 72.72 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 72.63 | 72.71 | 72.71 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 72.75 | 72.72 | 72.72 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 72.62 | 72.70 | 72.71 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 72.79 | 72.72 | 72.71 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 72.65 | 72.70 | 72.71 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 13:15:00 | 72.97 | 72.73 | 72.72 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 72.15 | 72.62 | 72.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 71.98 | 72.35 | 72.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 71.89 | 71.83 | 72.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:45:00 | 71.90 | 71.83 | 72.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 71.70 | 71.73 | 71.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:00:00 | 71.70 | 71.73 | 71.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 71.89 | 71.76 | 71.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:15:00 | 71.89 | 71.76 | 71.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 71.31 | 71.67 | 71.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:15:00 | 71.30 | 71.67 | 71.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 67.73 | 68.80 | 69.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 67.95 | 67.32 | 68.45 | SL hit (close>ema200) qty=0.50 sl=67.32 alert=retest2 |

### Cycle 51 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 67.24 | 66.59 | 66.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 67.52 | 66.85 | 66.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 67.99 | 68.86 | 68.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 11:15:00 | 67.99 | 68.86 | 68.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 67.99 | 68.86 | 68.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 67.99 | 68.86 | 68.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 67.00 | 68.49 | 68.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:30:00 | 67.14 | 68.49 | 68.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 66.73 | 68.13 | 67.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:30:00 | 66.94 | 68.13 | 67.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 14:15:00 | 65.84 | 67.68 | 67.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 11:15:00 | 65.63 | 66.61 | 67.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 66.26 | 65.96 | 66.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 66.26 | 65.96 | 66.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 66.26 | 65.96 | 66.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 66.26 | 65.96 | 66.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 66.25 | 66.02 | 66.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:30:00 | 66.20 | 66.02 | 66.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 66.33 | 66.08 | 66.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 66.49 | 66.08 | 66.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 66.54 | 66.17 | 66.40 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 66.88 | 66.57 | 66.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 67.39 | 66.80 | 66.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 66.57 | 66.99 | 66.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 14:15:00 | 66.57 | 66.99 | 66.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 66.57 | 66.99 | 66.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 66.57 | 66.99 | 66.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 66.46 | 66.89 | 66.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 66.39 | 66.89 | 66.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 65.90 | 66.59 | 66.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 65.61 | 66.39 | 66.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 66.26 | 66.08 | 66.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 66.26 | 66.08 | 66.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 66.26 | 66.08 | 66.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:45:00 | 66.32 | 66.08 | 66.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 66.44 | 66.15 | 66.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 66.11 | 66.15 | 66.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 66.84 | 66.29 | 66.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 66.84 | 66.29 | 66.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 66.57 | 66.35 | 66.40 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 66.80 | 66.44 | 66.43 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 66.27 | 66.46 | 66.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 65.81 | 66.23 | 66.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 64.38 | 63.90 | 64.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 64.38 | 63.90 | 64.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 64.38 | 63.90 | 64.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 64.38 | 63.90 | 64.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 64.75 | 64.07 | 64.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:45:00 | 64.70 | 64.07 | 64.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 65.36 | 64.33 | 64.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 65.36 | 64.33 | 64.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 14:15:00 | 65.46 | 64.83 | 64.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 66.26 | 65.23 | 65.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 64.64 | 65.41 | 65.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 64.64 | 65.41 | 65.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 64.64 | 65.41 | 65.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 64.64 | 65.41 | 65.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 64.48 | 65.22 | 65.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 63.36 | 65.22 | 65.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 62.97 | 64.77 | 64.95 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 65.60 | 64.44 | 64.37 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 64.22 | 64.72 | 64.75 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 64.94 | 64.38 | 64.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 65.19 | 64.69 | 64.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 14:15:00 | 65.95 | 65.97 | 65.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 14:45:00 | 65.90 | 65.97 | 65.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 65.84 | 65.95 | 65.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 65.84 | 65.95 | 65.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 66.12 | 65.98 | 65.69 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 65.36 | 65.68 | 65.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 65.20 | 65.59 | 65.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 65.86 | 65.43 | 65.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 65.86 | 65.43 | 65.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 65.86 | 65.43 | 65.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 65.86 | 65.43 | 65.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 65.80 | 65.50 | 65.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 65.60 | 65.50 | 65.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 65.47 | 65.49 | 65.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:30:00 | 65.49 | 65.49 | 65.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 65.48 | 65.48 | 65.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:45:00 | 65.51 | 65.48 | 65.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 65.53 | 65.49 | 65.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:00:00 | 65.53 | 65.49 | 65.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 65.24 | 65.44 | 65.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 64.96 | 65.27 | 65.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 65.36 | 64.17 | 64.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 65.36 | 64.17 | 64.16 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 11:15:00 | 63.59 | 64.29 | 64.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 62.92 | 64.02 | 64.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 62.20 | 62.14 | 62.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 13:45:00 | 62.17 | 62.14 | 62.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 62.63 | 62.24 | 62.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 63.10 | 62.24 | 62.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 62.42 | 62.27 | 62.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 62.35 | 62.27 | 62.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:15:00 | 62.33 | 62.26 | 62.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 15:15:00 | 62.50 | 62.39 | 62.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 62.50 | 62.39 | 62.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 63.34 | 62.58 | 62.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 62.58 | 62.96 | 62.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 62.58 | 62.96 | 62.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 62.58 | 62.96 | 62.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 62.58 | 62.96 | 62.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 63.98 | 63.17 | 62.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:45:00 | 64.20 | 63.46 | 63.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 12:45:00 | 64.09 | 63.69 | 63.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 63.09 | 64.03 | 64.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 63.09 | 64.03 | 64.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 62.66 | 63.75 | 64.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 62.48 | 62.27 | 62.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 62.48 | 62.27 | 62.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 62.72 | 62.36 | 62.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 62.48 | 62.41 | 62.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 62.89 | 62.51 | 62.76 | SL hit (close>static) qty=1.00 sl=62.88 alert=retest2 |

### Cycle 67 — BUY (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 14:15:00 | 62.00 | 61.21 | 61.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 62.74 | 61.82 | 61.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 14:15:00 | 61.99 | 62.18 | 61.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 15:00:00 | 61.99 | 62.18 | 61.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 62.55 | 62.67 | 62.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:45:00 | 62.74 | 62.54 | 62.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:45:00 | 62.72 | 62.51 | 62.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 63.05 | 62.55 | 62.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:15:00 | 62.90 | 62.55 | 62.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 62.78 | 63.30 | 63.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 62.93 | 63.30 | 63.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 63.19 | 63.28 | 63.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-22 11:15:00 | 61.95 | 62.81 | 62.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 61.95 | 62.81 | 62.92 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 63.44 | 62.93 | 62.90 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 62.35 | 62.92 | 62.94 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 11:15:00 | 63.27 | 63.00 | 62.97 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 62.45 | 62.86 | 62.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 62.25 | 62.74 | 62.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 10:15:00 | 58.56 | 58.41 | 59.85 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-29 10:45:00 | 58.16 | 58.43 | 59.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-29 13:00:00 | 58.15 | 58.34 | 59.02 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 59.85 | 58.68 | 59.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 59.85 | 58.68 | 59.06 | SL hit (close>ema400) qty=1.00 sl=59.06 alert=retest1 |

### Cycle 73 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 61.24 | 59.40 | 59.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 62.18 | 61.19 | 60.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 62.10 | 62.41 | 61.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 62.28 | 62.41 | 61.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 61.84 | 62.30 | 61.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 62.05 | 62.30 | 61.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 61.90 | 62.24 | 61.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:45:00 | 61.79 | 62.24 | 61.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 62.25 | 62.21 | 61.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 62.59 | 62.28 | 61.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:00:00 | 62.56 | 62.38 | 62.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:30:00 | 62.98 | 62.46 | 62.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 62.57 | 62.47 | 62.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 62.29 | 62.44 | 62.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 62.29 | 62.44 | 62.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 62.30 | 62.41 | 62.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 62.25 | 62.41 | 62.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 61.99 | 62.32 | 62.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 61.99 | 62.32 | 62.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 62.20 | 62.30 | 62.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 62.73 | 62.28 | 62.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 62.40 | 63.34 | 63.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 62.40 | 63.34 | 63.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 61.82 | 62.75 | 63.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 61.79 | 61.63 | 62.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 61.79 | 61.63 | 62.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 61.79 | 61.63 | 62.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 61.87 | 61.63 | 62.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 62.19 | 61.74 | 62.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 62.09 | 61.74 | 62.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 62.48 | 61.89 | 62.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 62.35 | 61.89 | 62.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 62.11 | 61.93 | 62.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:45:00 | 61.92 | 62.04 | 62.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 11:15:00 | 60.98 | 60.26 | 60.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 60.98 | 60.26 | 60.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 61.29 | 60.58 | 60.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 60.85 | 60.85 | 60.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 60.85 | 60.85 | 60.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 60.57 | 60.80 | 60.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 60.57 | 60.80 | 60.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 60.45 | 60.73 | 60.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 60.45 | 60.73 | 60.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 60.41 | 60.67 | 60.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 60.41 | 60.67 | 60.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 60.75 | 60.68 | 60.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 60.63 | 60.68 | 60.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 60.50 | 60.65 | 60.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 59.90 | 60.65 | 60.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 60.08 | 60.53 | 60.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 59.45 | 59.88 | 60.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 59.75 | 59.39 | 59.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 59.75 | 59.39 | 59.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 59.75 | 59.39 | 59.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:30:00 | 58.94 | 59.25 | 59.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:15:00 | 58.98 | 59.26 | 59.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 58.43 | 57.86 | 57.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 58.43 | 57.86 | 57.80 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 11:15:00 | 57.39 | 57.81 | 57.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 12:15:00 | 57.16 | 57.68 | 57.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 53.70 | 53.30 | 53.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:30:00 | 53.69 | 53.30 | 53.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 53.73 | 53.39 | 53.86 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 54.69 | 54.12 | 54.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 54.98 | 54.29 | 54.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 55.69 | 55.70 | 55.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 55.69 | 55.70 | 55.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 57.40 | 57.25 | 56.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 56.97 | 57.25 | 56.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 57.08 | 57.26 | 57.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 57.08 | 57.26 | 57.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 57.08 | 57.23 | 57.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:30:00 | 57.09 | 57.23 | 57.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 57.10 | 57.20 | 57.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 57.09 | 57.20 | 57.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 57.02 | 57.16 | 57.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 57.02 | 57.16 | 57.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 56.95 | 57.12 | 57.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 56.93 | 57.12 | 57.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 56.83 | 57.06 | 57.01 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 56.56 | 56.94 | 56.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 13:15:00 | 56.18 | 56.72 | 56.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 57.08 | 56.80 | 56.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 57.08 | 56.80 | 56.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 57.08 | 56.80 | 56.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 57.08 | 56.80 | 56.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 56.33 | 56.70 | 56.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 57.19 | 56.70 | 56.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 56.52 | 56.67 | 56.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:00:00 | 56.33 | 56.60 | 56.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 56.81 | 56.44 | 56.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 56.81 | 56.44 | 56.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 14:15:00 | 57.18 | 56.59 | 56.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 10:15:00 | 56.61 | 56.70 | 56.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 56.61 | 56.70 | 56.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 56.90 | 56.74 | 56.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 57.07 | 56.74 | 56.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:15:00 | 57.10 | 56.78 | 56.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 55.67 | 57.65 | 57.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 55.67 | 57.65 | 57.83 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 57.96 | 57.35 | 57.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 15:15:00 | 58.03 | 57.56 | 57.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 62.83 | 62.91 | 62.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 13:30:00 | 62.90 | 62.91 | 62.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 66.51 | 67.58 | 67.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 66.51 | 67.58 | 67.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 66.03 | 67.27 | 67.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 66.01 | 67.27 | 67.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 66.24 | 67.06 | 67.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 65.38 | 66.34 | 66.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 66.45 | 66.23 | 66.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 12:00:00 | 66.45 | 66.23 | 66.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 66.30 | 66.24 | 66.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 13:15:00 | 66.05 | 66.24 | 66.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 65.96 | 66.26 | 66.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:30:00 | 66.09 | 66.37 | 66.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 65.91 | 66.40 | 66.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 66.83 | 66.49 | 66.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-29 12:15:00 | 66.83 | 66.49 | 66.55 | SL hit (close>static) qty=1.00 sl=66.65 alert=retest2 |

### Cycle 85 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 67.16 | 66.62 | 66.60 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 65.68 | 66.51 | 66.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 65.14 | 66.23 | 66.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 66.33 | 65.56 | 65.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 66.33 | 65.56 | 65.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 66.33 | 65.56 | 65.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 66.39 | 65.56 | 65.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 65.99 | 65.64 | 65.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 65.84 | 65.73 | 65.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 13:15:00 | 66.40 | 65.92 | 65.99 | SL hit (close>static) qty=1.00 sl=66.35 alert=retest2 |

### Cycle 87 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 67.22 | 66.23 | 66.12 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 65.68 | 66.25 | 66.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 65.49 | 66.00 | 66.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 66.65 | 66.07 | 66.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 66.65 | 66.07 | 66.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 66.65 | 66.07 | 66.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 66.65 | 66.07 | 66.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 66.86 | 66.23 | 66.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 66.95 | 66.23 | 66.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 67.03 | 66.39 | 66.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 68.77 | 67.08 | 66.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 67.25 | 67.59 | 67.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 67.25 | 67.59 | 67.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 66.56 | 67.39 | 67.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 66.56 | 67.39 | 67.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 66.25 | 67.16 | 66.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 66.15 | 67.16 | 66.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 66.12 | 66.83 | 66.86 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 68.61 | 66.85 | 66.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 69.34 | 67.96 | 67.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 69.01 | 69.04 | 68.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 12:15:00 | 69.40 | 69.03 | 68.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 13:15:00 | 69.19 | 69.06 | 68.67 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 14:00:00 | 69.33 | 69.11 | 68.73 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 11:30:00 | 69.21 | 69.34 | 69.01 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 69.14 | 69.30 | 69.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 69.09 | 69.30 | 69.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 69.24 | 69.34 | 69.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:45:00 | 69.17 | 69.34 | 69.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 69.26 | 69.32 | 69.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:15:00 | 69.30 | 69.32 | 69.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 69.31 | 69.32 | 69.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 10:15:00 | 68.99 | 69.23 | 69.19 | SL hit (close<ema400) qty=1.00 sl=69.19 alert=retest1 |

### Cycle 92 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 68.80 | 69.11 | 69.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 14:15:00 | 68.67 | 69.03 | 69.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 66.85 | 66.62 | 67.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 66.85 | 66.62 | 67.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 67.03 | 66.74 | 67.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 67.03 | 66.74 | 67.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 67.13 | 66.82 | 67.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 67.13 | 66.82 | 67.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 67.12 | 66.88 | 67.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 67.19 | 66.88 | 67.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 67.01 | 66.90 | 67.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:15:00 | 66.77 | 66.98 | 67.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 67.74 | 66.92 | 66.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 67.74 | 66.92 | 66.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 12:15:00 | 68.01 | 67.13 | 67.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 67.70 | 68.00 | 67.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 14:00:00 | 67.70 | 68.00 | 67.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 68.01 | 68.00 | 67.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 68.14 | 68.01 | 67.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 67.61 | 67.88 | 67.70 | SL hit (close<static) qty=1.00 sl=67.67 alert=retest2 |

### Cycle 94 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 67.39 | 67.96 | 68.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 13:15:00 | 67.14 | 67.79 | 67.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 67.28 | 67.24 | 67.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 14:00:00 | 67.28 | 67.24 | 67.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 67.46 | 67.28 | 67.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 67.46 | 67.28 | 67.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 67.51 | 67.33 | 67.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 67.09 | 67.33 | 67.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 67.08 | 67.28 | 67.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 66.81 | 67.24 | 67.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:00:00 | 66.86 | 67.17 | 67.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 66.80 | 67.11 | 67.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:15:00 | 66.85 | 67.11 | 67.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 67.09 | 67.00 | 67.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 67.09 | 67.00 | 67.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 70.52 | 67.70 | 67.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 70.52 | 67.70 | 67.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 71.31 | 68.42 | 67.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 71.84 | 71.86 | 70.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:00:00 | 71.84 | 71.86 | 70.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 71.36 | 71.58 | 71.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:30:00 | 71.67 | 71.62 | 71.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 70.86 | 71.28 | 71.22 | SL hit (close<static) qty=1.00 sl=70.95 alert=retest2 |

### Cycle 96 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 70.60 | 71.07 | 71.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 70.16 | 70.81 | 70.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 11:15:00 | 70.80 | 70.75 | 70.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 12:00:00 | 70.80 | 70.75 | 70.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 70.21 | 70.64 | 70.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 69.79 | 70.55 | 70.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 71.19 | 70.67 | 70.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 71.19 | 70.67 | 70.66 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 70.26 | 70.61 | 70.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 15:15:00 | 69.99 | 70.49 | 70.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 70.80 | 70.55 | 70.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 70.80 | 70.55 | 70.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 70.80 | 70.55 | 70.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 70.80 | 70.55 | 70.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 70.92 | 70.62 | 70.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 71.08 | 70.62 | 70.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 71.03 | 70.71 | 70.67 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 70.37 | 70.64 | 70.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 69.74 | 70.38 | 70.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 70.17 | 69.97 | 70.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 70.17 | 69.97 | 70.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 70.03 | 69.98 | 70.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 69.75 | 69.98 | 70.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 69.99 | 69.97 | 70.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 69.83 | 70.04 | 70.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 70.86 | 70.20 | 70.22 | SL hit (close>static) qty=1.00 sl=70.24 alert=retest2 |

### Cycle 101 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 70.84 | 70.33 | 70.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 71.05 | 70.47 | 70.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 71.66 | 71.78 | 71.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 71.66 | 71.78 | 71.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 71.93 | 71.97 | 71.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 71.93 | 71.97 | 71.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 71.51 | 71.88 | 71.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 71.53 | 71.88 | 71.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 72.03 | 71.91 | 71.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 71.55 | 71.91 | 71.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 71.80 | 71.86 | 71.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 71.79 | 71.86 | 71.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 72.05 | 71.89 | 71.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 71.79 | 71.89 | 71.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 77.34 | 77.74 | 77.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 77.37 | 77.74 | 77.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 77.88 | 77.77 | 77.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 77.38 | 77.77 | 77.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 77.53 | 77.67 | 77.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 77.71 | 77.67 | 77.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 77.66 | 77.66 | 77.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 77.44 | 77.66 | 77.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 77.83 | 77.70 | 77.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:00:00 | 77.91 | 77.74 | 77.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 77.35 | 77.69 | 77.59 | SL hit (close<static) qty=1.00 sl=77.44 alert=retest2 |

### Cycle 102 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 77.49 | 77.54 | 77.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 77.35 | 77.50 | 77.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 10:15:00 | 77.70 | 77.53 | 77.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 10:15:00 | 77.70 | 77.53 | 77.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 77.70 | 77.53 | 77.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 77.63 | 77.53 | 77.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 77.17 | 77.46 | 77.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:45:00 | 76.92 | 77.34 | 77.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 13:15:00 | 73.07 | 75.46 | 75.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 73.56 | 73.48 | 74.26 | SL hit (close>ema200) qty=0.50 sl=73.48 alert=retest2 |

### Cycle 103 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 73.97 | 73.66 | 73.62 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 73.02 | 73.55 | 73.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 72.90 | 73.33 | 73.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 72.70 | 72.54 | 72.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 72.70 | 72.54 | 72.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 72.70 | 72.54 | 72.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 72.88 | 72.54 | 72.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 72.78 | 72.60 | 72.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 72.95 | 72.60 | 72.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 72.75 | 72.63 | 72.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 72.97 | 72.63 | 72.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 72.93 | 72.65 | 72.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 72.93 | 72.65 | 72.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 72.90 | 72.70 | 72.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 72.10 | 72.75 | 72.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 68.49 | 69.38 | 70.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 69.39 | 69.30 | 69.86 | SL hit (close>ema200) qty=0.50 sl=69.30 alert=retest2 |

### Cycle 105 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 69.13 | 68.80 | 68.79 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 68.67 | 68.90 | 68.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 68.06 | 68.65 | 68.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 68.59 | 68.41 | 68.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 68.59 | 68.41 | 68.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 68.59 | 68.41 | 68.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 68.59 | 68.41 | 68.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 69.17 | 68.56 | 68.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 69.17 | 68.56 | 68.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 69.36 | 68.72 | 68.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 70.19 | 68.72 | 68.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 69.43 | 68.86 | 68.80 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 68.90 | 69.43 | 69.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 68.74 | 69.20 | 69.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 69.92 | 69.23 | 69.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 69.92 | 69.23 | 69.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 69.92 | 69.23 | 69.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 69.92 | 69.23 | 69.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 70.14 | 69.41 | 69.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 70.76 | 70.13 | 69.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 70.88 | 71.11 | 70.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 70.88 | 71.11 | 70.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 70.88 | 71.11 | 70.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 70.59 | 71.11 | 70.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 70.59 | 70.98 | 70.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 70.59 | 70.98 | 70.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 70.59 | 70.90 | 70.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 70.60 | 70.90 | 70.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 70.09 | 70.55 | 70.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 69.70 | 70.38 | 70.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 70.00 | 69.99 | 70.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 70.00 | 69.99 | 70.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 70.00 | 69.99 | 70.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 69.34 | 70.00 | 70.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 69.36 | 69.80 | 70.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:15:00 | 69.47 | 69.62 | 69.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 70.02 | 69.02 | 68.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 70.02 | 69.02 | 68.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 70.32 | 69.65 | 69.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 13:15:00 | 72.60 | 72.62 | 72.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:45:00 | 72.53 | 72.62 | 72.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 73.22 | 73.19 | 72.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:15:00 | 72.84 | 73.19 | 72.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 72.70 | 73.09 | 72.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 72.70 | 73.09 | 72.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 72.69 | 73.01 | 72.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 72.81 | 73.01 | 72.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 72.98 | 72.95 | 72.86 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 72.44 | 72.75 | 72.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 10:15:00 | 72.06 | 72.47 | 72.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 71.55 | 71.49 | 71.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 15:15:00 | 71.60 | 71.49 | 71.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 71.82 | 71.57 | 71.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:15:00 | 71.55 | 71.63 | 71.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 71.57 | 71.65 | 71.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 71.56 | 71.63 | 71.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 72.15 | 71.79 | 71.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 72.15 | 71.79 | 71.79 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 71.75 | 71.88 | 71.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 71.44 | 71.75 | 71.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 71.09 | 70.92 | 71.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 71.09 | 70.92 | 71.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 71.09 | 70.92 | 71.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 71.09 | 70.92 | 71.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 70.32 | 70.80 | 71.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 70.11 | 70.55 | 70.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 69.89 | 69.65 | 69.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 69.89 | 69.65 | 69.63 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 69.30 | 69.58 | 69.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 10:15:00 | 68.94 | 69.45 | 69.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 09:15:00 | 69.16 | 69.10 | 69.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 69.16 | 69.10 | 69.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 69.16 | 69.10 | 69.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 68.81 | 69.04 | 69.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 69.80 | 69.26 | 69.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 69.80 | 69.26 | 69.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 70.45 | 69.50 | 69.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 73.98 | 74.02 | 73.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:45:00 | 73.65 | 74.02 | 73.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 73.68 | 73.86 | 73.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 73.91 | 73.86 | 73.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 73.84 | 73.86 | 73.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 73.15 | 73.72 | 73.52 | SL hit (close<static) qty=1.00 sl=73.50 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 72.73 | 73.31 | 73.36 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 73.61 | 73.38 | 73.36 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 72.99 | 73.28 | 73.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 72.02 | 73.01 | 73.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 72.50 | 72.22 | 72.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 72.50 | 72.22 | 72.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 72.50 | 72.22 | 72.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 72.65 | 72.22 | 72.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 72.49 | 72.28 | 72.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 72.54 | 72.28 | 72.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 72.09 | 72.24 | 72.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 72.50 | 72.24 | 72.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 75.31 | 72.72 | 72.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 76.36 | 73.45 | 72.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 78.56 | 78.65 | 77.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 10:00:00 | 78.56 | 78.65 | 77.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 78.40 | 78.37 | 77.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 78.08 | 78.37 | 77.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 77.97 | 78.31 | 77.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 77.97 | 78.31 | 77.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 78.18 | 78.29 | 77.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 78.04 | 78.29 | 77.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 78.54 | 78.81 | 78.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 78.57 | 78.81 | 78.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 78.24 | 78.69 | 78.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 78.27 | 78.69 | 78.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 79.01 | 78.76 | 78.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 79.12 | 78.87 | 78.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 80.38 | 78.86 | 78.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 80.25 | 80.78 | 80.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 80.25 | 80.78 | 80.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 80.00 | 80.62 | 80.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 80.30 | 80.11 | 80.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 80.30 | 80.11 | 80.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 80.30 | 80.11 | 80.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 80.25 | 80.11 | 80.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 81.62 | 80.41 | 80.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 81.62 | 80.41 | 80.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 81.20 | 80.57 | 80.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 81.65 | 81.25 | 80.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 81.21 | 81.24 | 81.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:45:00 | 81.26 | 81.24 | 81.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 80.03 | 80.97 | 80.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 80.03 | 80.97 | 80.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 79.86 | 80.74 | 80.82 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 81.50 | 80.86 | 80.79 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 80.17 | 80.80 | 80.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 80.07 | 80.65 | 80.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 80.52 | 80.49 | 80.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:00:00 | 80.52 | 80.49 | 80.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 80.55 | 80.50 | 80.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 80.58 | 80.50 | 80.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 80.34 | 80.47 | 80.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:45:00 | 80.09 | 80.39 | 80.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 81.72 | 80.68 | 80.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 81.72 | 80.68 | 80.66 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 80.31 | 80.80 | 80.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 80.00 | 80.64 | 80.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 78.93 | 78.91 | 79.37 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 13:30:00 | 78.62 | 78.92 | 79.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 14:30:00 | 78.64 | 78.82 | 79.25 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 12:00:00 | 78.51 | 78.61 | 79.00 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 78.99 | 78.42 | 78.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 78.99 | 78.42 | 78.74 | SL hit (close>ema400) qty=1.00 sl=78.74 alert=retest1 |

### Cycle 129 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 79.37 | 78.93 | 78.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 80.62 | 79.36 | 79.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 80.10 | 80.22 | 79.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 80.10 | 80.22 | 79.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 80.29 | 80.23 | 79.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 80.30 | 80.23 | 79.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 80.71 | 80.29 | 79.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 81.99 | 80.55 | 80.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 80.71 | 80.95 | 80.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 80.71 | 80.95 | 80.96 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 81.15 | 80.73 | 80.70 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 79.92 | 80.55 | 80.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 79.00 | 80.24 | 80.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 79.98 | 79.63 | 80.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 79.98 | 79.63 | 80.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 80.59 | 79.83 | 80.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 80.59 | 79.83 | 80.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 80.80 | 80.02 | 80.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 80.80 | 80.02 | 80.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 80.89 | 80.35 | 80.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 81.08 | 80.69 | 80.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 80.55 | 80.74 | 80.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 80.55 | 80.74 | 80.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 80.55 | 80.74 | 80.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 80.55 | 80.74 | 80.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 80.48 | 80.69 | 80.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 80.48 | 80.69 | 80.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 80.60 | 80.67 | 80.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 80.17 | 80.67 | 80.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 80.59 | 80.66 | 80.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 81.15 | 80.66 | 80.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 84.39 | 84.82 | 84.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 84.39 | 84.82 | 84.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 84.27 | 84.71 | 84.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 85.10 | 84.70 | 84.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 85.10 | 84.70 | 84.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 85.10 | 84.70 | 84.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 85.10 | 84.70 | 84.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 84.99 | 84.76 | 84.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 85.15 | 84.76 | 84.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 85.28 | 84.86 | 84.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 09:15:00 | 85.45 | 85.09 | 84.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 11:15:00 | 84.80 | 85.04 | 84.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 11:15:00 | 84.80 | 85.04 | 84.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 84.80 | 85.04 | 84.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 84.80 | 85.04 | 84.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 84.78 | 84.99 | 84.95 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 84.60 | 84.88 | 84.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 84.48 | 84.80 | 84.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 84.77 | 84.73 | 84.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 10:30:00 | 84.62 | 84.73 | 84.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 84.51 | 84.68 | 84.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:45:00 | 84.48 | 84.64 | 84.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 85.05 | 84.74 | 84.78 | SL hit (close>static) qty=1.00 sl=84.99 alert=retest2 |

### Cycle 137 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 85.63 | 84.95 | 84.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 86.87 | 85.73 | 85.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 85.82 | 85.95 | 85.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 15:00:00 | 85.82 | 85.95 | 85.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 85.40 | 85.83 | 85.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 85.53 | 85.83 | 85.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 85.28 | 85.72 | 85.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 85.24 | 85.72 | 85.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 85.10 | 85.55 | 85.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 84.72 | 85.38 | 85.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 85.48 | 85.26 | 85.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 85.48 | 85.26 | 85.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 85.48 | 85.26 | 85.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 84.91 | 85.15 | 85.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 84.83 | 85.01 | 85.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 86.08 | 85.17 | 85.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 12:15:00 | 86.08 | 85.17 | 85.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 13:15:00 | 86.12 | 85.36 | 85.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 13:15:00 | 85.69 | 85.87 | 85.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:00:00 | 85.69 | 85.87 | 85.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 86.05 | 85.91 | 85.63 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 84.47 | 85.46 | 85.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 84.07 | 85.18 | 85.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 85.37 | 85.22 | 85.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 85.37 | 85.22 | 85.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 85.37 | 85.22 | 85.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 85.37 | 85.22 | 85.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 85.06 | 85.19 | 85.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 14:15:00 | 84.87 | 85.19 | 85.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 80.63 | 82.01 | 82.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 81.50 | 81.48 | 82.02 | SL hit (close>ema200) qty=0.50 sl=81.48 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 83.57 | 82.50 | 82.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 83.92 | 82.94 | 82.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 83.71 | 83.83 | 83.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:45:00 | 83.68 | 83.83 | 83.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 83.05 | 83.68 | 83.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 83.00 | 83.68 | 83.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 82.88 | 83.52 | 83.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 82.88 | 83.52 | 83.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 82.59 | 83.33 | 83.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 82.28 | 83.33 | 83.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 82.38 | 83.14 | 83.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 82.01 | 83.14 | 83.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 81.87 | 82.89 | 83.00 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 83.18 | 82.78 | 82.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 83.41 | 82.91 | 82.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 83.50 | 83.63 | 83.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 83.50 | 83.63 | 83.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 83.50 | 83.63 | 83.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 82.68 | 83.63 | 83.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 83.14 | 83.53 | 83.34 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 82.05 | 83.06 | 83.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 80.90 | 82.24 | 82.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 81.24 | 81.04 | 81.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 81.24 | 81.04 | 81.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 83.68 | 81.63 | 81.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 83.51 | 81.63 | 81.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 84.29 | 82.59 | 82.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 84.29 | 82.59 | 82.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 84.55 | 82.98 | 82.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 84.95 | 85.02 | 84.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 84.95 | 85.02 | 84.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 84.69 | 84.96 | 84.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 84.55 | 84.96 | 84.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 84.64 | 84.89 | 84.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:15:00 | 84.60 | 84.89 | 84.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 84.35 | 84.79 | 84.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 84.35 | 84.79 | 84.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 84.77 | 84.78 | 84.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 84.52 | 84.78 | 84.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 85.03 | 84.92 | 84.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 84.75 | 84.92 | 84.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 84.66 | 84.88 | 84.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 84.66 | 84.88 | 84.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 84.75 | 84.85 | 84.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 84.65 | 84.85 | 84.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 84.83 | 84.85 | 84.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:30:00 | 84.70 | 84.85 | 84.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 84.80 | 84.84 | 84.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:30:00 | 84.51 | 84.84 | 84.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 84.72 | 84.81 | 84.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 84.60 | 84.81 | 84.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 09:15:00 | 84.09 | 84.67 | 84.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 83.96 | 84.33 | 84.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 82.20 | 81.60 | 82.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 13:15:00 | 82.20 | 81.60 | 82.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 82.20 | 81.60 | 82.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 82.20 | 81.60 | 82.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 81.85 | 81.65 | 82.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 81.42 | 81.65 | 82.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 12:15:00 | 82.57 | 81.95 | 82.10 | SL hit (close>static) qty=1.00 sl=82.20 alert=retest2 |

### Cycle 147 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 82.90 | 82.23 | 82.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 83.40 | 82.66 | 82.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 12:15:00 | 84.07 | 84.25 | 83.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:30:00 | 83.95 | 84.25 | 83.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 83.00 | 83.94 | 83.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 83.00 | 83.94 | 83.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 82.60 | 83.67 | 83.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 83.51 | 83.53 | 83.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:45:00 | 83.22 | 83.46 | 83.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 83.14 | 83.38 | 83.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 83.14 | 83.38 | 83.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 69.27 | 80.54 | 82.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 70.80 | 70.68 | 73.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 72.65 | 70.93 | 71.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 72.65 | 70.93 | 71.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 72.65 | 70.93 | 71.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 72.35 | 71.21 | 72.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 72.24 | 71.21 | 72.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:45:00 | 71.97 | 72.09 | 72.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 73.04 | 72.28 | 72.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 10:15:00 | 73.04 | 72.28 | 72.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 11:15:00 | 73.65 | 72.55 | 72.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 72.46 | 72.95 | 72.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 72.46 | 72.95 | 72.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 72.46 | 72.95 | 72.71 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 71.73 | 72.43 | 72.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 70.60 | 71.83 | 72.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 70.75 | 70.58 | 71.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 70.77 | 70.58 | 71.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 70.73 | 70.47 | 70.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 70.97 | 70.47 | 70.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 71.01 | 70.58 | 70.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 71.00 | 70.58 | 70.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 70.77 | 70.62 | 70.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 70.51 | 70.51 | 70.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 66.98 | 69.65 | 70.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 67.32 | 67.11 | 68.00 | SL hit (close>ema200) qty=0.50 sl=67.11 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 64.19 | 63.47 | 63.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 64.85 | 63.93 | 63.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 62.97 | 64.33 | 64.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 62.97 | 64.33 | 64.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 62.97 | 64.33 | 64.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 63.05 | 64.33 | 64.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 62.83 | 64.03 | 63.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:15:00 | 62.75 | 64.03 | 63.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 62.89 | 63.80 | 63.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 62.52 | 63.24 | 63.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 63.35 | 63.22 | 63.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 63.35 | 63.22 | 63.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 63.35 | 63.22 | 63.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 62.89 | 63.23 | 63.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 61.45 | 63.19 | 63.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 62.97 | 62.18 | 62.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 62.97 | 62.18 | 62.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 63.24 | 62.54 | 62.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 61.83 | 62.68 | 62.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 61.83 | 62.68 | 62.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 61.83 | 62.68 | 62.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 61.83 | 62.68 | 62.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 61.71 | 62.49 | 62.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 61.79 | 62.49 | 62.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 61.86 | 62.26 | 62.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 59.61 | 61.57 | 61.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 60.50 | 60.01 | 60.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 60.50 | 60.01 | 60.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 60.50 | 60.01 | 60.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 60.03 | 60.01 | 60.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 61.12 | 60.38 | 60.78 | SL hit (close>static) qty=1.00 sl=60.95 alert=retest2 |

### Cycle 155 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 61.10 | 60.22 | 60.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 61.24 | 60.43 | 60.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 64.51 | 65.68 | 65.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 64.51 | 65.68 | 65.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 64.51 | 65.68 | 65.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 66.78 | 65.10 | 64.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 67.85 | 68.01 | 68.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 67.85 | 68.01 | 68.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 67.13 | 67.70 | 67.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 68.75 | 67.71 | 67.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 68.75 | 67.71 | 67.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 68.75 | 67.71 | 67.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 69.55 | 67.71 | 67.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 68.63 | 67.90 | 67.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 69.44 | 68.35 | 68.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 69.30 | 69.31 | 68.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 12:15:00 | 68.76 | 69.10 | 68.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 68.76 | 69.10 | 68.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:30:00 | 69.00 | 68.99 | 68.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 68.95 | 68.99 | 68.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 69.55 | 69.22 | 68.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 69.09 | 69.65 | 69.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 69.09 | 69.65 | 69.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 69.03 | 69.52 | 69.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 69.49 | 69.15 | 69.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 69.49 | 69.15 | 69.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 69.49 | 69.15 | 69.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 69.31 | 69.15 | 69.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:00:00 | 69.28 | 69.27 | 69.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 69.54 | 69.33 | 69.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 69.54 | 69.33 | 69.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 69.74 | 69.41 | 69.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 75.30 | 2024-05-13 14:15:00 | 77.25 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-05-27 12:00:00 | 78.20 | 2024-05-29 10:15:00 | 77.75 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-28 12:45:00 | 78.20 | 2024-05-29 10:15:00 | 77.75 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-28 14:30:00 | 78.20 | 2024-05-29 10:15:00 | 77.75 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-29 09:30:00 | 78.35 | 2024-05-29 10:15:00 | 77.75 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-05-30 14:30:00 | 77.10 | 2024-05-30 15:15:00 | 78.10 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-05-31 10:45:00 | 77.05 | 2024-06-03 09:15:00 | 78.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-11 10:00:00 | 77.74 | 2024-06-13 15:15:00 | 77.47 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-06-11 12:00:00 | 77.66 | 2024-06-13 15:15:00 | 77.47 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-06-11 12:45:00 | 77.74 | 2024-06-13 15:15:00 | 77.47 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-06-12 09:15:00 | 77.74 | 2024-06-13 15:15:00 | 77.47 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-06-12 10:15:00 | 78.32 | 2024-06-13 15:15:00 | 77.47 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-06-14 15:15:00 | 78.12 | 2024-06-26 09:15:00 | 82.54 | STOP_HIT | 1.00 | 5.66% |
| SELL | retest2 | 2024-07-12 12:30:00 | 78.40 | 2024-07-26 09:15:00 | 74.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 13:30:00 | 78.50 | 2024-07-26 09:15:00 | 74.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 14:00:00 | 78.42 | 2024-07-26 09:15:00 | 74.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 12:30:00 | 78.40 | 2024-07-29 09:15:00 | 74.85 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2024-07-12 13:30:00 | 78.50 | 2024-07-29 09:15:00 | 74.85 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2024-07-12 14:00:00 | 78.42 | 2024-07-29 09:15:00 | 74.85 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2024-07-31 15:15:00 | 76.13 | 2024-08-01 13:15:00 | 75.46 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-08-01 11:00:00 | 76.07 | 2024-08-01 13:15:00 | 75.46 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest1 | 2024-08-06 13:30:00 | 72.32 | 2024-08-09 09:15:00 | 73.20 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest1 | 2024-08-06 14:00:00 | 72.17 | 2024-08-09 09:15:00 | 73.20 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-08-16 10:15:00 | 70.99 | 2024-08-19 09:15:00 | 71.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-08-26 09:15:00 | 74.65 | 2024-08-26 12:15:00 | 74.11 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-08-26 11:15:00 | 74.48 | 2024-08-26 12:15:00 | 74.11 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-09-18 09:15:00 | 73.63 | 2024-09-18 09:15:00 | 72.53 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-09-26 13:45:00 | 72.90 | 2024-09-26 14:15:00 | 73.54 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-10-01 09:30:00 | 74.81 | 2024-10-01 13:15:00 | 74.06 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-10-01 10:45:00 | 74.66 | 2024-10-01 13:15:00 | 74.06 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-07 10:30:00 | 70.98 | 2024-10-08 09:15:00 | 72.82 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-10-07 11:30:00 | 70.90 | 2024-10-08 09:15:00 | 72.82 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-10-07 12:30:00 | 71.32 | 2024-10-08 09:15:00 | 72.82 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-07 13:30:00 | 71.27 | 2024-10-08 09:15:00 | 72.82 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-10-21 12:15:00 | 71.30 | 2024-10-23 09:15:00 | 67.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:15:00 | 71.30 | 2024-10-24 09:15:00 | 67.95 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2024-12-12 09:30:00 | 64.96 | 2024-12-18 10:15:00 | 65.36 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-26 10:15:00 | 62.35 | 2024-12-27 15:15:00 | 62.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-12-27 12:15:00 | 62.33 | 2024-12-27 15:15:00 | 62.50 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-01-01 10:45:00 | 64.20 | 2025-01-06 12:15:00 | 63.09 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-01-01 12:45:00 | 64.09 | 2025-01-06 12:15:00 | 63.09 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-01-09 09:15:00 | 62.48 | 2025-01-09 09:15:00 | 62.89 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-01-09 10:45:00 | 62.25 | 2025-01-13 14:15:00 | 59.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 62.25 | 2025-01-14 09:15:00 | 61.19 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2025-01-10 09:15:00 | 61.69 | 2025-01-14 14:15:00 | 62.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-01-17 11:45:00 | 62.74 | 2025-01-22 11:15:00 | 61.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-01-17 14:45:00 | 62.72 | 2025-01-22 11:15:00 | 61.95 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-01-20 09:15:00 | 63.05 | 2025-01-22 11:15:00 | 61.95 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-01-20 10:15:00 | 62.90 | 2025-01-22 11:15:00 | 61.95 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest1 | 2025-01-29 10:45:00 | 58.16 | 2025-01-29 14:15:00 | 59.85 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest1 | 2025-01-29 13:00:00 | 58.15 | 2025-01-29 14:15:00 | 59.85 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-02-03 10:45:00 | 62.59 | 2025-02-10 11:15:00 | 62.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-02-03 13:00:00 | 62.56 | 2025-02-10 11:15:00 | 62.40 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-02-04 09:30:00 | 62.98 | 2025-02-10 11:15:00 | 62.40 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-02-04 11:15:00 | 62.57 | 2025-02-10 11:15:00 | 62.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-02-05 09:15:00 | 62.73 | 2025-02-10 11:15:00 | 62.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-02-13 12:45:00 | 61.92 | 2025-02-20 11:15:00 | 60.98 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-02-27 13:30:00 | 58.94 | 2025-03-06 09:15:00 | 58.43 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-02-27 15:15:00 | 58.98 | 2025-03-06 09:15:00 | 58.43 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-03-28 11:00:00 | 56.33 | 2025-04-01 13:15:00 | 56.81 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-04-02 12:15:00 | 57.07 | 2025-04-07 09:15:00 | 55.67 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-04-02 13:15:00 | 57.10 | 2025-04-07 09:15:00 | 55.67 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-04-28 13:15:00 | 66.05 | 2025-04-29 12:15:00 | 66.83 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-04-28 14:15:00 | 65.96 | 2025-04-29 12:15:00 | 66.83 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-04-29 10:30:00 | 66.09 | 2025-04-29 12:15:00 | 66.83 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-04-29 11:45:00 | 65.91 | 2025-04-29 12:15:00 | 66.83 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-05-02 12:15:00 | 65.84 | 2025-05-02 13:15:00 | 66.40 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest1 | 2025-05-14 12:15:00 | 69.40 | 2025-05-19 10:15:00 | 68.99 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-05-14 13:15:00 | 69.19 | 2025-05-19 10:15:00 | 68.99 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-05-14 14:00:00 | 69.33 | 2025-05-19 10:15:00 | 68.99 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-05-15 11:30:00 | 69.21 | 2025-05-19 10:15:00 | 68.99 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-19 10:30:00 | 69.43 | 2025-05-19 11:15:00 | 69.04 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-05-26 10:15:00 | 66.77 | 2025-05-27 11:15:00 | 67.74 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-05-29 09:15:00 | 68.14 | 2025-05-29 10:15:00 | 67.61 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-05-30 09:15:00 | 68.18 | 2025-05-30 11:15:00 | 67.53 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-30 13:15:00 | 68.26 | 2025-06-03 11:15:00 | 67.30 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-05-30 14:15:00 | 68.12 | 2025-06-03 11:15:00 | 67.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-02 09:15:00 | 68.45 | 2025-06-03 11:15:00 | 67.30 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-06-03 10:45:00 | 68.34 | 2025-06-03 11:15:00 | 67.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-05 10:30:00 | 66.81 | 2025-06-06 10:15:00 | 70.52 | STOP_HIT | 1.00 | -5.55% |
| SELL | retest2 | 2025-06-05 12:00:00 | 66.86 | 2025-06-06 10:15:00 | 70.52 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2025-06-05 12:30:00 | 66.80 | 2025-06-06 10:15:00 | 70.52 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2025-06-05 13:15:00 | 66.85 | 2025-06-06 10:15:00 | 70.52 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest2 | 2025-06-11 11:30:00 | 71.67 | 2025-06-12 10:15:00 | 70.86 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-06-16 09:15:00 | 69.79 | 2025-06-17 09:15:00 | 71.19 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-20 12:15:00 | 69.75 | 2025-06-23 10:15:00 | 70.86 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-06-20 14:15:00 | 69.99 | 2025-06-23 10:15:00 | 70.86 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-23 09:30:00 | 69.83 | 2025-06-23 10:15:00 | 70.86 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-08 14:00:00 | 77.91 | 2025-07-09 09:15:00 | 77.35 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-07-10 12:45:00 | 76.92 | 2025-07-14 13:15:00 | 73.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 12:45:00 | 76.92 | 2025-07-16 10:15:00 | 73.56 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2025-07-25 09:15:00 | 72.10 | 2025-07-31 09:15:00 | 68.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:15:00 | 72.10 | 2025-07-31 11:15:00 | 69.39 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-08-26 09:30:00 | 69.34 | 2025-09-02 09:15:00 | 70.02 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-26 10:30:00 | 69.36 | 2025-09-02 09:15:00 | 70.02 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-08-26 13:15:00 | 69.47 | 2025-09-02 09:15:00 | 70.02 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-17 12:15:00 | 71.55 | 2025-09-18 09:15:00 | 72.15 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-09-17 13:15:00 | 71.57 | 2025-09-18 09:15:00 | 72.15 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-17 14:00:00 | 71.56 | 2025-09-18 09:15:00 | 72.15 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-24 14:15:00 | 70.11 | 2025-09-30 15:15:00 | 69.89 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-10-03 13:15:00 | 68.81 | 2025-10-06 10:15:00 | 69.80 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-14 09:15:00 | 73.91 | 2025-10-14 10:15:00 | 73.15 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-14 10:00:00 | 73.84 | 2025-10-14 10:15:00 | 73.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-29 14:00:00 | 79.12 | 2025-11-06 10:15:00 | 80.25 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-10-31 09:15:00 | 80.38 | 2025-11-06 10:15:00 | 80.25 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-14 13:45:00 | 80.09 | 2025-11-17 09:15:00 | 81.72 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-11-21 13:30:00 | 78.62 | 2025-11-25 09:15:00 | 78.99 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-11-21 14:30:00 | 78.64 | 2025-11-25 09:15:00 | 78.99 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-11-24 12:00:00 | 78.51 | 2025-11-25 09:15:00 | 78.99 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-02 09:15:00 | 81.99 | 2025-12-03 14:15:00 | 80.71 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-12 09:15:00 | 81.15 | 2025-12-24 13:15:00 | 84.39 | STOP_HIT | 1.00 | 3.99% |
| SELL | retest2 | 2025-12-30 12:45:00 | 84.48 | 2025-12-30 14:15:00 | 85.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-30 15:15:00 | 84.50 | 2025-12-31 09:15:00 | 85.16 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-06 11:30:00 | 84.91 | 2026-01-08 12:15:00 | 86.08 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-07 12:15:00 | 84.83 | 2026-01-08 12:15:00 | 86.08 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-12 14:15:00 | 84.87 | 2026-01-21 09:15:00 | 80.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 14:15:00 | 84.87 | 2026-01-21 14:15:00 | 81.50 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2026-02-03 10:15:00 | 83.51 | 2026-02-03 11:15:00 | 84.29 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-02-13 15:15:00 | 81.42 | 2026-02-16 12:15:00 | 82.57 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-02-20 09:30:00 | 83.51 | 2026-02-20 12:15:00 | 83.14 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-02-20 10:45:00 | 83.22 | 2026-02-20 12:15:00 | 83.14 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2026-02-26 12:15:00 | 72.24 | 2026-02-27 10:15:00 | 73.04 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-02-27 09:45:00 | 71.97 | 2026-02-27 10:15:00 | 73.04 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-03-06 14:30:00 | 70.51 | 2026-03-09 09:15:00 | 66.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 70.51 | 2026-03-10 13:15:00 | 67.32 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2026-03-20 15:00:00 | 62.89 | 2026-03-25 10:15:00 | 62.97 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-03-23 09:15:00 | 61.45 | 2026-03-25 10:15:00 | 62.97 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-04-01 10:15:00 | 60.03 | 2026-04-01 12:15:00 | 61.12 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-01 14:30:00 | 60.08 | 2026-04-06 14:15:00 | 61.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-06 09:15:00 | 59.63 | 2026-04-06 14:15:00 | 61.10 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-04-15 09:15:00 | 66.78 | 2026-04-23 14:15:00 | 67.85 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2026-04-28 14:30:00 | 69.00 | 2026-05-05 09:15:00 | 69.09 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2026-04-28 15:15:00 | 68.95 | 2026-05-05 09:15:00 | 69.09 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-04-29 09:45:00 | 69.55 | 2026-05-05 09:15:00 | 69.09 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-05-06 10:15:00 | 69.31 | 2026-05-07 11:15:00 | 69.54 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-05-07 11:00:00 | 69.28 | 2026-05-07 11:15:00 | 69.54 | STOP_HIT | 1.00 | -0.38% |
