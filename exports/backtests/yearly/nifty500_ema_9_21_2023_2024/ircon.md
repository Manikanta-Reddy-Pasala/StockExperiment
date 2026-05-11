# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 158.99
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 199 |
| ALERT1 | 142 |
| ALERT2 | 139 |
| ALERT2_SKIP | 75 |
| ALERT3 | 357 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 157 |
| PARTIAL | 33 |
| TARGET_HIT | 26 |
| STOP_HIT | 133 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 192 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 97 / 95
- **Target hits / Stop hits / Partials:** 26 / 133 / 33
- **Avg / median % per leg:** 1.82% / 0.09%
- **Sum % (uncompounded):** 349.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 25 | 45.5% | 9 | 45 | 1 | 1.11% | 61.1% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.39% | 6.8% |
| BUY @ 3rd Alert (retest2) | 53 | 23 | 43.4% | 9 | 44 | 0 | 1.02% | 54.3% |
| SELL (all) | 137 | 72 | 52.6% | 17 | 88 | 32 | 2.11% | 288.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.42% | 8.8% |
| SELL @ 3rd Alert (retest2) | 135 | 70 | 51.9% | 17 | 87 | 31 | 2.07% | 280.1% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.91% | 15.6% |
| retest2 (combined) | 188 | 93 | 49.5% | 26 | 131 | 31 | 1.78% | 334.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 10:15:00 | 81.15 | 80.02 | 79.88 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 78.35 | 79.90 | 80.05 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 13:15:00 | 81.35 | 79.88 | 79.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 81.90 | 80.28 | 79.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 09:15:00 | 80.20 | 80.47 | 80.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 80.20 | 80.47 | 80.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 80.20 | 80.47 | 80.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 09:45:00 | 80.50 | 80.47 | 80.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 80.35 | 80.45 | 80.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:00:00 | 80.35 | 80.45 | 80.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 80.15 | 80.39 | 80.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 12:00:00 | 80.15 | 80.39 | 80.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 80.15 | 80.34 | 80.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 12:45:00 | 80.15 | 80.34 | 80.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 79.90 | 80.25 | 80.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 13:30:00 | 79.95 | 80.25 | 80.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 79.80 | 80.16 | 80.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 15:00:00 | 79.80 | 80.16 | 80.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 79.70 | 80.07 | 80.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 80.55 | 80.07 | 80.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 80.30 | 80.45 | 80.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:00:00 | 80.30 | 80.45 | 80.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 80.35 | 80.43 | 80.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:30:00 | 80.15 | 80.43 | 80.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 80.40 | 80.42 | 80.27 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 79.85 | 80.19 | 80.19 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 12:15:00 | 80.90 | 80.33 | 80.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 82.05 | 80.92 | 80.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 82.10 | 82.87 | 82.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 82.10 | 82.87 | 82.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 82.10 | 82.87 | 82.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:30:00 | 81.75 | 82.87 | 82.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 81.80 | 82.66 | 82.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 81.80 | 82.66 | 82.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 82.00 | 82.53 | 82.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:15:00 | 82.20 | 82.53 | 82.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 12:15:00 | 81.55 | 82.33 | 82.04 | SL hit (close<static) qty=1.00 sl=81.65 alert=retest2 |

### Cycle 6 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 79.85 | 81.73 | 81.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 09:15:00 | 79.10 | 80.96 | 81.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 81.35 | 80.16 | 80.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 81.35 | 80.16 | 80.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 81.35 | 80.16 | 80.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 81.35 | 80.16 | 80.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 81.90 | 80.51 | 80.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 81.10 | 80.51 | 80.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 83.20 | 81.40 | 81.22 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 14:15:00 | 80.80 | 81.66 | 81.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 09:15:00 | 80.30 | 81.25 | 81.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 09:15:00 | 82.45 | 80.84 | 81.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 09:15:00 | 82.45 | 80.84 | 81.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 82.45 | 80.84 | 81.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:00:00 | 82.45 | 80.84 | 81.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 81.60 | 80.99 | 81.10 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 11:15:00 | 82.25 | 81.24 | 81.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 14:15:00 | 82.55 | 81.72 | 81.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 84.45 | 85.62 | 84.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 84.45 | 85.62 | 84.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 84.45 | 85.62 | 84.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 83.70 | 85.62 | 84.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 84.30 | 85.35 | 84.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 84.35 | 85.35 | 84.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 84.25 | 85.13 | 84.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:00:00 | 84.25 | 85.13 | 84.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 84.25 | 84.96 | 84.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:45:00 | 83.85 | 84.96 | 84.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 84.10 | 84.78 | 84.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 83.25 | 84.78 | 84.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 83.40 | 84.33 | 84.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:00:00 | 83.40 | 84.33 | 84.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 83.35 | 84.09 | 84.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 13:15:00 | 82.90 | 83.86 | 84.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 84.00 | 83.73 | 83.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 84.00 | 83.73 | 83.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 84.00 | 83.73 | 83.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 84.40 | 83.73 | 83.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 83.90 | 83.76 | 83.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 12:15:00 | 83.65 | 83.78 | 83.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 14:00:00 | 83.60 | 83.74 | 83.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 09:45:00 | 83.40 | 83.78 | 83.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 15:15:00 | 83.25 | 82.67 | 82.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 15:15:00 | 83.25 | 82.67 | 82.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 85.80 | 83.30 | 82.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 82.45 | 83.75 | 83.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 82.45 | 83.75 | 83.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 82.45 | 83.75 | 83.45 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 12:15:00 | 82.65 | 83.22 | 83.26 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 86.30 | 83.82 | 83.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 89.55 | 85.47 | 84.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 14:15:00 | 86.05 | 86.89 | 85.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 14:15:00 | 86.05 | 86.89 | 85.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 86.05 | 86.89 | 85.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:30:00 | 85.75 | 86.89 | 85.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 85.55 | 86.62 | 85.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:15:00 | 85.35 | 86.62 | 85.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 84.55 | 86.20 | 85.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 84.55 | 86.20 | 85.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 85.05 | 85.97 | 85.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:15:00 | 84.70 | 85.97 | 85.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 13:15:00 | 84.10 | 85.03 | 85.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 82.95 | 84.44 | 84.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 83.15 | 83.08 | 83.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 14:00:00 | 83.15 | 83.08 | 83.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 83.25 | 83.11 | 83.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 83.25 | 83.11 | 83.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 84.30 | 83.31 | 83.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 84.35 | 83.31 | 83.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 84.15 | 83.48 | 83.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:30:00 | 84.15 | 83.48 | 83.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 84.05 | 83.78 | 83.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 84.25 | 83.87 | 83.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 12:15:00 | 83.95 | 84.08 | 83.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 12:15:00 | 83.95 | 84.08 | 83.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 12:15:00 | 83.95 | 84.08 | 83.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 12:45:00 | 83.90 | 84.08 | 83.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 83.70 | 84.00 | 83.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 13:30:00 | 83.75 | 84.00 | 83.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 83.80 | 83.96 | 83.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 83.80 | 83.96 | 83.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 83.70 | 83.89 | 83.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:30:00 | 83.95 | 83.89 | 83.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 83.65 | 83.84 | 83.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 12:15:00 | 83.45 | 83.74 | 83.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 83.65 | 83.63 | 83.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 83.65 | 83.63 | 83.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 83.65 | 83.63 | 83.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 15:15:00 | 83.35 | 83.68 | 83.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 09:30:00 | 83.10 | 83.50 | 83.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 10:45:00 | 83.20 | 83.46 | 83.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 11:30:00 | 83.35 | 83.47 | 83.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 83.05 | 83.36 | 83.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 13:30:00 | 83.45 | 83.36 | 83.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 83.35 | 83.36 | 83.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 15:00:00 | 83.35 | 83.36 | 83.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 84.05 | 83.44 | 83.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:00:00 | 84.05 | 83.44 | 83.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 83.45 | 83.44 | 83.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 11:45:00 | 83.30 | 83.42 | 83.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 12:15:00 | 83.25 | 83.42 | 83.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 10:15:00 | 84.10 | 83.60 | 83.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 84.10 | 83.60 | 83.54 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 14:15:00 | 83.30 | 83.48 | 83.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 09:15:00 | 83.10 | 83.36 | 83.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 82.85 | 82.50 | 82.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 82.85 | 82.50 | 82.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 82.85 | 82.50 | 82.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:45:00 | 82.90 | 82.50 | 82.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 82.90 | 82.58 | 82.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:00:00 | 82.90 | 82.58 | 82.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 82.55 | 82.57 | 82.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 13:15:00 | 82.35 | 82.54 | 82.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 14:15:00 | 82.25 | 82.53 | 82.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 09:15:00 | 82.00 | 82.48 | 82.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 11:30:00 | 82.25 | 82.39 | 82.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 82.90 | 82.28 | 82.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:45:00 | 83.15 | 82.28 | 82.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 83.20 | 82.46 | 82.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 11:00:00 | 83.20 | 82.46 | 82.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 82.40 | 82.50 | 82.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 13:15:00 | 82.25 | 82.50 | 82.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 10:15:00 | 83.10 | 81.00 | 81.05 | SL hit (close>static) qty=1.00 sl=82.80 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 83.95 | 81.59 | 81.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 87.80 | 83.73 | 82.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 14:15:00 | 86.20 | 86.53 | 85.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-19 15:00:00 | 86.20 | 86.53 | 85.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 92.60 | 93.52 | 91.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:15:00 | 91.75 | 93.52 | 91.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 91.75 | 93.17 | 91.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:15:00 | 90.45 | 93.17 | 91.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 91.15 | 92.76 | 91.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:30:00 | 91.60 | 92.76 | 91.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 91.40 | 92.49 | 91.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 93.10 | 91.60 | 91.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 12:15:00 | 92.35 | 94.18 | 94.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 12:15:00 | 92.35 | 94.18 | 94.22 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 94.85 | 94.16 | 94.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 12:15:00 | 95.05 | 94.34 | 94.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 97.05 | 97.77 | 96.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 97.05 | 97.77 | 96.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 97.05 | 97.77 | 96.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 96.95 | 97.77 | 96.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 95.50 | 97.32 | 96.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 95.50 | 97.32 | 96.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 94.90 | 96.83 | 96.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 94.90 | 96.83 | 96.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 93.60 | 95.87 | 96.02 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 96.95 | 96.09 | 95.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 98.70 | 96.73 | 96.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 99.95 | 100.35 | 98.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 09:30:00 | 99.90 | 100.35 | 98.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 99.85 | 100.15 | 99.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 99.60 | 100.15 | 99.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 99.80 | 100.03 | 99.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 105.90 | 100.03 | 99.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 14:15:00 | 103.05 | 104.09 | 104.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 103.05 | 104.09 | 104.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 101.90 | 103.56 | 103.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 102.90 | 102.72 | 103.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 14:15:00 | 102.90 | 102.72 | 103.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 102.90 | 102.72 | 103.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 15:00:00 | 102.90 | 102.72 | 103.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 103.05 | 102.79 | 103.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 102.35 | 102.79 | 103.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 11:30:00 | 101.60 | 102.78 | 103.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 12:15:00 | 102.10 | 102.78 | 103.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 12:15:00 | 103.40 | 102.90 | 103.17 | SL hit (close>static) qty=1.00 sl=103.35 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 105.05 | 103.56 | 103.42 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 101.25 | 103.07 | 103.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 100.85 | 102.41 | 102.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 104.80 | 101.88 | 102.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 104.80 | 101.88 | 102.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 104.80 | 101.88 | 102.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:45:00 | 104.70 | 101.88 | 102.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 106.25 | 102.75 | 102.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 12:15:00 | 107.95 | 104.37 | 103.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 109.60 | 109.86 | 108.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 11:00:00 | 109.60 | 109.86 | 108.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 110.25 | 110.75 | 109.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 111.25 | 110.75 | 109.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 110.35 | 110.67 | 109.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 108.90 | 110.67 | 109.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 108.65 | 110.27 | 109.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 108.50 | 110.27 | 109.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 108.35 | 109.88 | 109.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:00:00 | 108.35 | 109.88 | 109.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 107.85 | 109.48 | 109.52 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 110.90 | 109.44 | 109.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 112.80 | 110.30 | 109.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 12:15:00 | 113.05 | 113.68 | 112.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 12:15:00 | 113.05 | 113.68 | 112.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 113.05 | 113.68 | 112.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:30:00 | 112.90 | 113.68 | 112.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 113.80 | 113.70 | 113.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 10:30:00 | 115.20 | 114.02 | 113.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-04 12:15:00 | 126.72 | 122.42 | 118.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 14:15:00 | 121.60 | 122.87 | 122.99 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 14:15:00 | 123.15 | 122.96 | 122.96 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 15:15:00 | 122.80 | 122.93 | 122.95 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 09:15:00 | 123.85 | 123.11 | 123.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 129.60 | 124.43 | 123.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 11:15:00 | 153.15 | 153.41 | 145.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 12:00:00 | 153.15 | 153.41 | 145.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 140.50 | 149.79 | 146.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:00:00 | 140.50 | 149.79 | 146.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 143.55 | 148.54 | 146.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:30:00 | 139.10 | 148.54 | 146.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 146.80 | 147.81 | 146.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:30:00 | 146.40 | 147.81 | 146.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 147.25 | 147.70 | 146.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 14:30:00 | 148.20 | 147.40 | 146.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 15:15:00 | 147.95 | 147.40 | 146.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 12:15:00 | 145.35 | 147.56 | 147.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 12:15:00 | 145.35 | 147.56 | 147.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 144.15 | 146.36 | 146.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 14:15:00 | 140.70 | 136.98 | 138.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 14:15:00 | 140.70 | 136.98 | 138.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 140.70 | 136.98 | 138.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 140.70 | 136.98 | 138.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 140.90 | 137.76 | 138.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 140.35 | 137.76 | 138.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 10:15:00 | 146.10 | 140.05 | 139.61 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 140.75 | 141.64 | 141.74 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 11:15:00 | 142.25 | 141.79 | 141.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 14:15:00 | 144.80 | 142.52 | 142.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 09:15:00 | 143.10 | 144.80 | 143.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 143.10 | 144.80 | 143.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 143.10 | 144.80 | 143.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 143.10 | 144.80 | 143.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 142.90 | 144.42 | 143.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:15:00 | 142.70 | 144.42 | 143.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 141.30 | 143.45 | 143.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 137.10 | 140.96 | 141.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 137.25 | 135.99 | 138.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 10:00:00 | 137.25 | 135.99 | 138.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 137.50 | 136.29 | 138.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:30:00 | 138.15 | 136.29 | 138.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 137.70 | 136.57 | 138.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:15:00 | 137.05 | 136.57 | 138.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 138.90 | 137.39 | 137.99 | SL hit (close>static) qty=1.00 sl=138.25 alert=retest2 |

### Cycle 39 — BUY (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 09:15:00 | 141.75 | 137.36 | 137.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 10:15:00 | 145.30 | 138.95 | 138.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 156.20 | 157.39 | 152.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 11:00:00 | 156.20 | 157.39 | 152.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 155.90 | 156.62 | 154.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 160.75 | 155.19 | 154.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 148.85 | 154.37 | 154.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 148.85 | 154.37 | 154.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 146.80 | 151.39 | 153.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 139.00 | 135.40 | 139.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 14:15:00 | 139.00 | 135.40 | 139.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 139.00 | 135.40 | 139.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 15:00:00 | 139.00 | 135.40 | 139.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 139.45 | 136.21 | 139.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 141.40 | 136.21 | 139.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 141.95 | 137.36 | 139.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 09:15:00 | 137.90 | 139.85 | 140.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 12:15:00 | 140.30 | 139.96 | 139.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 140.30 | 139.96 | 139.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 13:15:00 | 140.85 | 140.14 | 140.02 | Break + close above crossover candle high |

### Cycle 42 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 139.00 | 139.91 | 139.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 137.35 | 138.88 | 139.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 139.90 | 138.62 | 139.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 139.90 | 138.62 | 139.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 139.90 | 138.62 | 139.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 11:00:00 | 139.35 | 138.76 | 139.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 141.25 | 139.19 | 139.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 141.25 | 139.19 | 139.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 10:15:00 | 143.05 | 139.96 | 139.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 12:15:00 | 152.40 | 153.74 | 151.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 13:00:00 | 152.40 | 153.74 | 151.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 152.80 | 153.45 | 152.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:30:00 | 154.40 | 153.45 | 152.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 153.60 | 153.49 | 152.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:45:00 | 152.35 | 153.49 | 152.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 152.45 | 153.19 | 152.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:15:00 | 152.25 | 153.19 | 152.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 153.20 | 153.19 | 152.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:30:00 | 152.55 | 153.19 | 152.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 154.50 | 153.54 | 152.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 10:15:00 | 156.70 | 153.54 | 152.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 15:15:00 | 154.75 | 155.99 | 156.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 15:15:00 | 154.75 | 155.99 | 156.13 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 10:15:00 | 160.40 | 156.99 | 156.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 12:15:00 | 166.40 | 159.20 | 157.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 13:15:00 | 163.80 | 164.27 | 161.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 13:45:00 | 163.65 | 164.27 | 161.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 164.85 | 165.77 | 164.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:00:00 | 164.85 | 165.77 | 164.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 162.35 | 165.09 | 164.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:00:00 | 162.35 | 165.09 | 164.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 162.90 | 164.65 | 164.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:30:00 | 164.20 | 164.59 | 164.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 15:15:00 | 165.95 | 167.14 | 167.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 15:15:00 | 165.95 | 167.14 | 167.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 09:15:00 | 162.25 | 166.16 | 166.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 14:15:00 | 164.95 | 164.40 | 165.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 14:15:00 | 164.95 | 164.40 | 165.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 164.95 | 164.40 | 165.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 15:00:00 | 164.95 | 164.40 | 165.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 15:15:00 | 167.90 | 165.10 | 165.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 09:15:00 | 172.95 | 165.10 | 165.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 171.35 | 166.35 | 166.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 177.30 | 171.88 | 169.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 172.55 | 174.10 | 172.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 172.55 | 174.10 | 172.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 172.55 | 174.10 | 172.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 172.65 | 174.10 | 172.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 172.85 | 173.85 | 172.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 14:45:00 | 173.55 | 173.49 | 172.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 09:15:00 | 171.10 | 172.94 | 172.41 | SL hit (close<static) qty=1.00 sl=172.25 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 09:15:00 | 159.20 | 169.63 | 171.04 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 169.45 | 163.84 | 163.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 171.25 | 165.32 | 163.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 14:15:00 | 176.70 | 176.85 | 174.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 15:00:00 | 176.70 | 176.85 | 174.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 174.00 | 176.01 | 175.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:45:00 | 174.40 | 176.01 | 175.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 172.40 | 175.28 | 174.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 172.40 | 175.28 | 174.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 165.25 | 173.28 | 173.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 162.60 | 171.14 | 172.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 167.25 | 166.62 | 169.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 167.25 | 166.62 | 169.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 172.05 | 167.55 | 168.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:45:00 | 172.90 | 167.55 | 168.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 171.95 | 168.43 | 168.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:30:00 | 172.05 | 168.43 | 168.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 11:15:00 | 171.80 | 169.10 | 168.79 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 09:15:00 | 168.65 | 169.44 | 169.46 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 12:15:00 | 170.40 | 169.48 | 169.46 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 15:15:00 | 168.90 | 169.43 | 169.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 09:15:00 | 168.40 | 169.23 | 169.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 11:15:00 | 170.00 | 169.24 | 169.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 11:15:00 | 170.00 | 169.24 | 169.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 170.00 | 169.24 | 169.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 12:00:00 | 170.00 | 169.24 | 169.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 169.80 | 169.35 | 169.38 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 13:15:00 | 171.70 | 169.82 | 169.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 174.40 | 171.14 | 170.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 172.30 | 173.23 | 172.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 172.30 | 173.23 | 172.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 172.30 | 173.23 | 172.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 171.75 | 173.23 | 172.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 172.85 | 173.15 | 172.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 12:45:00 | 173.90 | 173.18 | 172.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 13:30:00 | 173.90 | 173.34 | 172.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:00:00 | 174.00 | 173.34 | 172.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:45:00 | 173.90 | 173.38 | 172.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 180.05 | 174.91 | 173.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 10:45:00 | 181.25 | 176.31 | 174.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 12:00:00 | 181.10 | 183.29 | 183.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 13:15:00 | 181.45 | 182.58 | 182.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 13:15:00 | 181.45 | 182.58 | 182.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 179.65 | 181.99 | 182.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 11:15:00 | 182.90 | 181.40 | 181.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 11:15:00 | 182.90 | 181.40 | 181.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 182.90 | 181.40 | 181.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 11:30:00 | 184.00 | 181.40 | 181.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 183.20 | 181.76 | 182.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:30:00 | 183.15 | 181.76 | 182.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 180.50 | 181.40 | 181.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:15:00 | 182.10 | 181.40 | 181.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 181.00 | 181.32 | 181.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 11:45:00 | 180.25 | 181.08 | 181.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 13:15:00 | 189.70 | 182.59 | 182.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 13:15:00 | 189.70 | 182.59 | 182.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 14:15:00 | 191.80 | 184.43 | 183.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 15:15:00 | 194.60 | 194.67 | 192.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 09:15:00 | 198.50 | 194.67 | 192.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 10:15:00 | 208.43 | 198.93 | 194.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-15 14:15:00 | 202.05 | 202.35 | 197.92 | SL hit (close<ema200) qty=0.50 sl=202.35 alert=retest1 |

### Cycle 58 — SELL (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 13:15:00 | 202.50 | 204.28 | 204.31 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 211.10 | 205.47 | 204.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 10:15:00 | 214.40 | 207.25 | 205.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 240.55 | 250.90 | 236.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 240.55 | 250.90 | 236.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 240.55 | 250.90 | 236.81 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 10:15:00 | 234.70 | 242.26 | 242.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 224.25 | 228.37 | 231.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 13:15:00 | 231.60 | 226.33 | 228.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 13:15:00 | 231.60 | 226.33 | 228.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 231.60 | 226.33 | 228.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:00:00 | 231.60 | 226.33 | 228.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 231.80 | 227.42 | 228.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:30:00 | 231.20 | 227.42 | 228.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 230.15 | 229.42 | 229.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:00:00 | 230.15 | 229.42 | 229.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 12:15:00 | 230.75 | 229.68 | 229.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 13:15:00 | 232.45 | 230.24 | 229.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 14:15:00 | 230.10 | 230.21 | 229.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-07 15:00:00 | 230.10 | 230.21 | 229.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 15:15:00 | 229.85 | 230.14 | 229.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:15:00 | 230.40 | 230.14 | 229.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 228.50 | 229.81 | 229.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 10:15:00 | 232.45 | 229.81 | 229.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 15:15:00 | 228.70 | 229.70 | 229.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-02-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 15:15:00 | 228.70 | 229.70 | 229.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 225.35 | 228.83 | 229.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 209.20 | 206.44 | 214.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:00:00 | 209.20 | 206.44 | 214.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 210.70 | 207.64 | 211.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 210.70 | 207.64 | 211.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 212.00 | 208.52 | 211.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 209.15 | 208.52 | 211.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 09:15:00 | 215.00 | 209.81 | 212.11 | SL hit (close>static) qty=1.00 sl=212.50 alert=retest2 |

### Cycle 63 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 220.30 | 214.18 | 213.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 221.90 | 215.72 | 214.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 226.10 | 226.59 | 222.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 11:45:00 | 226.25 | 226.59 | 222.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 229.95 | 231.45 | 228.92 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 228.60 | 229.35 | 229.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 223.40 | 228.16 | 228.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 225.65 | 225.62 | 227.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 15:00:00 | 225.65 | 225.62 | 227.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 230.80 | 226.64 | 227.31 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 230.25 | 228.06 | 227.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 233.40 | 229.61 | 228.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 14:15:00 | 231.75 | 231.96 | 230.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-26 15:00:00 | 231.75 | 231.96 | 230.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 231.90 | 233.21 | 232.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:15:00 | 230.85 | 233.21 | 232.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 230.85 | 232.74 | 231.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 229.35 | 232.74 | 231.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 226.05 | 230.82 | 231.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 220.65 | 227.89 | 229.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 226.90 | 224.60 | 226.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 12:15:00 | 226.90 | 224.60 | 226.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 226.90 | 224.60 | 226.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:45:00 | 226.10 | 224.60 | 226.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 226.15 | 224.91 | 226.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:30:00 | 225.40 | 224.91 | 226.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 229.35 | 225.80 | 226.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 229.35 | 225.80 | 226.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 229.50 | 226.54 | 227.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 230.00 | 226.54 | 227.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 228.30 | 227.57 | 227.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 229.15 | 227.89 | 227.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 226.55 | 227.63 | 227.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 226.55 | 227.63 | 227.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 226.55 | 227.63 | 227.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:45:00 | 224.95 | 227.63 | 227.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 10:15:00 | 225.45 | 227.19 | 227.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 224.70 | 225.76 | 226.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 225.70 | 221.57 | 222.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 225.70 | 221.57 | 222.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 225.70 | 221.57 | 222.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 225.70 | 221.57 | 222.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 224.10 | 222.08 | 222.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:15:00 | 223.35 | 222.43 | 223.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:45:00 | 223.50 | 222.61 | 223.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 13:45:00 | 223.35 | 222.63 | 223.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 15:15:00 | 223.40 | 222.86 | 223.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 223.40 | 222.96 | 223.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:15:00 | 222.80 | 222.96 | 223.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 220.20 | 222.41 | 222.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 10:45:00 | 219.70 | 221.83 | 222.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 14:15:00 | 219.90 | 220.65 | 221.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 212.18 | 218.12 | 220.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 212.32 | 218.12 | 220.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 212.18 | 218.12 | 220.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 212.23 | 218.12 | 220.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 208.71 | 218.12 | 220.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 208.91 | 218.12 | 220.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 201.01 | 208.54 | 213.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 69 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 209.50 | 204.51 | 204.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 212.15 | 206.04 | 204.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 215.55 | 215.60 | 212.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 209.50 | 213.98 | 213.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 209.50 | 213.98 | 213.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 209.50 | 213.98 | 213.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 213.25 | 213.83 | 213.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 11:30:00 | 214.00 | 213.53 | 213.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 13:00:00 | 213.75 | 213.57 | 213.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 14:15:00 | 210.30 | 212.61 | 212.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 210.30 | 212.61 | 212.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 15:15:00 | 209.95 | 212.08 | 212.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 215.10 | 212.68 | 212.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 215.10 | 212.68 | 212.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 215.10 | 212.68 | 212.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:45:00 | 214.95 | 212.68 | 212.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 216.35 | 213.42 | 213.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 219.75 | 216.23 | 214.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 218.55 | 218.85 | 216.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 218.55 | 218.85 | 216.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 227.50 | 220.68 | 218.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 218.50 | 220.68 | 218.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 223.20 | 224.74 | 222.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 223.20 | 224.74 | 222.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 222.50 | 224.29 | 222.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 221.90 | 224.29 | 222.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 221.65 | 223.76 | 222.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:30:00 | 224.90 | 223.40 | 222.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 220.30 | 222.41 | 222.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 220.30 | 222.41 | 222.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 15:15:00 | 218.75 | 221.68 | 222.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 223.85 | 222.11 | 222.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 223.85 | 222.11 | 222.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 223.85 | 222.11 | 222.25 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 224.50 | 222.59 | 222.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 226.50 | 223.66 | 222.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 232.35 | 232.37 | 230.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 14:45:00 | 232.05 | 232.37 | 230.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 230.35 | 231.84 | 230.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 230.00 | 231.84 | 230.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 229.25 | 231.32 | 230.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 229.25 | 231.32 | 230.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 230.30 | 231.12 | 230.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 13:30:00 | 230.70 | 231.04 | 230.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 10:00:00 | 231.65 | 231.20 | 230.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 12:15:00 | 230.55 | 232.41 | 232.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 12:15:00 | 230.55 | 232.41 | 232.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 228.50 | 231.63 | 232.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 230.20 | 229.15 | 230.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 230.20 | 229.15 | 230.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 230.20 | 229.15 | 230.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:00:00 | 226.25 | 228.57 | 229.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 214.94 | 227.04 | 228.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 15:15:00 | 220.90 | 220.89 | 222.93 | SL hit (close>ema200) qty=0.50 sl=220.89 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 222.05 | 220.95 | 220.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 223.35 | 221.57 | 221.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 09:15:00 | 250.60 | 253.04 | 248.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 09:45:00 | 250.85 | 253.04 | 248.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 248.95 | 252.22 | 248.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 249.75 | 252.22 | 248.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 249.55 | 251.69 | 248.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 12:15:00 | 250.20 | 251.69 | 248.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 245.90 | 249.65 | 249.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 245.90 | 249.65 | 249.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 12:15:00 | 245.25 | 248.77 | 249.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 249.50 | 248.34 | 249.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 14:15:00 | 249.50 | 248.34 | 249.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 249.50 | 248.34 | 249.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 14:45:00 | 249.60 | 248.34 | 249.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 249.15 | 248.50 | 249.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 246.00 | 248.50 | 249.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 244.55 | 247.71 | 248.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 242.30 | 246.64 | 247.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 10:15:00 | 241.90 | 246.05 | 247.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 09:15:00 | 230.19 | 237.59 | 241.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 09:15:00 | 238.00 | 237.59 | 241.44 | SL hit (close>static) qty=0.50 sl=237.59 alert=retest2 |

### Cycle 77 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 241.90 | 231.43 | 230.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 252.45 | 243.35 | 239.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 280.75 | 285.42 | 276.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 280.75 | 285.42 | 276.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 280.75 | 285.42 | 276.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 280.20 | 285.42 | 276.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 278.60 | 281.56 | 278.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 287.80 | 281.56 | 278.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 15:15:00 | 281.50 | 281.76 | 279.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 09:15:00 | 274.55 | 280.28 | 279.57 | SL hit (close<static) qty=1.00 sl=277.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 273.90 | 279.00 | 279.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 272.00 | 275.90 | 277.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 275.50 | 275.18 | 276.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 11:15:00 | 276.70 | 275.18 | 276.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 275.75 | 275.29 | 276.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 274.70 | 275.29 | 276.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 277.75 | 275.85 | 276.60 | SL hit (close>static) qty=1.00 sl=276.75 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 288.45 | 273.86 | 272.33 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 250.90 | 275.48 | 275.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 232.80 | 266.94 | 271.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 260.30 | 244.64 | 251.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 260.30 | 244.64 | 251.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 260.30 | 244.64 | 251.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 260.30 | 244.64 | 251.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 262.20 | 248.15 | 252.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 256.95 | 249.37 | 252.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 12:15:00 | 252.90 | 251.46 | 251.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 12:15:00 | 252.90 | 251.46 | 251.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 263.40 | 254.16 | 252.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 269.55 | 269.58 | 265.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 269.30 | 269.58 | 265.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 268.30 | 268.96 | 265.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 268.95 | 268.96 | 265.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 268.55 | 268.70 | 266.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 268.70 | 268.53 | 266.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 270.70 | 268.67 | 266.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 267.35 | 272.27 | 270.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 267.75 | 272.27 | 270.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 268.00 | 271.41 | 270.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 266.90 | 271.41 | 270.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 269.20 | 270.12 | 270.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 269.20 | 270.12 | 270.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 265.80 | 268.60 | 269.40 | Break + close below crossover candle low |

### Cycle 83 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 283.65 | 270.61 | 269.71 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 272.25 | 275.54 | 275.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 270.30 | 272.59 | 273.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 274.00 | 272.08 | 273.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 274.00 | 272.08 | 273.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 274.00 | 272.08 | 273.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 274.00 | 272.08 | 273.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 272.35 | 272.13 | 273.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:15:00 | 271.30 | 272.13 | 273.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 13:30:00 | 271.40 | 270.13 | 270.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 271.15 | 270.73 | 270.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:45:00 | 271.50 | 270.73 | 270.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 271.10 | 270.80 | 270.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:00:00 | 271.10 | 270.80 | 270.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 271.00 | 270.84 | 270.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 271.00 | 270.84 | 270.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-01 15:15:00 | 271.70 | 271.01 | 271.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 271.70 | 271.01 | 271.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 272.55 | 271.32 | 271.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 272.30 | 272.68 | 271.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 272.30 | 272.68 | 271.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 272.30 | 272.68 | 271.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 272.30 | 272.68 | 271.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 273.90 | 272.93 | 272.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 275.55 | 273.42 | 272.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:30:00 | 275.25 | 274.66 | 273.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 14:45:00 | 275.55 | 275.28 | 274.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 13:15:00 | 303.11 | 292.10 | 284.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 323.50 | 329.60 | 330.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 315.75 | 325.11 | 327.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 12:15:00 | 316.55 | 314.46 | 318.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 13:00:00 | 316.55 | 314.46 | 318.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 321.90 | 315.95 | 319.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 326.90 | 315.95 | 319.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 315.00 | 315.76 | 318.70 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 323.20 | 319.54 | 319.11 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 315.05 | 318.64 | 318.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 298.30 | 314.57 | 316.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 286.70 | 279.31 | 284.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 286.70 | 279.31 | 284.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 286.70 | 279.31 | 284.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:45:00 | 286.95 | 279.31 | 284.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 286.80 | 280.80 | 284.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:15:00 | 287.95 | 280.80 | 284.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 285.70 | 281.78 | 284.96 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 14:15:00 | 298.55 | 287.46 | 286.97 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 288.90 | 290.38 | 290.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 288.20 | 289.70 | 290.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 13:15:00 | 284.70 | 284.68 | 286.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 13:15:00 | 284.70 | 284.68 | 286.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 284.70 | 284.68 | 286.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:45:00 | 285.50 | 284.68 | 286.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 285.75 | 284.89 | 286.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 15:00:00 | 285.75 | 284.89 | 286.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 285.50 | 285.01 | 286.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 268.35 | 285.01 | 286.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 278.80 | 273.28 | 272.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 278.80 | 273.28 | 272.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 282.90 | 275.96 | 274.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 270.25 | 276.19 | 274.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 14:15:00 | 270.25 | 276.19 | 274.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 270.25 | 276.19 | 274.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 271.00 | 276.19 | 274.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 269.60 | 274.87 | 274.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 269.45 | 274.87 | 274.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 09:15:00 | 266.15 | 273.13 | 273.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 11:15:00 | 264.90 | 270.28 | 272.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 273.25 | 268.55 | 270.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 10:15:00 | 273.25 | 268.55 | 270.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 273.25 | 268.55 | 270.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:45:00 | 274.20 | 268.55 | 270.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 271.00 | 269.04 | 270.21 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 272.75 | 271.01 | 270.92 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 266.05 | 270.37 | 270.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 262.10 | 268.63 | 269.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 265.65 | 264.39 | 266.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 265.65 | 264.39 | 266.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 265.65 | 264.39 | 266.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 268.70 | 264.39 | 266.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 266.25 | 265.00 | 266.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 265.50 | 265.10 | 266.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 268.00 | 265.68 | 266.54 | SL hit (close>static) qty=1.00 sl=266.75 alert=retest2 |

### Cycle 95 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 271.25 | 267.03 | 266.96 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 266.50 | 267.60 | 267.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 14:15:00 | 266.25 | 267.18 | 267.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 266.30 | 265.42 | 266.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 266.30 | 265.42 | 266.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 266.30 | 265.42 | 266.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 14:00:00 | 265.45 | 265.64 | 266.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 14:45:00 | 265.20 | 265.68 | 266.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 15:15:00 | 265.20 | 265.68 | 266.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 270.40 | 266.55 | 266.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 09:15:00 | 270.40 | 266.55 | 266.34 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 265.10 | 266.27 | 266.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 264.05 | 265.25 | 265.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 11:15:00 | 264.30 | 264.29 | 265.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 11:15:00 | 264.30 | 264.29 | 265.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 264.30 | 264.29 | 265.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:30:00 | 262.80 | 263.91 | 264.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:00:00 | 263.15 | 263.56 | 264.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:45:00 | 262.85 | 263.39 | 263.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 262.50 | 263.35 | 263.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 261.85 | 263.05 | 263.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 261.00 | 262.57 | 263.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:30:00 | 260.50 | 261.36 | 262.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 249.66 | 252.41 | 254.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 249.99 | 252.41 | 254.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 249.71 | 252.41 | 254.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 249.38 | 252.41 | 254.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 247.95 | 252.41 | 254.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:15:00 | 247.47 | 252.41 | 254.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 242.80 | 242.26 | 245.56 | SL hit (close>ema200) qty=0.50 sl=242.26 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 245.65 | 241.60 | 241.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 247.20 | 243.36 | 242.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 241.20 | 243.60 | 242.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 241.20 | 243.60 | 242.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 241.20 | 243.60 | 242.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 241.20 | 243.60 | 242.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 241.35 | 243.15 | 242.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:45:00 | 241.20 | 243.15 | 242.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 240.80 | 242.31 | 242.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 239.85 | 241.58 | 242.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 224.40 | 224.28 | 228.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 226.00 | 224.28 | 228.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 227.15 | 224.85 | 228.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 227.00 | 224.85 | 228.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 233.50 | 226.58 | 228.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 232.30 | 226.58 | 228.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 237.40 | 228.75 | 229.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 236.95 | 228.75 | 229.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 232.20 | 230.22 | 230.00 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 227.05 | 229.83 | 230.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 226.50 | 228.41 | 229.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 13:15:00 | 227.90 | 227.03 | 228.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 13:15:00 | 227.90 | 227.03 | 228.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 227.90 | 227.03 | 228.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 227.90 | 227.03 | 228.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 226.90 | 226.25 | 227.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:45:00 | 227.10 | 226.25 | 227.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 225.45 | 226.09 | 227.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:45:00 | 224.80 | 225.87 | 226.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:15:00 | 224.75 | 225.87 | 226.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 224.60 | 225.40 | 226.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 15:15:00 | 227.50 | 226.91 | 226.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 227.50 | 226.91 | 226.85 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 225.05 | 226.54 | 226.69 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 14:15:00 | 228.65 | 226.77 | 226.70 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 225.65 | 226.54 | 226.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 225.05 | 226.08 | 226.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 210.90 | 209.67 | 213.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 212.87 | 209.67 | 213.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 216.26 | 210.99 | 214.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 216.26 | 210.99 | 214.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 214.71 | 211.73 | 214.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 217.34 | 211.73 | 214.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 216.25 | 212.64 | 214.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 216.25 | 212.64 | 214.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 216.66 | 213.44 | 214.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 216.66 | 213.44 | 214.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 219.50 | 215.38 | 215.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 223.25 | 216.95 | 215.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 220.87 | 221.41 | 219.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:00:00 | 220.87 | 221.41 | 219.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 228.17 | 222.32 | 220.67 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 222.25 | 222.74 | 222.80 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 10:15:00 | 225.18 | 223.08 | 222.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-17 09:15:00 | 227.24 | 224.18 | 223.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 13:15:00 | 223.65 | 224.60 | 224.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 13:15:00 | 223.65 | 224.60 | 224.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 223.65 | 224.60 | 224.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:45:00 | 223.78 | 224.60 | 224.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 223.39 | 224.35 | 223.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 223.39 | 224.35 | 223.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 224.36 | 224.36 | 223.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 217.40 | 224.36 | 223.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 219.28 | 223.34 | 223.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 215.87 | 218.91 | 220.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 11:15:00 | 203.56 | 202.86 | 205.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 11:45:00 | 203.67 | 202.86 | 205.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 201.45 | 196.81 | 199.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 201.45 | 196.81 | 199.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 201.49 | 197.74 | 199.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 199.81 | 200.06 | 200.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 201.31 | 200.31 | 200.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 201.31 | 200.31 | 200.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 202.90 | 201.09 | 200.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 212.36 | 216.14 | 213.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 212.36 | 216.14 | 213.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 212.36 | 216.14 | 213.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 212.36 | 216.14 | 213.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 212.15 | 215.34 | 213.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 212.15 | 215.34 | 213.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 213.68 | 215.01 | 213.07 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 210.69 | 212.29 | 212.33 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 216.70 | 212.28 | 211.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 218.96 | 215.81 | 214.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 216.84 | 216.87 | 215.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 216.84 | 216.87 | 215.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 215.16 | 216.45 | 215.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 215.16 | 216.45 | 215.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 214.96 | 216.15 | 215.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 205.50 | 216.15 | 215.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 205.23 | 213.97 | 214.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 200.96 | 206.83 | 210.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 202.68 | 202.44 | 205.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 12:45:00 | 200.50 | 201.87 | 204.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 190.47 | 193.26 | 195.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 192.80 | 189.86 | 192.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 192.80 | 189.86 | 192.44 | SL hit (close>ema200) qty=0.50 sl=189.86 alert=retest1 |

### Cycle 115 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 197.20 | 188.87 | 188.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 204.48 | 198.17 | 195.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 208.02 | 210.23 | 206.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 208.02 | 210.23 | 206.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 211.67 | 210.07 | 208.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:00:00 | 216.09 | 211.42 | 209.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 223.00 | 223.65 | 223.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 223.00 | 223.65 | 223.67 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 227.95 | 224.51 | 224.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 233.28 | 226.26 | 224.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 231.35 | 232.00 | 229.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 12:00:00 | 231.35 | 232.00 | 229.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 224.93 | 230.82 | 229.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 225.55 | 230.82 | 229.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 225.37 | 229.73 | 229.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 224.68 | 229.73 | 229.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 11:15:00 | 226.47 | 229.08 | 229.22 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 230.45 | 228.77 | 228.63 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 227.85 | 228.79 | 228.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 224.71 | 227.97 | 228.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 14:15:00 | 218.31 | 217.46 | 219.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 15:00:00 | 218.31 | 217.46 | 219.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 218.50 | 217.67 | 219.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 215.24 | 217.67 | 219.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 213.77 | 216.89 | 219.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 211.25 | 214.94 | 217.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:45:00 | 211.72 | 213.15 | 215.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 210.80 | 211.54 | 213.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 211.15 | 209.71 | 210.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 218.16 | 211.40 | 210.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 218.16 | 211.40 | 210.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 219.18 | 215.20 | 213.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 216.26 | 217.10 | 215.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:30:00 | 216.90 | 217.10 | 215.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 217.77 | 218.61 | 217.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 217.77 | 218.61 | 217.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 216.40 | 218.17 | 217.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 216.40 | 218.17 | 217.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 216.33 | 217.80 | 217.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 212.63 | 217.80 | 217.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 213.47 | 216.94 | 216.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 208.10 | 215.17 | 216.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 207.92 | 207.84 | 210.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:00:00 | 207.92 | 207.84 | 210.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 207.45 | 207.50 | 209.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:00:00 | 206.14 | 207.15 | 208.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 205.65 | 206.81 | 208.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:30:00 | 205.67 | 207.13 | 207.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 203.40 | 207.13 | 207.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 195.83 | 204.70 | 206.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:15:00 | 195.37 | 203.04 | 205.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:15:00 | 195.39 | 203.04 | 205.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 193.23 | 198.89 | 202.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 185.53 | 191.50 | 197.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 202.83 | 192.69 | 191.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 205.56 | 195.26 | 193.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 14:15:00 | 219.22 | 219.23 | 214.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 15:00:00 | 219.22 | 219.23 | 214.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 215.00 | 218.18 | 214.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 215.00 | 218.18 | 214.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 213.49 | 217.24 | 214.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 213.51 | 217.24 | 214.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 214.31 | 216.66 | 214.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 214.54 | 216.66 | 214.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 214.01 | 216.13 | 214.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 213.79 | 216.13 | 214.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 214.60 | 215.82 | 214.59 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 207.31 | 213.32 | 213.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 204.62 | 210.47 | 212.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 210.55 | 209.00 | 211.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 210.55 | 209.00 | 211.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 210.55 | 209.00 | 211.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:45:00 | 210.74 | 209.00 | 211.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 212.81 | 209.76 | 211.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 212.81 | 209.76 | 211.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 213.00 | 210.41 | 211.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 213.57 | 210.41 | 211.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 210.88 | 210.68 | 211.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 208.66 | 210.55 | 211.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 209.14 | 210.09 | 210.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 198.23 | 205.12 | 207.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 198.68 | 205.12 | 207.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 10:15:00 | 187.79 | 195.33 | 200.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 204.19 | 197.57 | 197.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 209.80 | 202.73 | 200.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 203.99 | 214.67 | 210.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 203.99 | 214.67 | 210.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 203.99 | 214.67 | 210.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 203.99 | 214.67 | 210.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 205.15 | 212.77 | 209.90 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 191.27 | 204.96 | 206.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 11:15:00 | 189.15 | 191.08 | 192.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 189.11 | 188.22 | 189.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 189.11 | 188.22 | 189.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 182.64 | 187.23 | 189.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 14:15:00 | 180.55 | 184.35 | 187.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:00:00 | 181.35 | 183.75 | 186.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-12 09:15:00 | 162.50 | 178.96 | 183.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 160.97 | 156.75 | 156.71 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 155.32 | 158.57 | 159.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 150.16 | 153.60 | 155.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 140.07 | 139.75 | 143.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 140.07 | 139.75 | 143.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 145.05 | 141.13 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 145.82 | 141.13 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 144.30 | 141.76 | 143.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 143.41 | 141.87 | 143.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 143.26 | 142.35 | 143.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 147.24 | 143.47 | 143.56 | SL hit (close>static) qty=1.00 sl=145.95 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 147.35 | 144.25 | 143.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 149.41 | 147.10 | 145.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 148.44 | 151.05 | 149.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 148.44 | 151.05 | 149.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 148.44 | 151.05 | 149.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 148.44 | 151.05 | 149.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 148.25 | 150.49 | 149.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:30:00 | 148.86 | 149.98 | 149.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 148.21 | 149.17 | 149.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 148.21 | 149.17 | 149.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 147.10 | 148.76 | 149.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 146.00 | 145.50 | 146.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 145.65 | 145.50 | 146.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 144.18 | 145.24 | 146.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 143.19 | 145.04 | 146.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 142.93 | 143.48 | 144.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 147.81 | 141.14 | 141.90 | SL hit (close>static) qty=1.00 sl=146.90 alert=retest2 |

### Cycle 131 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 145.70 | 142.83 | 142.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 147.67 | 145.07 | 143.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 160.41 | 160.41 | 156.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:30:00 | 159.72 | 160.41 | 156.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 164.30 | 165.01 | 162.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 162.50 | 165.01 | 162.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 163.49 | 164.45 | 163.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 163.96 | 164.45 | 163.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 162.37 | 164.03 | 163.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 161.90 | 164.03 | 163.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 161.20 | 163.46 | 162.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 161.20 | 163.46 | 162.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 161.15 | 163.00 | 162.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:45:00 | 160.51 | 163.00 | 162.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 160.40 | 162.48 | 162.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 160.06 | 162.00 | 162.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 159.47 | 159.32 | 160.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 159.47 | 159.32 | 160.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 159.47 | 159.32 | 160.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 160.25 | 159.32 | 160.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 159.98 | 159.45 | 160.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 161.89 | 159.45 | 160.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 160.82 | 159.72 | 160.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 159.86 | 159.78 | 160.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 160.82 | 159.06 | 158.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 160.82 | 159.06 | 158.95 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 154.91 | 159.25 | 159.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 154.72 | 157.77 | 158.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 146.18 | 145.85 | 150.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 149.45 | 145.85 | 150.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 147.19 | 146.12 | 149.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 146.35 | 146.35 | 149.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:30:00 | 146.56 | 147.39 | 148.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 146.47 | 146.61 | 148.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 14:15:00 | 149.72 | 148.31 | 148.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 149.72 | 148.31 | 148.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 156.58 | 150.15 | 149.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 162.79 | 163.32 | 161.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 162.79 | 163.32 | 161.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 160.01 | 162.61 | 161.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 160.01 | 162.61 | 161.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 160.73 | 162.23 | 161.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 159.79 | 162.23 | 161.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 162.45 | 162.13 | 161.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:15:00 | 162.85 | 162.13 | 161.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 158.10 | 163.21 | 163.18 | SL hit (close<static) qty=1.00 sl=161.70 alert=retest2 |

### Cycle 136 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 156.64 | 161.90 | 162.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 155.65 | 158.27 | 160.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 15:15:00 | 157.32 | 157.28 | 158.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 09:15:00 | 159.44 | 157.28 | 158.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 157.92 | 157.41 | 158.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 157.31 | 157.41 | 158.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 157.23 | 157.46 | 158.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 157.04 | 157.46 | 158.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:45:00 | 157.24 | 157.40 | 158.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 156.06 | 154.82 | 156.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 156.06 | 154.82 | 156.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 156.58 | 155.17 | 156.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 155.61 | 155.17 | 156.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 157.95 | 156.05 | 155.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 157.95 | 156.05 | 155.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 158.47 | 156.84 | 156.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 157.82 | 158.04 | 157.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:30:00 | 157.60 | 158.04 | 157.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 156.80 | 157.80 | 157.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 156.80 | 157.80 | 157.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 156.70 | 157.58 | 157.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:15:00 | 155.79 | 157.58 | 157.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 156.26 | 157.31 | 157.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 156.26 | 157.31 | 157.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 155.58 | 156.97 | 156.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 153.51 | 156.97 | 156.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 151.98 | 155.97 | 156.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 149.03 | 153.52 | 154.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 156.44 | 150.19 | 151.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 156.44 | 150.19 | 151.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 156.44 | 150.19 | 151.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 157.68 | 150.19 | 151.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 157.90 | 152.98 | 152.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 159.36 | 155.17 | 153.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 189.70 | 193.60 | 187.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:45:00 | 189.29 | 193.60 | 187.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 188.00 | 190.54 | 188.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 187.52 | 190.54 | 188.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 188.62 | 190.15 | 188.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 190.98 | 190.15 | 188.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 186.96 | 189.58 | 188.61 | SL hit (close<static) qty=1.00 sl=187.33 alert=retest2 |

### Cycle 140 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 185.55 | 188.10 | 188.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 185.24 | 187.53 | 188.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 13:15:00 | 185.23 | 185.15 | 186.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:00:00 | 185.23 | 185.15 | 186.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 187.91 | 185.35 | 185.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 189.90 | 185.35 | 185.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 187.60 | 185.80 | 186.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 186.87 | 186.06 | 186.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 187.31 | 186.44 | 186.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 187.31 | 186.44 | 186.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 188.20 | 186.97 | 186.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 186.20 | 186.82 | 186.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 186.20 | 186.82 | 186.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 186.20 | 186.82 | 186.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 185.35 | 186.82 | 186.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 187.92 | 187.04 | 186.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 189.21 | 187.04 | 186.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 188.46 | 188.07 | 187.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 11:15:00 | 208.13 | 199.60 | 195.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 215.00 | 216.07 | 216.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 212.99 | 215.45 | 215.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 203.98 | 202.75 | 205.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 203.98 | 202.75 | 205.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 205.50 | 203.30 | 205.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 205.50 | 203.30 | 205.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 205.42 | 203.73 | 205.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 203.42 | 203.73 | 205.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 204.20 | 203.78 | 205.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 193.25 | 196.70 | 199.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 193.99 | 196.70 | 199.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 194.85 | 193.40 | 196.21 | SL hit (close>ema200) qty=0.50 sl=193.40 alert=retest2 |

### Cycle 143 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 201.68 | 197.20 | 196.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 204.50 | 203.00 | 201.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 203.05 | 203.33 | 202.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 203.05 | 203.33 | 202.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 202.69 | 203.51 | 203.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 202.69 | 203.51 | 203.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 202.59 | 203.32 | 202.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 202.45 | 203.32 | 202.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 202.91 | 203.19 | 202.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 201.76 | 202.91 | 202.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 201.80 | 202.69 | 202.76 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 203.71 | 202.80 | 202.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 15:15:00 | 204.00 | 203.04 | 202.88 | Break + close above crossover candle high |

### Cycle 146 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 201.03 | 202.64 | 202.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 200.10 | 201.21 | 201.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 199.70 | 198.96 | 199.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 199.70 | 198.96 | 199.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 199.70 | 198.96 | 199.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 199.64 | 198.96 | 199.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 197.79 | 198.80 | 199.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:00:00 | 197.53 | 198.55 | 199.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 09:15:00 | 187.65 | 189.73 | 190.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 191.05 | 188.64 | 189.40 | SL hit (close>ema200) qty=0.50 sl=188.64 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 190.86 | 189.99 | 189.91 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 188.90 | 189.82 | 189.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 188.59 | 189.24 | 189.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 188.40 | 187.70 | 188.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 15:15:00 | 188.40 | 187.70 | 188.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 188.40 | 187.70 | 188.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 188.35 | 187.70 | 188.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 187.50 | 187.66 | 188.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 186.81 | 187.39 | 187.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 186.85 | 187.34 | 187.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 186.85 | 187.24 | 187.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 177.47 | 180.79 | 183.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 177.51 | 180.79 | 183.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:15:00 | 177.51 | 180.79 | 183.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 178.09 | 177.76 | 179.93 | SL hit (close>ema200) qty=0.50 sl=177.76 alert=retest2 |

### Cycle 149 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 177.60 | 177.13 | 177.10 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 176.70 | 177.05 | 177.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 175.08 | 176.58 | 176.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 168.40 | 168.19 | 170.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 168.60 | 168.19 | 170.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 168.45 | 166.50 | 168.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 169.14 | 166.50 | 168.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 167.40 | 166.68 | 168.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 166.40 | 167.18 | 167.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 168.71 | 166.60 | 166.94 | SL hit (close>static) qty=1.00 sl=168.65 alert=retest2 |

### Cycle 151 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 167.70 | 167.06 | 167.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 168.35 | 167.32 | 167.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 170.61 | 171.78 | 170.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 14:15:00 | 170.61 | 171.78 | 170.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 170.61 | 171.78 | 170.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 170.61 | 171.78 | 170.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 170.00 | 171.42 | 170.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 169.38 | 171.42 | 170.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 169.10 | 170.68 | 170.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 169.29 | 170.68 | 170.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 171.32 | 171.19 | 170.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 171.12 | 171.19 | 170.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 171.25 | 171.21 | 170.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 170.43 | 171.21 | 170.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 169.66 | 170.90 | 170.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 169.66 | 170.90 | 170.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 170.15 | 170.75 | 170.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 169.28 | 169.99 | 170.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 162.75 | 161.47 | 163.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 162.75 | 161.47 | 163.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 162.75 | 161.47 | 163.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 163.78 | 161.47 | 163.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 162.84 | 161.74 | 162.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 162.84 | 161.74 | 162.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 164.82 | 162.36 | 163.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 164.82 | 162.36 | 163.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 164.64 | 162.82 | 163.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 164.90 | 162.82 | 163.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 165.83 | 163.81 | 163.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 168.72 | 165.29 | 164.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 172.75 | 172.83 | 170.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 172.75 | 172.83 | 170.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 170.72 | 172.19 | 170.78 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 167.58 | 169.86 | 170.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 167.23 | 169.34 | 169.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 168.86 | 168.23 | 168.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 168.86 | 168.23 | 168.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 168.86 | 168.23 | 168.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 169.05 | 168.23 | 168.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 168.76 | 168.34 | 168.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 168.76 | 168.34 | 168.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 169.15 | 168.50 | 168.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 170.20 | 168.50 | 168.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 171.88 | 169.18 | 169.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 173.89 | 171.69 | 170.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 173.18 | 173.57 | 172.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 173.18 | 173.57 | 172.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 172.94 | 173.45 | 172.61 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 170.64 | 172.14 | 172.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 169.60 | 171.63 | 172.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 171.19 | 170.78 | 171.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 171.19 | 170.78 | 171.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 171.19 | 170.78 | 171.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 171.10 | 170.78 | 171.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 171.00 | 170.78 | 171.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 171.30 | 170.78 | 171.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 171.35 | 170.89 | 171.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:15:00 | 172.04 | 170.89 | 171.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 172.26 | 171.17 | 171.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 172.26 | 171.17 | 171.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 172.69 | 171.47 | 171.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 185.63 | 174.30 | 172.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 185.13 | 185.71 | 183.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 185.13 | 185.71 | 183.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 185.16 | 185.61 | 184.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 185.96 | 185.56 | 184.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 183.64 | 184.93 | 184.61 | SL hit (close<static) qty=1.00 sl=184.10 alert=retest2 |

### Cycle 158 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 182.73 | 184.16 | 184.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 182.01 | 183.56 | 183.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 171.44 | 170.91 | 173.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 172.40 | 170.91 | 173.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 171.77 | 171.08 | 172.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 171.77 | 171.08 | 172.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 170.90 | 170.40 | 171.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 169.92 | 170.35 | 171.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 172.70 | 171.02 | 171.38 | SL hit (close>static) qty=1.00 sl=171.97 alert=retest2 |

### Cycle 159 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 172.82 | 171.62 | 171.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 173.36 | 172.13 | 171.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 172.80 | 172.94 | 172.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 10:45:00 | 172.68 | 172.94 | 172.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 173.28 | 173.00 | 172.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:30:00 | 174.00 | 173.41 | 172.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:45:00 | 173.88 | 173.58 | 172.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 175.03 | 173.66 | 173.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 176.48 | 177.24 | 177.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 176.48 | 177.24 | 177.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 174.13 | 176.62 | 176.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 172.73 | 172.53 | 173.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 172.73 | 172.53 | 173.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 173.61 | 172.75 | 173.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 173.61 | 172.75 | 173.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 173.79 | 172.96 | 173.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 173.79 | 172.96 | 173.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 174.23 | 173.21 | 173.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 174.65 | 173.21 | 173.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 173.28 | 173.23 | 173.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 173.98 | 173.23 | 173.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 173.53 | 173.29 | 173.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 173.53 | 173.29 | 173.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 173.51 | 173.33 | 173.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 174.56 | 173.33 | 173.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 174.25 | 173.52 | 173.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 174.85 | 173.52 | 173.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 173.62 | 173.54 | 173.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 173.48 | 173.54 | 173.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 171.74 | 170.10 | 170.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 171.74 | 170.10 | 170.04 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 170.30 | 171.11 | 171.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 170.07 | 170.90 | 171.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 170.47 | 170.43 | 170.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 170.47 | 170.43 | 170.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 170.47 | 170.43 | 170.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 168.65 | 169.84 | 170.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 160.22 | 164.13 | 166.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 163.45 | 163.30 | 164.92 | SL hit (close>ema200) qty=0.50 sl=163.30 alert=retest2 |

### Cycle 163 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 166.64 | 164.45 | 164.31 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 162.44 | 164.70 | 164.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 12:15:00 | 161.71 | 163.74 | 164.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 162.62 | 162.51 | 163.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 162.62 | 162.51 | 163.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 162.62 | 162.51 | 163.37 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 169.90 | 164.66 | 164.12 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 165.60 | 166.18 | 166.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 163.45 | 165.30 | 165.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 160.87 | 160.52 | 161.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 11:15:00 | 160.87 | 160.52 | 161.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 160.87 | 160.52 | 161.82 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 163.68 | 162.02 | 161.86 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 160.80 | 161.88 | 161.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 159.62 | 160.76 | 161.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 148.21 | 148.05 | 150.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:30:00 | 148.01 | 148.05 | 150.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 150.83 | 148.89 | 150.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 150.83 | 148.89 | 150.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 151.27 | 149.36 | 150.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 151.42 | 149.36 | 150.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 152.28 | 150.31 | 150.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 153.79 | 150.31 | 150.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 153.75 | 151.00 | 150.91 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 149.70 | 150.85 | 150.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 149.36 | 150.56 | 150.79 | Break + close below crossover candle low |

### Cycle 171 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 155.51 | 151.55 | 151.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 157.90 | 154.89 | 153.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 156.06 | 156.25 | 155.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 156.06 | 156.25 | 155.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 156.06 | 156.25 | 155.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 154.60 | 156.25 | 155.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 154.96 | 156.31 | 155.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 154.96 | 156.31 | 155.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 154.60 | 155.97 | 155.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 154.60 | 155.97 | 155.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 154.31 | 155.36 | 155.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 153.47 | 154.79 | 155.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 150.51 | 150.49 | 151.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 150.95 | 150.49 | 151.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 152.08 | 150.76 | 151.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 152.08 | 150.76 | 151.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 152.86 | 151.18 | 151.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 152.86 | 151.18 | 151.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 153.24 | 151.59 | 151.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 155.16 | 151.59 | 151.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 156.45 | 152.57 | 152.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 163.10 | 156.97 | 154.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 176.06 | 177.65 | 173.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 12:15:00 | 173.24 | 176.01 | 173.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 173.24 | 176.01 | 173.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 173.77 | 176.01 | 173.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 173.46 | 175.50 | 173.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:30:00 | 173.10 | 175.50 | 173.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 172.72 | 174.94 | 173.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 172.72 | 174.94 | 173.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 172.99 | 174.55 | 173.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 170.93 | 174.55 | 173.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 173.22 | 173.78 | 173.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 173.00 | 173.78 | 173.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 173.15 | 173.65 | 173.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 173.20 | 173.65 | 173.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 172.70 | 173.46 | 173.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:30:00 | 172.72 | 173.46 | 173.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 171.67 | 173.10 | 173.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 171.67 | 173.10 | 173.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 171.81 | 172.85 | 172.96 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 178.25 | 173.93 | 173.44 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 176.68 | 177.46 | 177.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 176.00 | 177.08 | 177.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 177.33 | 177.13 | 177.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 177.33 | 177.13 | 177.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 177.33 | 177.13 | 177.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 177.33 | 177.13 | 177.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 176.96 | 177.10 | 177.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 177.93 | 177.10 | 177.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 177.50 | 177.18 | 177.34 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 177.66 | 177.44 | 177.43 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 175.15 | 176.98 | 177.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 173.06 | 176.20 | 176.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 164.60 | 163.87 | 166.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 164.01 | 163.87 | 166.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 164.01 | 163.90 | 166.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 162.22 | 163.27 | 165.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 162.14 | 162.43 | 164.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 166.50 | 163.33 | 164.29 | SL hit (close>static) qty=1.00 sl=166.48 alert=retest2 |

### Cycle 179 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 165.61 | 164.71 | 164.70 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 164.36 | 164.64 | 164.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 163.13 | 164.26 | 164.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 163.40 | 163.35 | 163.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:00:00 | 163.40 | 163.35 | 163.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 156.67 | 154.61 | 156.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 156.84 | 154.61 | 156.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 155.90 | 154.86 | 156.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 155.67 | 154.86 | 156.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 157.92 | 155.51 | 156.31 | SL hit (close>static) qty=1.00 sl=156.88 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 159.70 | 155.98 | 155.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 160.84 | 156.95 | 156.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 160.41 | 160.85 | 159.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 160.41 | 160.85 | 159.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 163.97 | 161.86 | 160.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:45:00 | 164.90 | 162.39 | 160.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 166.56 | 163.41 | 161.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 157.11 | 161.53 | 161.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 157.11 | 161.53 | 161.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 152.92 | 159.81 | 160.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 154.46 | 153.45 | 156.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:45:00 | 153.57 | 153.45 | 156.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 156.10 | 154.31 | 156.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:30:00 | 155.45 | 155.26 | 156.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 13:15:00 | 157.57 | 156.28 | 156.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 157.57 | 156.28 | 156.27 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 154.97 | 156.17 | 156.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 151.50 | 154.40 | 155.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 153.25 | 153.13 | 154.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 153.25 | 153.13 | 154.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 153.96 | 153.32 | 154.10 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 156.21 | 154.77 | 154.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 156.52 | 155.12 | 154.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 157.85 | 158.82 | 157.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 157.85 | 158.82 | 157.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 157.85 | 158.82 | 157.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 157.56 | 158.82 | 157.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 157.11 | 158.48 | 157.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 157.11 | 158.48 | 157.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 157.31 | 158.25 | 157.46 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 156.10 | 157.08 | 157.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 154.56 | 156.57 | 156.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 153.30 | 152.96 | 153.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 153.30 | 152.96 | 153.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 152.90 | 153.01 | 153.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 153.65 | 153.01 | 153.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 153.55 | 153.11 | 153.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:30:00 | 153.80 | 153.12 | 153.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 152.10 | 152.92 | 153.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 152.85 | 152.92 | 153.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 152.45 | 152.53 | 153.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 151.75 | 152.53 | 153.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:30:00 | 152.11 | 152.23 | 152.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 151.85 | 152.06 | 152.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 144.16 | 145.45 | 146.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 144.50 | 145.45 | 146.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 144.26 | 145.45 | 146.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 136.58 | 143.11 | 144.78 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 187 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 147.14 | 137.74 | 136.60 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 137.79 | 139.42 | 139.50 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 139.61 | 139.40 | 139.37 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 138.85 | 139.29 | 139.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 138.19 | 138.98 | 139.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 137.28 | 137.22 | 138.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 137.28 | 137.22 | 138.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 137.46 | 137.34 | 137.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 137.81 | 137.34 | 137.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 128.16 | 126.58 | 128.12 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 129.80 | 128.92 | 128.81 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 125.95 | 128.33 | 128.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 124.68 | 126.62 | 127.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 125.98 | 125.79 | 126.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 125.98 | 125.79 | 126.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 125.98 | 125.79 | 126.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 126.46 | 125.79 | 126.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 125.21 | 125.72 | 126.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 126.39 | 125.72 | 126.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 123.73 | 119.37 | 120.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 124.00 | 119.37 | 120.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 127.28 | 120.95 | 120.74 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 121.21 | 121.59 | 121.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 118.90 | 120.94 | 121.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 124.38 | 118.90 | 119.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 124.38 | 118.90 | 119.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 124.38 | 118.90 | 119.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 124.38 | 118.90 | 119.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 123.44 | 119.81 | 120.05 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 125.29 | 120.90 | 120.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 126.05 | 121.93 | 121.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 120.25 | 122.97 | 121.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 120.25 | 122.97 | 121.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 120.25 | 122.97 | 121.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 122.71 | 122.09 | 121.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 122.99 | 123.47 | 122.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 123.32 | 123.47 | 122.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 134.98 | 132.48 | 129.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 149.39 | 152.96 | 153.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 148.75 | 151.55 | 152.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 152.81 | 151.23 | 151.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 152.81 | 151.23 | 151.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 152.81 | 151.23 | 151.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 153.70 | 151.23 | 151.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 154.23 | 151.83 | 152.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 154.43 | 151.83 | 152.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 154.12 | 152.50 | 152.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 154.48 | 152.89 | 152.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 153.00 | 153.42 | 153.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 153.00 | 153.42 | 153.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 153.00 | 153.42 | 153.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 153.20 | 153.42 | 153.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 153.21 | 153.38 | 153.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 13:15:00 | 153.34 | 153.38 | 153.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:30:00 | 154.00 | 153.42 | 153.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 154.00 | 153.95 | 153.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 151.09 | 153.39 | 153.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 151.09 | 153.39 | 153.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 150.61 | 152.83 | 153.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 152.51 | 152.13 | 152.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 152.51 | 152.13 | 152.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 152.51 | 152.13 | 152.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 154.06 | 152.13 | 152.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 153.69 | 152.44 | 152.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 153.40 | 152.81 | 152.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 153.33 | 152.91 | 152.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 153.33 | 152.91 | 152.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 158.80 | 154.44 | 153.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 158.41 | 158.47 | 157.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 158.41 | 158.47 | 157.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 159.04 | 160.46 | 159.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 158.88 | 160.46 | 159.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 158.35 | 160.04 | 159.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 158.35 | 160.04 | 159.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 158.98 | 159.67 | 159.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 158.67 | 159.67 | 159.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 158.05 | 159.34 | 158.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 157.95 | 159.34 | 158.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 158.45 | 159.16 | 158.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 158.99 | 159.16 | 158.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 158.99 | 159.13 | 158.94 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 14:45:00 | 78.60 | 2023-05-17 09:15:00 | 82.25 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2023-05-30 12:15:00 | 82.20 | 2023-05-30 12:15:00 | 81.55 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-06-12 12:15:00 | 83.65 | 2023-06-16 15:15:00 | 83.25 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2023-06-12 14:00:00 | 83.60 | 2023-06-16 15:15:00 | 83.25 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2023-06-13 09:45:00 | 83.40 | 2023-06-16 15:15:00 | 83.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2023-07-03 15:15:00 | 83.35 | 2023-07-06 10:15:00 | 84.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-07-04 09:30:00 | 83.10 | 2023-07-06 10:15:00 | 84.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-07-04 10:45:00 | 83.20 | 2023-07-06 10:15:00 | 84.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-07-04 11:30:00 | 83.35 | 2023-07-06 10:15:00 | 84.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-07-05 11:45:00 | 83.30 | 2023-07-06 10:15:00 | 84.10 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-07-05 12:15:00 | 83.25 | 2023-07-06 10:15:00 | 84.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-07-10 13:15:00 | 82.35 | 2023-07-17 10:15:00 | 83.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-07-10 14:15:00 | 82.25 | 2023-07-17 11:15:00 | 83.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2023-07-11 09:15:00 | 82.00 | 2023-07-17 11:15:00 | 83.95 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2023-07-11 11:30:00 | 82.25 | 2023-07-17 11:15:00 | 83.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2023-07-12 13:15:00 | 82.25 | 2023-07-17 11:15:00 | 83.95 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-07-26 09:15:00 | 93.10 | 2023-07-28 12:15:00 | 92.35 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-08-09 09:15:00 | 105.90 | 2023-08-11 14:15:00 | 103.05 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2023-08-16 09:15:00 | 102.35 | 2023-08-16 12:15:00 | 103.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-08-16 11:30:00 | 101.60 | 2023-08-16 12:15:00 | 103.40 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-08-16 12:15:00 | 102.10 | 2023-08-16 12:15:00 | 103.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-09-01 10:30:00 | 115.20 | 2023-09-04 12:15:00 | 126.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-13 14:30:00 | 148.20 | 2023-09-15 12:15:00 | 145.35 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-09-13 15:15:00 | 147.95 | 2023-09-15 12:15:00 | 145.35 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2023-10-10 12:15:00 | 137.05 | 2023-10-11 09:15:00 | 138.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-10-11 13:45:00 | 137.25 | 2023-10-13 09:15:00 | 141.75 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2023-10-20 09:15:00 | 160.75 | 2023-10-23 09:15:00 | 148.85 | STOP_HIT | 1.00 | -7.40% |
| SELL | retest2 | 2023-10-30 09:15:00 | 137.90 | 2023-10-31 12:15:00 | 140.30 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2023-11-02 11:00:00 | 139.35 | 2023-11-03 09:15:00 | 141.25 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-11-13 10:15:00 | 156.70 | 2023-11-16 15:15:00 | 154.75 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-11-22 14:30:00 | 164.20 | 2023-11-29 15:15:00 | 165.95 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2023-12-05 14:45:00 | 173.55 | 2023-12-06 09:15:00 | 171.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2023-12-06 11:45:00 | 173.70 | 2023-12-06 12:15:00 | 171.80 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-01-02 12:45:00 | 173.90 | 2024-01-08 13:15:00 | 181.45 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2024-01-02 13:30:00 | 173.90 | 2024-01-08 13:15:00 | 181.45 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2024-01-02 14:00:00 | 174.00 | 2024-01-08 13:15:00 | 181.45 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2024-01-02 14:45:00 | 173.90 | 2024-01-08 13:15:00 | 181.45 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2024-01-03 10:45:00 | 181.25 | 2024-01-08 13:15:00 | 181.45 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-01-08 12:00:00 | 181.10 | 2024-01-08 13:15:00 | 181.45 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-01-10 11:45:00 | 180.25 | 2024-01-10 13:15:00 | 189.70 | STOP_HIT | 1.00 | -5.24% |
| BUY | retest1 | 2024-01-15 09:15:00 | 198.50 | 2024-01-15 10:15:00 | 208.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-15 09:15:00 | 198.50 | 2024-01-15 14:15:00 | 202.05 | STOP_HIT | 0.50 | 1.79% |
| BUY | retest2 | 2024-02-08 10:15:00 | 232.45 | 2024-02-08 15:15:00 | 228.70 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-02-14 09:15:00 | 209.15 | 2024-02-14 09:15:00 | 215.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-03-07 12:15:00 | 223.35 | 2024-03-12 09:15:00 | 212.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 12:45:00 | 223.50 | 2024-03-12 09:15:00 | 212.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 13:45:00 | 223.35 | 2024-03-12 09:15:00 | 212.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 15:15:00 | 223.40 | 2024-03-12 09:15:00 | 212.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 10:45:00 | 219.70 | 2024-03-12 09:15:00 | 208.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 14:15:00 | 219.90 | 2024-03-12 09:15:00 | 208.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 12:15:00 | 223.35 | 2024-03-13 09:15:00 | 201.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 12:45:00 | 223.50 | 2024-03-13 09:15:00 | 201.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 13:45:00 | 223.35 | 2024-03-13 09:15:00 | 201.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 15:15:00 | 223.40 | 2024-03-13 09:15:00 | 201.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-11 10:45:00 | 219.70 | 2024-03-13 09:15:00 | 197.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-11 14:15:00 | 219.90 | 2024-03-13 09:15:00 | 197.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-20 11:30:00 | 214.00 | 2024-03-20 14:15:00 | 210.30 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-03-20 13:00:00 | 213.75 | 2024-03-20 14:15:00 | 210.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-03-28 11:30:00 | 224.90 | 2024-03-28 14:15:00 | 220.30 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-04-04 13:30:00 | 230.70 | 2024-04-09 12:15:00 | 230.55 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-04-05 10:00:00 | 231.65 | 2024-04-09 12:15:00 | 230.55 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-04-12 15:00:00 | 226.25 | 2024-04-15 09:15:00 | 214.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 15:00:00 | 226.25 | 2024-04-16 15:15:00 | 220.90 | STOP_HIT | 0.50 | 2.36% |
| BUY | retest2 | 2024-04-30 12:15:00 | 250.20 | 2024-05-03 11:15:00 | 245.90 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-05-07 09:15:00 | 242.30 | 2024-05-08 09:15:00 | 230.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 09:15:00 | 242.30 | 2024-05-08 09:15:00 | 238.00 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2024-05-07 10:15:00 | 241.90 | 2024-05-08 09:15:00 | 229.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 10:15:00 | 241.90 | 2024-05-08 09:15:00 | 238.00 | STOP_HIT | 0.50 | 1.61% |
| BUY | retest2 | 2024-05-23 09:15:00 | 287.80 | 2024-05-24 09:15:00 | 274.55 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2024-05-23 15:15:00 | 281.50 | 2024-05-24 09:15:00 | 274.55 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-05-27 12:15:00 | 274.70 | 2024-05-27 13:15:00 | 277.75 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-28 09:15:00 | 274.10 | 2024-06-03 09:15:00 | 288.45 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2024-05-30 11:00:00 | 273.80 | 2024-06-03 09:15:00 | 288.45 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2024-06-06 11:30:00 | 256.95 | 2024-06-10 12:15:00 | 252.90 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2024-06-13 12:15:00 | 268.95 | 2024-06-19 13:15:00 | 269.20 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-06-13 13:30:00 | 268.55 | 2024-06-19 13:15:00 | 269.20 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-06-13 15:15:00 | 268.70 | 2024-06-19 13:15:00 | 269.20 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-06-14 09:45:00 | 270.70 | 2024-06-19 13:15:00 | 269.20 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-06-27 11:15:00 | 271.30 | 2024-07-01 15:15:00 | 271.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-06-28 13:30:00 | 271.40 | 2024-07-01 15:15:00 | 271.70 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-07-01 12:15:00 | 271.15 | 2024-07-01 15:15:00 | 271.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-07-01 12:45:00 | 271.50 | 2024-07-01 15:15:00 | 271.70 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-07-02 14:30:00 | 275.55 | 2024-07-05 13:15:00 | 303.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 11:30:00 | 275.25 | 2024-07-05 13:15:00 | 302.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 14:45:00 | 275.55 | 2024-07-05 13:15:00 | 303.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-05 09:15:00 | 268.35 | 2024-08-08 09:15:00 | 278.80 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-08-16 13:00:00 | 265.50 | 2024-08-16 13:15:00 | 268.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-08-22 14:00:00 | 265.45 | 2024-08-23 09:15:00 | 270.40 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-08-22 14:45:00 | 265.20 | 2024-08-23 09:15:00 | 270.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-08-22 15:15:00 | 265.20 | 2024-08-23 09:15:00 | 270.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-08-29 10:30:00 | 262.80 | 2024-09-06 09:15:00 | 249.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-29 13:00:00 | 263.15 | 2024-09-06 09:15:00 | 249.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-29 13:45:00 | 262.85 | 2024-09-06 09:15:00 | 249.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 09:15:00 | 262.50 | 2024-09-06 09:15:00 | 249.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 15:00:00 | 261.00 | 2024-09-06 09:15:00 | 247.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:30:00 | 260.50 | 2024-09-06 09:15:00 | 247.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-29 10:30:00 | 262.80 | 2024-09-10 09:15:00 | 242.80 | STOP_HIT | 0.50 | 7.61% |
| SELL | retest2 | 2024-08-29 13:00:00 | 263.15 | 2024-09-10 09:15:00 | 242.80 | STOP_HIT | 0.50 | 7.73% |
| SELL | retest2 | 2024-08-29 13:45:00 | 262.85 | 2024-09-10 09:15:00 | 242.80 | STOP_HIT | 0.50 | 7.63% |
| SELL | retest2 | 2024-08-30 09:15:00 | 262.50 | 2024-09-10 09:15:00 | 242.80 | STOP_HIT | 0.50 | 7.50% |
| SELL | retest2 | 2024-08-30 15:00:00 | 261.00 | 2024-09-10 09:15:00 | 242.80 | STOP_HIT | 0.50 | 6.97% |
| SELL | retest2 | 2024-09-02 09:30:00 | 260.50 | 2024-09-10 09:15:00 | 242.80 | STOP_HIT | 0.50 | 6.79% |
| SELL | retest2 | 2024-09-26 12:45:00 | 224.80 | 2024-09-27 15:15:00 | 227.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-26 13:15:00 | 224.75 | 2024-09-27 15:15:00 | 227.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-09-26 15:00:00 | 224.60 | 2024-09-27 15:15:00 | 227.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-10-29 10:30:00 | 199.81 | 2024-10-29 11:15:00 | 201.31 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2024-11-12 12:45:00 | 200.50 | 2024-11-18 09:15:00 | 190.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-12 12:45:00 | 200.50 | 2024-11-19 09:15:00 | 192.80 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2024-11-19 14:00:00 | 190.46 | 2024-11-25 09:15:00 | 197.20 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2024-12-02 15:00:00 | 216.09 | 2024-12-10 15:15:00 | 223.00 | STOP_HIT | 1.00 | 3.20% |
| SELL | retest2 | 2024-12-23 15:15:00 | 211.25 | 2024-12-31 10:15:00 | 218.16 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-12-24 13:45:00 | 211.72 | 2024-12-31 10:15:00 | 218.16 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-12-26 09:30:00 | 210.80 | 2024-12-31 10:15:00 | 218.16 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-12-31 10:00:00 | 211.15 | 2024-12-31 10:15:00 | 218.16 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-01-08 13:00:00 | 206.14 | 2025-01-10 09:15:00 | 195.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 205.65 | 2025-01-10 10:15:00 | 195.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:30:00 | 205.67 | 2025-01-10 10:15:00 | 195.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 203.40 | 2025-01-10 13:15:00 | 193.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 13:00:00 | 206.14 | 2025-01-13 12:15:00 | 185.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 205.65 | 2025-01-13 13:15:00 | 185.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 14:30:00 | 205.67 | 2025-01-13 13:15:00 | 185.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 15:15:00 | 203.40 | 2025-01-13 13:15:00 | 183.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 14:15:00 | 188.90 | 2025-01-16 09:15:00 | 202.83 | STOP_HIT | 1.00 | -7.37% |
| SELL | retest2 | 2025-01-23 14:15:00 | 208.66 | 2025-01-27 09:15:00 | 198.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 209.14 | 2025-01-27 09:15:00 | 198.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 208.66 | 2025-01-28 10:15:00 | 187.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 209.14 | 2025-01-28 10:15:00 | 188.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 14:15:00 | 180.55 | 2025-02-12 09:15:00 | 162.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-11 15:00:00 | 181.35 | 2025-02-12 09:15:00 | 163.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-04 11:30:00 | 143.41 | 2025-03-05 09:15:00 | 147.24 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-03-04 15:00:00 | 143.26 | 2025-03-05 09:15:00 | 147.24 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-03-10 11:30:00 | 148.86 | 2025-03-10 14:15:00 | 148.21 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-03-12 11:15:00 | 143.19 | 2025-03-18 09:15:00 | 147.81 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-03-13 10:30:00 | 142.93 | 2025-03-18 09:15:00 | 147.81 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-03-28 11:15:00 | 159.86 | 2025-04-02 14:15:00 | 160.82 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-04-08 10:30:00 | 146.35 | 2025-04-11 14:15:00 | 149.72 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-04-09 09:30:00 | 146.56 | 2025-04-11 14:15:00 | 149.72 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-04-09 13:30:00 | 146.47 | 2025-04-11 14:15:00 | 149.72 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-04-23 14:15:00 | 162.85 | 2025-04-25 09:15:00 | 158.10 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-04-29 10:15:00 | 157.31 | 2025-05-05 11:15:00 | 157.95 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-04-29 11:45:00 | 157.23 | 2025-05-05 11:15:00 | 157.95 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-04-29 12:15:00 | 157.04 | 2025-05-05 11:15:00 | 157.95 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-04-29 12:45:00 | 157.24 | 2025-05-05 11:15:00 | 157.95 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-05-02 11:15:00 | 155.61 | 2025-05-05 11:15:00 | 157.95 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-05-21 09:15:00 | 190.98 | 2025-05-21 11:15:00 | 186.96 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-21 14:00:00 | 189.08 | 2025-05-22 09:15:00 | 186.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-21 14:45:00 | 189.73 | 2025-05-22 09:15:00 | 186.10 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-05-26 11:45:00 | 186.87 | 2025-05-26 13:15:00 | 187.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-27 11:15:00 | 189.21 | 2025-06-04 11:15:00 | 208.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 14:30:00 | 188.46 | 2025-06-04 11:15:00 | 207.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 203.42 | 2025-06-19 11:15:00 | 193.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 10:30:00 | 204.20 | 2025-06-19 11:15:00 | 193.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 203.42 | 2025-06-20 10:15:00 | 194.85 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-06-17 10:30:00 | 204.20 | 2025-06-20 10:15:00 | 194.85 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2025-07-07 11:00:00 | 197.53 | 2025-07-18 09:15:00 | 187.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 11:00:00 | 197.53 | 2025-07-21 09:15:00 | 191.05 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-07-24 13:00:00 | 186.81 | 2025-07-28 12:15:00 | 177.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:15:00 | 186.85 | 2025-07-28 12:15:00 | 177.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 15:00:00 | 186.85 | 2025-07-28 12:15:00 | 177.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:00:00 | 186.81 | 2025-07-29 13:15:00 | 178.09 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2025-07-24 14:15:00 | 186.85 | 2025-07-29 13:15:00 | 178.09 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-07-24 15:00:00 | 186.85 | 2025-07-29 13:15:00 | 178.09 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-14 10:30:00 | 166.40 | 2025-08-18 09:15:00 | 168.71 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-08-18 12:00:00 | 166.39 | 2025-08-18 15:15:00 | 167.70 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-09-19 12:00:00 | 185.96 | 2025-09-22 09:15:00 | 183.64 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-30 13:15:00 | 169.92 | 2025-10-01 09:15:00 | 172.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-10-03 14:30:00 | 174.00 | 2025-10-10 15:15:00 | 176.48 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-10-06 09:45:00 | 173.88 | 2025-10-10 15:15:00 | 176.48 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2025-10-07 12:45:00 | 175.03 | 2025-10-10 15:15:00 | 176.48 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-10-16 11:15:00 | 173.48 | 2025-10-29 11:15:00 | 171.74 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 168.65 | 2025-11-07 09:15:00 | 160.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 168.65 | 2025-11-07 14:15:00 | 163.45 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2026-01-13 11:30:00 | 162.22 | 2026-01-14 11:15:00 | 166.50 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-01-14 10:00:00 | 162.14 | 2026-01-14 11:15:00 | 166.50 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-01-22 11:15:00 | 155.67 | 2026-01-22 14:15:00 | 157.92 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-23 13:15:00 | 155.04 | 2026-01-28 09:15:00 | 157.36 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-30 10:45:00 | 164.90 | 2026-02-01 13:15:00 | 157.11 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest2 | 2026-02-01 09:15:00 | 166.56 | 2026-02-01 13:15:00 | 157.11 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2026-02-03 14:30:00 | 155.45 | 2026-02-04 13:15:00 | 157.57 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-02-18 10:15:00 | 151.75 | 2026-02-27 09:15:00 | 144.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 152.11 | 2026-02-27 09:15:00 | 144.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 151.85 | 2026-02-27 09:15:00 | 144.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 151.75 | 2026-03-02 09:15:00 | 136.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 152.11 | 2026-03-02 09:15:00 | 136.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 151.85 | 2026-03-02 09:15:00 | 136.66 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-02 13:15:00 | 122.71 | 2026-04-09 09:15:00 | 134.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:30:00 | 122.99 | 2026-04-09 09:15:00 | 135.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:00:00 | 123.32 | 2026-04-09 09:15:00 | 135.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-28 13:15:00 | 153.34 | 2026-04-30 09:15:00 | 151.09 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-04-28 14:30:00 | 154.00 | 2026-04-30 09:15:00 | 151.09 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-04-29 15:15:00 | 154.00 | 2026-04-30 09:15:00 | 151.09 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-05-04 10:45:00 | 153.40 | 2026-05-04 11:15:00 | 153.33 | STOP_HIT | 1.00 | 0.05% |
