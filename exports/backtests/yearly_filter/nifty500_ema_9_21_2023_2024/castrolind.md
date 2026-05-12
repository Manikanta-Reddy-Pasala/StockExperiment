# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 185.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 224 |
| ALERT1 | 144 |
| ALERT2 | 143 |
| ALERT2_SKIP | 81 |
| ALERT3 | 388 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 157 |
| PARTIAL | 32 |
| TARGET_HIT | 9 |
| STOP_HIT | 154 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 195 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 100 / 95
- **Target hits / Stop hits / Partials:** 9 / 154 / 32
- **Avg / median % per leg:** 1.31% / 0.61%
- **Sum % (uncompounded):** 255.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 18 | 31.0% | 3 | 53 | 2 | 0.46% | 26.5% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 0 | 3 | 2 | 3.82% | 19.1% |
| BUY @ 3rd Alert (retest2) | 53 | 13 | 24.5% | 3 | 50 | 0 | 0.14% | 7.4% |
| SELL (all) | 137 | 82 | 59.9% | 6 | 101 | 30 | 1.67% | 229.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.93% | -5.8% |
| SELL @ 3rd Alert (retest2) | 134 | 82 | 61.2% | 6 | 98 | 30 | 1.75% | 235.2% |
| retest1 (combined) | 8 | 5 | 62.5% | 0 | 6 | 2 | 1.66% | 13.3% |
| retest2 (combined) | 187 | 95 | 50.8% | 9 | 148 | 30 | 1.30% | 242.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 113.65 | 113.01 | 112.98 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 112.85 | 113.05 | 113.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 12:15:00 | 112.70 | 112.98 | 113.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 15:15:00 | 113.15 | 112.98 | 113.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 15:15:00 | 113.15 | 112.98 | 113.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 113.15 | 112.98 | 113.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 114.20 | 112.98 | 113.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 114.35 | 113.25 | 113.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 09:15:00 | 114.65 | 114.21 | 113.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 13:15:00 | 113.80 | 114.19 | 113.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 13:15:00 | 113.80 | 114.19 | 113.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 113.80 | 114.19 | 113.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 14:00:00 | 113.80 | 114.19 | 113.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 14:15:00 | 113.70 | 114.09 | 113.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 14:45:00 | 113.70 | 114.09 | 113.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 113.55 | 113.97 | 113.92 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 10:15:00 | 113.35 | 113.84 | 113.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 11:15:00 | 113.05 | 113.69 | 113.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 113.40 | 113.03 | 113.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 113.40 | 113.03 | 113.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 113.40 | 113.03 | 113.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:00:00 | 113.40 | 113.03 | 113.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 113.75 | 113.17 | 113.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:30:00 | 113.85 | 113.17 | 113.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 11:15:00 | 113.70 | 113.28 | 113.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 11:30:00 | 113.75 | 113.28 | 113.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 13:15:00 | 113.60 | 113.42 | 113.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 13:30:00 | 113.85 | 113.42 | 113.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 15:15:00 | 113.85 | 113.56 | 113.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 11:15:00 | 114.40 | 113.83 | 113.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 14:15:00 | 114.55 | 114.72 | 114.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 14:15:00 | 114.55 | 114.72 | 114.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 114.55 | 114.72 | 114.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 15:00:00 | 114.55 | 114.72 | 114.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 114.70 | 114.71 | 114.40 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 14:15:00 | 113.85 | 114.27 | 114.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 09:15:00 | 113.50 | 113.87 | 114.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 113.80 | 113.25 | 113.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 113.80 | 113.25 | 113.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 113.80 | 113.25 | 113.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 114.00 | 113.25 | 113.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 10:15:00 | 116.25 | 113.85 | 113.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 118.05 | 115.93 | 114.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 15:15:00 | 119.50 | 119.72 | 118.88 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:15:00 | 120.70 | 119.72 | 118.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 10:30:00 | 120.40 | 119.83 | 119.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 09:15:00 | 126.42 | 123.39 | 122.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-06-22 09:15:00 | 124.55 | 124.87 | 123.69 | SL hit (close<ema200) qty=0.50 sl=124.87 alert=retest1 |

### Cycle 8 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 121.95 | 123.26 | 123.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 120.80 | 122.54 | 122.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 10:15:00 | 121.50 | 121.06 | 121.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 11:00:00 | 121.50 | 121.06 | 121.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 121.35 | 121.03 | 121.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 121.90 | 121.03 | 121.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 121.40 | 121.10 | 121.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:45:00 | 121.45 | 121.10 | 121.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 121.30 | 121.14 | 121.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:00:00 | 121.30 | 121.14 | 121.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 121.50 | 121.20 | 121.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:00:00 | 121.50 | 121.20 | 121.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 14:15:00 | 121.60 | 121.28 | 121.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 14:30:00 | 121.70 | 121.28 | 121.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 15:15:00 | 121.95 | 121.41 | 121.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:15:00 | 122.30 | 121.41 | 121.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 122.15 | 121.56 | 121.52 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 121.25 | 121.62 | 121.62 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 122.90 | 121.48 | 121.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 12:15:00 | 124.00 | 122.32 | 121.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 11:15:00 | 126.50 | 126.89 | 125.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-10 12:00:00 | 126.50 | 126.89 | 125.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 125.45 | 126.65 | 126.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 09:30:00 | 127.40 | 126.90 | 126.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 14:15:00 | 131.65 | 131.92 | 131.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 14:15:00 | 131.65 | 131.92 | 131.94 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 15:15:00 | 132.70 | 132.08 | 132.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 132.95 | 132.25 | 132.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 13:15:00 | 131.90 | 132.25 | 132.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 13:15:00 | 131.90 | 132.25 | 132.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 131.90 | 132.25 | 132.15 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 14:15:00 | 131.25 | 132.05 | 132.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 09:15:00 | 130.15 | 131.57 | 131.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 11:15:00 | 132.10 | 130.53 | 130.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 11:15:00 | 132.10 | 130.53 | 130.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 132.10 | 130.53 | 130.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:00:00 | 132.10 | 130.53 | 130.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 133.70 | 131.16 | 131.17 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 133.00 | 131.53 | 131.34 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 15:15:00 | 130.75 | 131.44 | 131.47 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 09:15:00 | 132.05 | 131.56 | 131.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 10:15:00 | 133.00 | 131.85 | 131.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 134.85 | 134.89 | 133.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:15:00 | 137.55 | 134.89 | 133.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 134.70 | 135.32 | 134.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 13:30:00 | 134.55 | 135.32 | 134.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 137.20 | 135.70 | 134.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 11:30:00 | 137.40 | 136.74 | 135.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 12:15:00 | 144.43 | 138.20 | 136.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-07-28 09:15:00 | 143.75 | 144.38 | 142.01 | SL hit (close<ema200) qty=0.50 sl=144.38 alert=retest1 |

### Cycle 18 — SELL (started 2023-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 14:15:00 | 144.35 | 145.31 | 145.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 10:15:00 | 142.40 | 144.58 | 145.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 11:15:00 | 141.70 | 141.54 | 142.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 12:45:00 | 140.80 | 141.30 | 142.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 15:15:00 | 141.30 | 141.17 | 142.30 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 144.50 | 141.86 | 142.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-04 09:15:00 | 144.50 | 141.86 | 142.41 | SL hit (close>ema400) qty=1.00 sl=142.41 alert=retest1 |

### Cycle 19 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 147.20 | 142.93 | 142.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 13:15:00 | 147.90 | 144.91 | 143.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 149.35 | 150.55 | 148.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 149.35 | 150.55 | 148.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 149.35 | 150.55 | 148.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 09:30:00 | 148.15 | 150.55 | 148.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 147.60 | 149.96 | 148.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 147.60 | 149.96 | 148.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 147.70 | 149.51 | 148.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 147.50 | 149.51 | 148.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 147.65 | 148.64 | 148.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 15:00:00 | 147.65 | 148.64 | 148.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 147.65 | 148.44 | 148.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:15:00 | 147.05 | 148.44 | 148.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 148.80 | 148.50 | 148.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 148.50 | 148.50 | 148.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 156.25 | 152.67 | 150.74 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 149.10 | 151.70 | 152.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 11:15:00 | 145.40 | 147.56 | 148.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 10:15:00 | 146.60 | 145.66 | 147.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 10:15:00 | 146.60 | 145.66 | 147.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 146.60 | 145.66 | 147.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:45:00 | 146.75 | 145.66 | 147.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 146.50 | 146.04 | 146.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 14:30:00 | 147.10 | 146.04 | 146.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 146.90 | 146.25 | 146.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:15:00 | 147.50 | 146.25 | 146.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 147.60 | 146.52 | 146.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:00:00 | 147.60 | 146.52 | 146.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 146.95 | 146.79 | 146.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 15:15:00 | 146.50 | 146.83 | 146.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-22 09:45:00 | 146.65 | 146.71 | 146.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 10:45:00 | 146.65 | 145.96 | 146.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 09:15:00 | 147.85 | 143.99 | 144.53 | SL hit (close>static) qty=1.00 sl=147.30 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-25 11:15:00 | 147.40 | 145.23 | 145.04 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 10:15:00 | 144.80 | 145.07 | 145.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 09:15:00 | 142.95 | 144.22 | 144.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 144.90 | 143.73 | 144.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 144.90 | 143.73 | 144.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 144.90 | 143.73 | 144.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:45:00 | 144.90 | 143.73 | 144.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 144.85 | 143.95 | 144.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:30:00 | 145.70 | 143.95 | 144.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 144.15 | 144.03 | 144.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 13:00:00 | 144.15 | 144.03 | 144.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 143.50 | 143.92 | 144.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 12:45:00 | 142.90 | 143.42 | 143.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 14:00:00 | 143.15 | 143.37 | 143.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 14:45:00 | 142.95 | 143.53 | 143.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 15:15:00 | 145.05 | 143.83 | 143.86 | SL hit (close>static) qty=1.00 sl=144.20 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 144.15 | 143.89 | 143.89 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 10:15:00 | 143.35 | 143.79 | 143.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 12:15:00 | 143.05 | 143.58 | 143.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 143.60 | 143.31 | 143.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 143.60 | 143.31 | 143.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 143.60 | 143.31 | 143.53 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 148.55 | 144.43 | 143.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 09:15:00 | 157.65 | 149.69 | 147.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 12:15:00 | 157.20 | 158.29 | 156.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-08 13:00:00 | 157.20 | 158.29 | 156.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 156.05 | 157.59 | 156.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:30:00 | 156.00 | 157.59 | 156.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 157.45 | 157.57 | 156.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:15:00 | 155.20 | 157.57 | 156.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 154.15 | 156.88 | 156.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:45:00 | 154.45 | 156.88 | 156.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 154.05 | 156.32 | 155.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:45:00 | 153.85 | 156.32 | 155.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 12:15:00 | 154.00 | 155.51 | 155.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 13:15:00 | 153.75 | 155.16 | 155.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 148.20 | 148.08 | 149.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 15:00:00 | 148.20 | 148.08 | 149.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 147.80 | 148.09 | 149.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:45:00 | 147.55 | 148.09 | 149.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:15:00 | 147.40 | 148.00 | 149.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 15:00:00 | 145.20 | 147.64 | 148.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:30:00 | 147.30 | 146.78 | 147.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 14:15:00 | 140.17 | 142.57 | 144.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 14:15:00 | 140.03 | 142.57 | 144.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-21 14:15:00 | 139.94 | 142.57 | 144.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-22 11:15:00 | 142.20 | 142.00 | 143.38 | SL hit (close>ema200) qty=0.50 sl=142.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 15:15:00 | 138.60 | 138.14 | 138.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 139.20 | 138.39 | 138.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 12:15:00 | 138.20 | 138.43 | 138.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 12:15:00 | 138.20 | 138.43 | 138.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 138.20 | 138.43 | 138.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:45:00 | 138.35 | 138.43 | 138.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 139.00 | 138.54 | 138.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 13:30:00 | 138.60 | 138.54 | 138.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 138.25 | 138.48 | 138.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 15:00:00 | 138.25 | 138.48 | 138.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 137.85 | 138.36 | 138.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 09:30:00 | 138.35 | 138.26 | 138.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 10:15:00 | 138.80 | 138.26 | 138.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 11:15:00 | 137.50 | 138.21 | 138.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 137.50 | 138.21 | 138.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 136.70 | 137.26 | 137.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 139.55 | 137.46 | 137.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 139.55 | 137.46 | 137.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 139.55 | 137.46 | 137.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 10:00:00 | 139.55 | 137.46 | 137.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 138.50 | 137.67 | 137.66 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 136.20 | 137.45 | 137.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 134.75 | 135.86 | 136.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 137.15 | 136.12 | 136.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 137.15 | 136.12 | 136.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 137.15 | 136.12 | 136.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 136.90 | 136.12 | 136.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 137.00 | 136.29 | 136.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:30:00 | 136.90 | 136.29 | 136.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 137.25 | 136.49 | 136.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:00:00 | 137.25 | 136.49 | 136.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 138.30 | 137.15 | 137.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 140.70 | 137.86 | 137.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 138.85 | 138.97 | 138.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 14:00:00 | 138.85 | 138.97 | 138.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 138.75 | 138.85 | 138.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 139.80 | 138.85 | 138.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 143.40 | 145.03 | 145.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 143.40 | 145.03 | 145.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 142.85 | 144.59 | 144.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 144.85 | 144.06 | 144.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 144.85 | 144.06 | 144.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 144.85 | 144.06 | 144.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 144.85 | 144.06 | 144.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 144.95 | 144.24 | 144.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 144.00 | 144.24 | 144.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 10:30:00 | 144.20 | 144.42 | 144.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 11:45:00 | 144.25 | 144.47 | 144.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 12:15:00 | 143.75 | 144.47 | 144.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 144.05 | 144.38 | 144.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:45:00 | 143.50 | 144.34 | 144.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 143.10 | 144.03 | 144.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 136.80 | 138.99 | 140.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 136.99 | 138.99 | 140.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 137.04 | 138.99 | 140.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 136.56 | 138.99 | 140.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 136.32 | 138.99 | 140.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 135.94 | 138.99 | 140.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 12:15:00 | 135.90 | 135.79 | 137.62 | SL hit (close>ema200) qty=0.50 sl=135.79 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 139.60 | 137.53 | 137.41 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 13:15:00 | 137.20 | 138.23 | 138.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 14:15:00 | 136.15 | 137.82 | 138.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 12:15:00 | 135.45 | 135.44 | 136.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 13:00:00 | 135.45 | 135.44 | 136.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 134.65 | 135.09 | 135.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 13:45:00 | 133.90 | 134.34 | 134.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 14:45:00 | 133.90 | 134.31 | 134.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 09:30:00 | 133.85 | 134.28 | 134.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 11:15:00 | 136.05 | 134.51 | 134.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 136.05 | 134.51 | 134.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 137.15 | 135.66 | 135.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 14:15:00 | 136.75 | 136.82 | 135.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 14:15:00 | 136.75 | 136.82 | 135.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 136.75 | 136.82 | 135.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:45:00 | 136.00 | 136.82 | 135.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 137.60 | 137.10 | 136.25 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 134.85 | 136.38 | 136.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 13:15:00 | 133.90 | 134.56 | 134.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 135.15 | 134.58 | 134.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 135.15 | 134.58 | 134.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 135.15 | 134.58 | 134.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:30:00 | 134.80 | 134.58 | 134.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 134.40 | 134.54 | 134.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:30:00 | 135.15 | 134.54 | 134.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 134.30 | 134.11 | 134.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:15:00 | 134.95 | 134.11 | 134.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 134.25 | 134.14 | 134.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:30:00 | 134.55 | 134.14 | 134.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 134.00 | 134.11 | 134.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:30:00 | 134.00 | 134.11 | 134.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 134.20 | 134.13 | 134.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:30:00 | 134.30 | 134.13 | 134.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 134.10 | 134.12 | 134.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 13:45:00 | 134.20 | 134.12 | 134.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 134.50 | 134.20 | 134.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 15:00:00 | 134.50 | 134.20 | 134.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 134.70 | 134.30 | 134.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:15:00 | 135.00 | 134.30 | 134.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 134.90 | 134.42 | 134.42 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 10:15:00 | 134.60 | 134.45 | 134.44 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 15:15:00 | 133.90 | 134.36 | 134.41 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 134.75 | 134.44 | 134.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 11:15:00 | 135.00 | 134.55 | 134.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 13:15:00 | 134.40 | 134.56 | 134.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 13:15:00 | 134.40 | 134.56 | 134.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 134.40 | 134.56 | 134.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 14:00:00 | 134.40 | 134.56 | 134.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 134.40 | 134.52 | 134.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 15:00:00 | 134.40 | 134.52 | 134.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 134.35 | 134.49 | 134.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 135.40 | 134.49 | 134.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 12:15:00 | 138.00 | 139.29 | 139.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 12:15:00 | 138.00 | 139.29 | 139.39 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 12:15:00 | 141.85 | 139.51 | 139.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 142.25 | 140.30 | 139.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 140.55 | 141.83 | 141.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 140.55 | 141.83 | 141.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 140.55 | 141.83 | 141.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 140.85 | 141.83 | 141.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 141.30 | 141.73 | 141.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 12:30:00 | 140.70 | 141.73 | 141.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 13:15:00 | 140.90 | 141.56 | 141.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 14:00:00 | 140.90 | 141.56 | 141.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 141.00 | 141.45 | 141.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 15:15:00 | 141.00 | 141.45 | 141.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 141.00 | 141.36 | 141.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:15:00 | 141.35 | 141.36 | 141.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 14:15:00 | 140.55 | 141.03 | 141.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 14:15:00 | 140.55 | 141.03 | 141.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 09:15:00 | 140.10 | 140.76 | 140.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 09:15:00 | 140.90 | 140.21 | 140.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 09:15:00 | 140.90 | 140.21 | 140.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 140.90 | 140.21 | 140.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:15:00 | 140.90 | 140.21 | 140.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 140.60 | 140.29 | 140.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:30:00 | 141.15 | 140.29 | 140.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 140.20 | 140.27 | 140.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:45:00 | 139.90 | 140.21 | 140.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 09:15:00 | 141.20 | 140.43 | 140.46 | SL hit (close>static) qty=1.00 sl=140.70 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 140.85 | 140.51 | 140.49 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 10:15:00 | 140.20 | 140.51 | 140.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 11:15:00 | 139.80 | 140.37 | 140.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 140.10 | 139.05 | 139.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 140.10 | 139.05 | 139.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 140.10 | 139.05 | 139.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:30:00 | 140.30 | 139.05 | 139.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 141.50 | 139.54 | 139.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:45:00 | 141.60 | 139.54 | 139.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 142.10 | 140.05 | 139.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 149.65 | 141.97 | 140.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 14:15:00 | 145.65 | 146.37 | 144.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 15:00:00 | 145.65 | 146.37 | 144.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 146.95 | 151.62 | 150.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 146.95 | 151.62 | 150.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 145.70 | 150.43 | 150.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 145.70 | 150.43 | 150.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 145.10 | 149.37 | 149.59 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 10:15:00 | 150.30 | 149.83 | 149.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 14:15:00 | 151.75 | 150.48 | 150.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 13:15:00 | 177.90 | 177.98 | 171.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 13:30:00 | 177.85 | 177.98 | 171.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 178.75 | 179.19 | 176.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 14:45:00 | 180.95 | 179.16 | 177.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 15:15:00 | 187.05 | 189.14 | 189.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 15:15:00 | 187.05 | 189.14 | 189.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 185.00 | 188.32 | 188.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 180.40 | 179.71 | 181.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 14:15:00 | 180.40 | 179.71 | 181.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 180.40 | 179.71 | 181.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 14:30:00 | 180.45 | 179.71 | 181.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 180.75 | 179.96 | 181.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:30:00 | 180.05 | 179.97 | 181.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 15:00:00 | 179.75 | 180.26 | 180.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 09:30:00 | 179.50 | 180.02 | 180.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 13:15:00 | 171.05 | 175.60 | 177.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 09:15:00 | 170.76 | 173.81 | 175.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 09:15:00 | 170.53 | 173.81 | 175.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-18 10:15:00 | 175.95 | 170.90 | 172.65 | SL hit (close>ema200) qty=0.50 sl=170.90 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 177.80 | 174.09 | 173.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 180.50 | 176.46 | 175.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 13:15:00 | 178.30 | 178.35 | 176.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-19 14:00:00 | 178.30 | 178.35 | 176.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 178.50 | 179.69 | 178.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 176.50 | 179.69 | 178.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 175.55 | 178.86 | 178.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 175.55 | 178.86 | 178.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 175.25 | 178.14 | 178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 174.20 | 178.14 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 173.55 | 177.22 | 177.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 171.80 | 175.46 | 176.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 176.10 | 174.08 | 175.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 176.10 | 174.08 | 175.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 176.10 | 174.08 | 175.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 176.10 | 174.08 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 176.50 | 174.57 | 175.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:15:00 | 176.60 | 174.57 | 175.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 176.65 | 175.65 | 175.64 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 14:15:00 | 175.30 | 175.64 | 175.66 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 176.20 | 175.72 | 175.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 180.80 | 176.73 | 176.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 190.95 | 191.53 | 186.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 190.95 | 191.53 | 186.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 192.40 | 194.48 | 191.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 11:45:00 | 192.40 | 194.48 | 191.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 192.50 | 194.09 | 191.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:15:00 | 191.20 | 194.09 | 191.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 13:15:00 | 190.85 | 193.44 | 191.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 13:45:00 | 190.75 | 193.44 | 191.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 191.00 | 192.95 | 191.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 14:30:00 | 191.10 | 192.95 | 191.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 199.10 | 200.00 | 198.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:30:00 | 196.45 | 200.00 | 198.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 197.95 | 199.59 | 198.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 198.00 | 199.59 | 198.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 201.20 | 199.91 | 198.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 10:30:00 | 204.10 | 200.69 | 198.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 09:15:00 | 202.60 | 201.94 | 200.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 14:15:00 | 198.55 | 199.98 | 199.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 14:15:00 | 198.55 | 199.98 | 199.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 196.70 | 198.54 | 199.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 11:15:00 | 198.55 | 197.32 | 198.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 11:15:00 | 198.55 | 197.32 | 198.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 198.55 | 197.32 | 198.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 12:00:00 | 198.55 | 197.32 | 198.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 12:15:00 | 201.55 | 198.17 | 198.60 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 13:15:00 | 202.90 | 199.11 | 198.99 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 11:15:00 | 194.60 | 198.24 | 198.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 09:15:00 | 188.95 | 194.83 | 196.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 15:15:00 | 192.50 | 192.21 | 194.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 09:15:00 | 192.05 | 192.21 | 194.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 194.55 | 192.67 | 194.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:45:00 | 194.70 | 192.67 | 194.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 194.25 | 192.99 | 194.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:45:00 | 195.00 | 192.99 | 194.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 196.05 | 193.60 | 194.47 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 196.60 | 195.12 | 194.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 201.00 | 196.30 | 195.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 14:15:00 | 198.00 | 198.62 | 197.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 15:00:00 | 198.00 | 198.62 | 197.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 207.45 | 208.92 | 207.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 207.45 | 208.92 | 207.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 207.50 | 208.64 | 207.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:15:00 | 207.50 | 208.64 | 207.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 207.50 | 208.41 | 207.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:15:00 | 206.30 | 208.41 | 207.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 207.00 | 208.13 | 207.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:15:00 | 204.80 | 208.13 | 207.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 205.35 | 207.57 | 207.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:45:00 | 205.15 | 207.57 | 207.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 205.15 | 207.09 | 207.07 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 12:15:00 | 205.30 | 206.73 | 206.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 203.60 | 206.10 | 206.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 204.55 | 204.08 | 205.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 11:45:00 | 203.15 | 204.08 | 205.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 202.30 | 203.72 | 204.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 201.55 | 204.00 | 204.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 09:15:00 | 206.30 | 201.70 | 202.52 | SL hit (close>static) qty=1.00 sl=205.30 alert=retest2 |

### Cycle 59 — BUY (started 2024-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 12:15:00 | 204.80 | 203.26 | 203.11 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 199.55 | 203.05 | 203.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 195.60 | 201.56 | 202.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 13:15:00 | 202.35 | 199.25 | 200.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 13:15:00 | 202.35 | 199.25 | 200.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 202.35 | 199.25 | 200.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:45:00 | 202.05 | 199.25 | 200.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 209.45 | 201.29 | 201.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 215.65 | 205.37 | 203.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 209.90 | 212.25 | 209.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 209.90 | 212.25 | 209.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 209.90 | 212.25 | 209.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:00:00 | 209.90 | 212.25 | 209.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 209.90 | 211.78 | 209.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 209.90 | 211.78 | 209.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 210.00 | 211.12 | 209.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:30:00 | 209.80 | 211.12 | 209.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 209.50 | 213.53 | 212.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 209.50 | 213.53 | 212.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 208.20 | 212.47 | 212.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:30:00 | 208.75 | 212.47 | 212.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 208.05 | 211.58 | 211.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 206.25 | 209.07 | 209.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 212.20 | 205.90 | 207.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 212.20 | 205.90 | 207.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 212.20 | 205.90 | 207.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:00:00 | 212.20 | 205.90 | 207.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 208.85 | 206.49 | 207.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 11:15:00 | 206.90 | 206.49 | 207.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 196.56 | 202.18 | 204.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-14 09:15:00 | 186.21 | 195.96 | 200.17 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 12:15:00 | 200.95 | 198.89 | 198.67 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 196.80 | 198.82 | 198.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 193.85 | 196.58 | 197.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 14:15:00 | 195.80 | 195.30 | 196.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 14:15:00 | 195.80 | 195.30 | 196.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 195.80 | 195.30 | 196.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:30:00 | 194.20 | 195.30 | 196.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 196.00 | 195.44 | 196.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 194.90 | 195.44 | 196.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 09:15:00 | 196.60 | 189.84 | 189.92 | SL hit (close>static) qty=1.00 sl=196.40 alert=retest2 |

### Cycle 65 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 196.70 | 191.21 | 190.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 201.70 | 195.43 | 192.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 208.25 | 209.82 | 205.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-03 15:00:00 | 208.25 | 209.82 | 205.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 208.15 | 208.71 | 206.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 13:30:00 | 209.40 | 208.70 | 206.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 09:45:00 | 210.50 | 209.40 | 207.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 209.80 | 208.86 | 208.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-10 09:15:00 | 230.34 | 220.29 | 215.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 11:15:00 | 214.30 | 219.64 | 219.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 12:15:00 | 213.35 | 218.38 | 219.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 215.95 | 215.29 | 217.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 215.95 | 215.29 | 217.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 215.95 | 215.29 | 217.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 209.95 | 213.27 | 214.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 10:15:00 | 210.70 | 211.22 | 213.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 10:15:00 | 209.80 | 208.96 | 209.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 09:15:00 | 215.40 | 210.05 | 209.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 215.40 | 210.05 | 209.92 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 210.60 | 212.29 | 212.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 13:15:00 | 209.85 | 211.45 | 211.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 13:15:00 | 197.25 | 196.50 | 200.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 14:00:00 | 197.25 | 196.50 | 200.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 200.25 | 197.25 | 200.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 15:00:00 | 200.25 | 197.25 | 200.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 199.20 | 197.64 | 200.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 199.40 | 197.64 | 200.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 196.90 | 197.49 | 199.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 10:15:00 | 195.80 | 197.49 | 199.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:00:00 | 196.05 | 197.20 | 199.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 15:15:00 | 186.25 | 190.05 | 192.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 193.75 | 190.79 | 192.93 | SL hit (close>ema200) qty=0.50 sl=190.79 alert=retest2 |

### Cycle 69 — BUY (started 2024-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 12:15:00 | 190.90 | 190.36 | 190.36 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 189.90 | 190.27 | 190.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 14:15:00 | 189.20 | 190.06 | 190.21 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 192.55 | 190.43 | 190.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 194.90 | 192.74 | 191.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 193.95 | 194.58 | 193.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 193.95 | 194.58 | 193.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 193.50 | 194.36 | 193.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 194.05 | 194.36 | 193.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 192.35 | 193.89 | 193.44 | SL hit (close<static) qty=1.00 sl=193.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 191.80 | 193.17 | 193.18 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 195.05 | 193.36 | 193.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 196.45 | 194.08 | 193.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 193.15 | 194.10 | 193.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 193.15 | 194.10 | 193.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 193.15 | 194.10 | 193.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 193.15 | 194.10 | 193.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 192.95 | 193.87 | 193.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 192.45 | 193.87 | 193.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 191.00 | 193.29 | 193.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 10:15:00 | 190.00 | 192.63 | 193.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 10:15:00 | 190.45 | 190.43 | 191.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 11:00:00 | 190.45 | 190.43 | 191.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 188.45 | 188.72 | 190.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:00:00 | 186.05 | 187.53 | 188.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:15:00 | 186.00 | 187.31 | 188.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 190.85 | 188.24 | 188.40 | SL hit (close>static) qty=1.00 sl=190.35 alert=retest2 |

### Cycle 75 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 191.75 | 188.94 | 188.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 195.35 | 191.00 | 189.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 12:15:00 | 193.40 | 193.57 | 191.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 13:00:00 | 193.40 | 193.57 | 191.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 192.10 | 193.02 | 191.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:15:00 | 192.60 | 193.02 | 191.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 192.60 | 192.94 | 191.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 187.55 | 192.94 | 191.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 189.80 | 192.31 | 191.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 188.05 | 192.31 | 191.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 183.20 | 190.49 | 190.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 173.70 | 187.13 | 189.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 187.45 | 184.65 | 186.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 187.45 | 184.65 | 186.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 187.45 | 184.65 | 186.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 187.45 | 184.65 | 186.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 187.05 | 185.13 | 186.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 14:30:00 | 185.50 | 185.94 | 186.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 189.35 | 186.71 | 186.99 | SL hit (close>static) qty=1.00 sl=189.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 189.90 | 187.35 | 187.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 193.00 | 189.09 | 188.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 200.10 | 201.39 | 198.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 200.10 | 201.39 | 198.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 202.41 | 203.60 | 202.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 202.41 | 203.60 | 202.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 203.10 | 203.50 | 202.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 202.41 | 203.50 | 202.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 205.30 | 203.86 | 202.89 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 202.57 | 203.34 | 203.37 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 09:15:00 | 205.51 | 203.71 | 203.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 210.94 | 207.01 | 205.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 209.70 | 209.89 | 208.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 209.70 | 209.89 | 208.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 213.80 | 210.86 | 209.25 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 207.68 | 210.40 | 210.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 206.67 | 208.72 | 209.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 204.40 | 203.96 | 205.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 204.40 | 203.96 | 205.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 204.40 | 203.96 | 205.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 201.91 | 203.70 | 205.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 209.00 | 203.36 | 204.34 | SL hit (close>static) qty=1.00 sl=207.37 alert=retest2 |

### Cycle 81 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 213.17 | 206.40 | 205.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 216.00 | 208.32 | 206.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 212.74 | 213.80 | 210.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 13:00:00 | 212.74 | 213.80 | 210.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 251.36 | 251.96 | 248.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 251.36 | 251.96 | 248.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 258.69 | 253.31 | 249.85 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 250.52 | 251.15 | 251.15 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 251.73 | 251.21 | 251.18 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 12:15:00 | 250.20 | 251.01 | 251.09 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 13:15:00 | 252.66 | 251.34 | 251.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 255.87 | 252.48 | 251.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 266.85 | 268.45 | 263.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 266.85 | 268.45 | 263.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 266.85 | 268.45 | 263.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 264.53 | 268.45 | 263.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 263.91 | 267.55 | 263.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 10:30:00 | 263.88 | 267.55 | 263.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 261.91 | 266.42 | 263.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:45:00 | 262.00 | 266.42 | 263.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 261.39 | 265.41 | 263.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 261.39 | 265.41 | 263.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 250.53 | 260.61 | 261.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 249.96 | 255.94 | 259.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 252.88 | 251.77 | 255.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:00:00 | 252.88 | 251.77 | 255.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 252.16 | 247.78 | 249.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 251.67 | 247.78 | 249.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 267.59 | 251.74 | 251.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 272.00 | 262.71 | 257.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 268.10 | 269.47 | 264.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 10:00:00 | 268.10 | 269.47 | 264.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 265.75 | 267.69 | 265.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 267.70 | 267.69 | 265.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 264.22 | 270.10 | 269.25 | SL hit (close<static) qty=1.00 sl=264.55 alert=retest2 |

### Cycle 88 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 265.53 | 268.11 | 268.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 264.61 | 267.41 | 268.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 09:15:00 | 264.85 | 260.57 | 263.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 264.85 | 260.57 | 263.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 264.85 | 260.57 | 263.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 264.85 | 260.57 | 263.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 264.70 | 261.40 | 263.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:30:00 | 264.35 | 261.40 | 263.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 262.00 | 261.52 | 263.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:45:00 | 263.30 | 261.52 | 263.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 254.90 | 251.82 | 255.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:00:00 | 248.85 | 251.23 | 255.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 14:15:00 | 255.90 | 252.76 | 252.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 255.90 | 252.76 | 252.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 257.60 | 254.07 | 253.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 256.00 | 256.91 | 255.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 256.00 | 256.91 | 255.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 257.70 | 257.07 | 255.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:30:00 | 256.00 | 257.07 | 255.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 256.80 | 257.01 | 255.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 256.80 | 257.01 | 255.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 255.75 | 256.76 | 255.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 259.10 | 256.76 | 255.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 257.35 | 256.88 | 256.02 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 255.30 | 256.14 | 256.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 11:15:00 | 253.95 | 255.70 | 255.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 250.50 | 247.87 | 250.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 250.50 | 247.87 | 250.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 250.50 | 247.87 | 250.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 249.15 | 247.87 | 250.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 251.50 | 248.60 | 250.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 251.05 | 248.60 | 250.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 249.80 | 248.84 | 250.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:15:00 | 248.85 | 249.53 | 250.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 253.70 | 251.01 | 250.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 253.70 | 251.01 | 250.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 255.20 | 252.14 | 251.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 13:15:00 | 252.60 | 252.67 | 251.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 14:00:00 | 252.60 | 252.67 | 251.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 252.50 | 252.80 | 252.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 252.50 | 252.80 | 252.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 252.25 | 252.69 | 252.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 251.80 | 252.69 | 252.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 251.45 | 252.44 | 252.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 251.45 | 252.44 | 252.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 251.10 | 252.17 | 252.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:30:00 | 250.35 | 252.17 | 252.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 14:15:00 | 250.85 | 251.91 | 251.93 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 264.40 | 254.30 | 253.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 269.40 | 257.32 | 254.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 268.45 | 269.15 | 265.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 268.45 | 269.15 | 265.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 266.70 | 268.14 | 266.07 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 263.55 | 265.28 | 265.42 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 280.85 | 267.79 | 266.47 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 269.65 | 271.61 | 271.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 267.00 | 270.68 | 271.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 271.70 | 265.00 | 265.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 271.70 | 265.00 | 265.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 271.70 | 265.00 | 265.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 273.30 | 265.00 | 265.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 271.75 | 266.35 | 266.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:30:00 | 272.40 | 266.35 | 266.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 271.80 | 267.44 | 266.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 276.80 | 270.24 | 268.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 13:15:00 | 271.65 | 272.33 | 270.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 14:00:00 | 271.65 | 272.33 | 270.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 268.85 | 271.67 | 270.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:00:00 | 268.85 | 271.67 | 270.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 268.90 | 271.11 | 270.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:15:00 | 267.35 | 271.11 | 270.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 266.95 | 270.28 | 270.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 266.60 | 269.41 | 269.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 262.35 | 259.34 | 262.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 262.35 | 259.34 | 262.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 262.35 | 259.34 | 262.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 258.40 | 259.56 | 262.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:15:00 | 258.60 | 259.36 | 260.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:00:00 | 258.60 | 259.20 | 260.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:30:00 | 258.20 | 257.55 | 259.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 259.45 | 257.93 | 259.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 259.85 | 257.93 | 259.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 259.20 | 258.18 | 259.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:45:00 | 259.35 | 258.18 | 259.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 261.75 | 258.90 | 259.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 261.75 | 258.90 | 259.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 261.40 | 259.40 | 259.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:00:00 | 260.20 | 259.56 | 259.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 15:15:00 | 261.20 | 259.89 | 259.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 261.20 | 259.89 | 259.80 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 258.85 | 259.72 | 259.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 256.45 | 258.74 | 259.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 15:15:00 | 258.20 | 257.33 | 258.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 15:15:00 | 258.20 | 257.33 | 258.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 258.20 | 257.33 | 258.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 258.85 | 257.33 | 258.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 255.20 | 256.91 | 257.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 12:00:00 | 253.60 | 255.99 | 257.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 14:45:00 | 253.50 | 255.29 | 256.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:00:00 | 254.00 | 255.11 | 256.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:30:00 | 253.75 | 252.76 | 254.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 249.75 | 248.14 | 250.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-23 13:15:00 | 251.35 | 251.01 | 251.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 251.35 | 251.01 | 251.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 254.40 | 252.09 | 251.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 12:15:00 | 252.95 | 252.98 | 252.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:45:00 | 252.65 | 252.98 | 252.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 252.00 | 252.78 | 252.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 252.00 | 252.78 | 252.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 249.90 | 252.21 | 251.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 249.90 | 252.21 | 251.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 250.00 | 251.77 | 251.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 251.60 | 251.77 | 251.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 248.20 | 251.05 | 251.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 13:15:00 | 247.25 | 249.10 | 250.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 250.80 | 248.17 | 249.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 11:15:00 | 250.80 | 248.17 | 249.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 250.80 | 248.17 | 249.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:30:00 | 251.50 | 248.17 | 249.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 248.00 | 248.14 | 249.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:15:00 | 246.50 | 248.14 | 249.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 14:00:00 | 247.00 | 247.91 | 248.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 246.75 | 246.96 | 248.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 247.25 | 247.21 | 248.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 247.80 | 247.33 | 248.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:15:00 | 246.90 | 247.33 | 248.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 234.65 | 241.40 | 243.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 234.41 | 241.40 | 243.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 234.89 | 241.40 | 243.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:15:00 | 234.56 | 241.40 | 243.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 234.17 | 240.15 | 242.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 10:15:00 | 221.85 | 230.32 | 235.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 230.68 | 227.74 | 227.50 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 226.63 | 227.66 | 227.74 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 228.93 | 227.95 | 227.85 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 226.33 | 228.21 | 228.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 12:15:00 | 224.69 | 227.12 | 227.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 221.69 | 220.94 | 222.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 15:00:00 | 221.69 | 220.94 | 222.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 222.25 | 221.20 | 222.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 219.79 | 221.20 | 222.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 208.80 | 213.89 | 217.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 210.78 | 210.46 | 213.99 | SL hit (close>ema200) qty=0.50 sl=210.46 alert=retest2 |

### Cycle 107 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 209.70 | 207.66 | 207.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 212.73 | 209.05 | 208.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 211.79 | 212.02 | 210.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 211.79 | 212.02 | 210.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 211.79 | 212.02 | 210.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 214.69 | 211.56 | 210.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 210.10 | 211.77 | 211.08 | SL hit (close<static) qty=1.00 sl=210.35 alert=retest2 |

### Cycle 108 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 209.09 | 210.70 | 210.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 207.86 | 210.13 | 210.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 206.20 | 205.33 | 207.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:45:00 | 206.37 | 205.33 | 207.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 206.35 | 205.53 | 207.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 206.35 | 205.53 | 207.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 208.74 | 206.33 | 207.34 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 211.85 | 208.33 | 208.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 215.85 | 209.84 | 208.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 212.51 | 213.08 | 211.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 13:30:00 | 212.03 | 213.08 | 211.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 211.20 | 212.71 | 211.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 211.20 | 212.71 | 211.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 211.70 | 212.51 | 211.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 211.06 | 212.51 | 211.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 210.35 | 212.08 | 211.37 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 206.97 | 210.53 | 210.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 205.21 | 207.81 | 209.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 193.63 | 193.40 | 196.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 193.63 | 193.40 | 196.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 193.22 | 191.24 | 193.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 190.75 | 191.85 | 192.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 196.20 | 190.42 | 190.66 | SL hit (close>static) qty=1.00 sl=194.20 alert=retest2 |

### Cycle 111 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 195.09 | 191.35 | 191.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 198.75 | 193.40 | 192.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 196.70 | 197.08 | 194.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 196.70 | 197.08 | 194.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 192.99 | 196.12 | 194.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 192.99 | 196.12 | 194.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 193.00 | 195.50 | 194.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 195.70 | 195.50 | 194.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 14:15:00 | 215.27 | 209.21 | 204.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 216.87 | 218.92 | 219.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 11:15:00 | 215.33 | 217.51 | 218.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 210.50 | 210.44 | 212.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 11:45:00 | 210.68 | 210.44 | 212.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 212.79 | 210.91 | 212.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 212.79 | 210.91 | 212.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 211.04 | 210.93 | 212.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 208.73 | 210.02 | 211.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 208.80 | 209.71 | 210.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:00:00 | 209.07 | 209.58 | 210.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 198.29 | 203.19 | 205.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 198.36 | 203.19 | 205.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 198.62 | 203.19 | 205.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 203.90 | 203.33 | 205.58 | SL hit (close>ema200) qty=0.50 sl=203.33 alert=retest2 |

### Cycle 113 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 203.38 | 196.95 | 196.15 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 196.48 | 200.95 | 201.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 195.19 | 198.97 | 200.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 196.54 | 196.47 | 197.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 14:00:00 | 196.54 | 196.47 | 197.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 199.23 | 196.76 | 197.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 197.58 | 197.31 | 197.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:45:00 | 197.40 | 197.41 | 197.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 200.70 | 198.23 | 198.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 14:15:00 | 200.70 | 198.23 | 198.13 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 192.80 | 197.88 | 198.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 15:15:00 | 189.00 | 193.24 | 195.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 183.70 | 183.59 | 187.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 183.70 | 183.59 | 187.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 184.29 | 183.25 | 184.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 15:00:00 | 183.88 | 184.08 | 184.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 186.72 | 184.59 | 184.69 | SL hit (close>static) qty=1.00 sl=186.51 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 14:15:00 | 184.77 | 184.75 | 184.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 186.25 | 185.23 | 184.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 184.60 | 185.56 | 185.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 184.60 | 185.56 | 185.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 184.60 | 185.56 | 185.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 184.60 | 185.56 | 185.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 185.10 | 185.47 | 185.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 185.03 | 185.47 | 185.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 184.03 | 185.18 | 185.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:45:00 | 184.29 | 185.18 | 185.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 183.81 | 184.91 | 185.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 182.72 | 184.47 | 184.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 181.25 | 179.69 | 181.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 181.25 | 179.69 | 181.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 181.25 | 179.69 | 181.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 181.11 | 179.69 | 181.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 181.93 | 180.14 | 181.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 181.99 | 180.14 | 181.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 180.33 | 180.17 | 181.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:45:00 | 179.69 | 180.16 | 181.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 179.35 | 180.11 | 181.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 179.50 | 179.28 | 180.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 170.71 | 175.45 | 177.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 170.38 | 175.45 | 177.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 170.53 | 175.45 | 177.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 170.51 | 169.30 | 172.51 | SL hit (close>ema200) qty=0.50 sl=169.30 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 175.89 | 173.11 | 172.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 177.02 | 174.30 | 173.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 174.60 | 175.07 | 174.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 174.60 | 175.07 | 174.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 174.60 | 175.07 | 174.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 174.41 | 175.07 | 174.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 174.99 | 175.06 | 174.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 175.13 | 175.06 | 174.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 177.50 | 175.52 | 174.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 179.11 | 176.99 | 175.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 178.91 | 177.87 | 176.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 12:15:00 | 174.01 | 176.27 | 176.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 174.01 | 176.27 | 176.47 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 187.63 | 178.50 | 177.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 199.70 | 187.92 | 183.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 10:15:00 | 208.25 | 208.96 | 204.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 11:00:00 | 208.25 | 208.96 | 204.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 207.23 | 208.81 | 207.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:15:00 | 206.93 | 208.81 | 207.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 207.03 | 208.45 | 207.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 207.03 | 208.45 | 207.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 207.30 | 208.22 | 207.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:15:00 | 208.82 | 207.47 | 206.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 13:00:00 | 208.96 | 208.26 | 207.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 210.17 | 207.52 | 207.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 14:15:00 | 206.50 | 207.41 | 207.40 | SL hit (close<static) qty=1.00 sl=206.55 alert=retest2 |

### Cycle 122 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 206.65 | 207.26 | 207.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 204.95 | 206.80 | 207.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 200.71 | 199.45 | 202.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 200.71 | 199.45 | 202.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 200.71 | 199.45 | 202.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 201.52 | 199.45 | 202.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 201.78 | 200.07 | 201.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 201.78 | 200.07 | 201.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 203.44 | 200.75 | 202.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 203.50 | 200.75 | 202.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 209.18 | 202.43 | 202.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 209.18 | 202.43 | 202.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 207.00 | 203.35 | 203.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 210.86 | 207.30 | 205.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 13:15:00 | 207.95 | 207.96 | 206.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-19 13:45:00 | 207.23 | 207.96 | 206.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 212.02 | 208.64 | 207.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 11:00:00 | 212.85 | 209.48 | 207.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:45:00 | 213.09 | 214.24 | 212.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 12:15:00 | 210.82 | 216.08 | 216.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 12:15:00 | 210.82 | 216.08 | 216.15 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 218.57 | 215.39 | 215.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 221.56 | 217.07 | 216.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 240.99 | 241.48 | 234.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 240.99 | 241.48 | 234.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 232.97 | 239.29 | 236.31 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 11:15:00 | 235.57 | 237.46 | 237.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 232.41 | 236.45 | 237.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 13:15:00 | 234.60 | 233.04 | 234.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 13:15:00 | 234.60 | 233.04 | 234.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 234.60 | 233.04 | 234.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 234.60 | 233.04 | 234.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 235.84 | 233.60 | 234.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 235.00 | 233.60 | 234.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 236.69 | 234.22 | 234.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 222.24 | 234.22 | 234.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 11:15:00 | 211.13 | 215.58 | 217.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 209.07 | 208.17 | 210.97 | SL hit (close>ema200) qty=0.50 sl=208.17 alert=retest2 |

### Cycle 127 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 198.58 | 196.31 | 196.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 201.30 | 198.93 | 197.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 202.85 | 204.82 | 203.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 202.85 | 204.82 | 203.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 202.85 | 204.82 | 203.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:15:00 | 202.74 | 204.82 | 203.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 203.23 | 204.51 | 203.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 203.66 | 204.51 | 203.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 15:15:00 | 201.99 | 203.42 | 203.07 | SL hit (close<static) qty=1.00 sl=202.10 alert=retest2 |

### Cycle 128 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 202.63 | 206.52 | 206.93 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 208.02 | 206.40 | 206.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 209.26 | 206.97 | 206.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 205.00 | 206.93 | 206.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 205.00 | 206.93 | 206.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 205.00 | 206.93 | 206.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 205.60 | 206.93 | 206.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 205.22 | 206.59 | 206.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 203.99 | 206.59 | 206.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 204.68 | 206.21 | 206.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 204.01 | 205.77 | 206.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 199.85 | 199.82 | 201.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 14:45:00 | 200.10 | 199.82 | 201.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 204.38 | 200.44 | 201.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 204.38 | 200.44 | 201.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 205.42 | 201.44 | 201.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 205.42 | 201.44 | 201.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 204.20 | 201.99 | 201.91 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 200.01 | 202.05 | 202.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 199.60 | 200.98 | 201.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 199.10 | 198.96 | 199.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 201.28 | 198.96 | 199.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 202.27 | 199.62 | 200.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:45:00 | 202.60 | 199.62 | 200.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 201.86 | 200.07 | 200.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:30:00 | 202.10 | 200.07 | 200.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 201.85 | 200.43 | 200.41 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 199.49 | 200.24 | 200.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 199.07 | 200.00 | 200.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 09:15:00 | 200.12 | 199.74 | 200.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 200.12 | 199.74 | 200.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 200.12 | 199.74 | 200.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 200.12 | 199.74 | 200.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 199.03 | 199.60 | 199.93 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 203.94 | 200.47 | 200.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 206.44 | 205.07 | 203.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 211.26 | 211.52 | 210.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 211.26 | 211.52 | 210.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 211.32 | 211.39 | 210.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 209.60 | 211.39 | 210.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 210.04 | 211.12 | 210.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 209.11 | 211.12 | 210.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 209.86 | 210.87 | 210.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 209.87 | 210.87 | 210.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 207.31 | 209.80 | 209.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 206.70 | 209.18 | 209.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 209.70 | 208.88 | 209.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 209.70 | 208.88 | 209.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 209.70 | 208.88 | 209.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 209.25 | 208.88 | 209.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 208.72 | 208.85 | 209.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 208.00 | 208.68 | 209.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 208.27 | 208.66 | 208.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 208.46 | 208.71 | 208.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 208.31 | 208.80 | 208.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 208.77 | 208.73 | 208.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 208.77 | 208.73 | 208.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 208.44 | 208.64 | 208.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 208.45 | 208.64 | 208.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 208.41 | 208.59 | 208.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 208.80 | 208.59 | 208.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 207.91 | 208.36 | 208.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 209.25 | 208.36 | 208.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 208.03 | 208.10 | 208.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 208.48 | 208.10 | 208.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 208.00 | 208.08 | 208.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 208.00 | 208.08 | 208.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 207.60 | 207.95 | 208.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 206.87 | 207.51 | 207.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:45:00 | 207.04 | 207.30 | 207.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:15:00 | 206.90 | 207.27 | 207.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 217.40 | 209.02 | 208.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 217.40 | 209.02 | 208.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 222.25 | 217.39 | 215.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 217.72 | 218.83 | 217.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 217.72 | 218.83 | 217.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 217.72 | 218.83 | 217.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 217.72 | 218.83 | 217.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 217.98 | 218.66 | 217.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 216.77 | 218.66 | 217.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 217.80 | 218.49 | 217.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 218.94 | 218.51 | 217.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 218.78 | 218.56 | 217.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 218.69 | 218.60 | 217.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 218.54 | 218.55 | 217.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 217.26 | 218.29 | 217.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 217.26 | 218.29 | 217.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 217.65 | 218.16 | 217.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 217.86 | 218.16 | 217.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 218.96 | 219.43 | 218.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 218.96 | 219.43 | 218.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 218.93 | 219.24 | 218.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 216.08 | 219.24 | 218.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 217.72 | 218.94 | 218.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 217.35 | 218.34 | 218.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 217.35 | 218.34 | 218.38 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 219.50 | 218.40 | 218.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 219.72 | 218.85 | 218.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 220.83 | 222.10 | 220.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 220.83 | 222.10 | 220.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 220.83 | 222.10 | 220.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 220.83 | 222.10 | 220.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 220.30 | 221.74 | 220.88 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 217.41 | 220.40 | 220.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 214.89 | 218.06 | 219.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 212.77 | 211.46 | 213.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 212.77 | 211.46 | 213.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 212.77 | 211.46 | 213.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 212.77 | 211.46 | 213.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 212.63 | 211.69 | 213.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 213.80 | 211.69 | 213.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 212.80 | 211.97 | 213.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 213.36 | 211.97 | 213.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 212.75 | 212.12 | 213.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 211.89 | 212.04 | 213.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 211.99 | 207.59 | 207.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 211.99 | 207.59 | 207.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 213.91 | 211.66 | 210.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 222.09 | 223.68 | 221.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 222.09 | 223.68 | 221.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 221.57 | 223.26 | 221.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 221.57 | 223.26 | 221.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 223.25 | 223.25 | 221.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 223.25 | 223.25 | 221.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 222.80 | 224.11 | 222.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 222.30 | 224.11 | 222.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 221.78 | 223.64 | 222.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 221.78 | 223.64 | 222.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 221.31 | 222.45 | 222.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 12:15:00 | 220.62 | 221.85 | 222.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 221.50 | 221.46 | 221.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 221.80 | 221.46 | 221.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 221.76 | 221.52 | 221.89 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 222.19 | 222.12 | 222.11 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 222.00 | 222.10 | 222.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 220.25 | 221.70 | 221.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 222.99 | 220.31 | 220.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 222.99 | 220.31 | 220.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 222.99 | 220.31 | 220.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 222.99 | 220.31 | 220.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 223.25 | 220.89 | 221.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 223.25 | 220.89 | 221.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 221.94 | 221.38 | 221.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 223.10 | 221.72 | 221.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 221.10 | 222.04 | 221.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 221.10 | 222.04 | 221.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 221.10 | 222.04 | 221.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 221.14 | 222.04 | 221.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 220.20 | 221.67 | 221.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 220.11 | 221.67 | 221.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 220.22 | 221.38 | 221.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 219.52 | 220.80 | 221.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 220.50 | 220.37 | 220.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 220.50 | 220.37 | 220.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 220.50 | 220.37 | 220.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 220.50 | 220.37 | 220.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 220.16 | 220.33 | 220.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 220.00 | 220.33 | 220.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 229.12 | 222.01 | 221.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 231.09 | 223.83 | 222.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 12:15:00 | 224.77 | 224.83 | 223.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 13:00:00 | 224.77 | 224.83 | 223.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 224.10 | 224.68 | 223.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 224.10 | 224.68 | 223.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 222.86 | 224.32 | 223.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 222.86 | 224.32 | 223.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 222.90 | 224.03 | 223.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 226.18 | 224.03 | 223.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 223.38 | 224.46 | 223.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:45:00 | 223.28 | 224.13 | 223.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 222.79 | 223.62 | 223.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 222.79 | 223.62 | 223.64 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 224.50 | 223.60 | 223.60 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 221.51 | 223.30 | 223.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 220.77 | 222.80 | 223.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 221.80 | 221.55 | 222.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:15:00 | 219.81 | 221.55 | 222.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 221.80 | 220.54 | 221.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 221.80 | 220.54 | 221.19 | SL hit (close>ema400) qty=1.00 sl=221.19 alert=retest1 |

### Cycle 151 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 225.00 | 221.16 | 221.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 225.25 | 222.58 | 221.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 227.12 | 227.20 | 225.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:00:00 | 227.12 | 227.20 | 225.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 225.52 | 227.11 | 225.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 225.52 | 227.11 | 225.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 224.05 | 226.49 | 225.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 224.68 | 226.49 | 225.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 221.69 | 224.57 | 224.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 221.40 | 222.96 | 223.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 220.39 | 220.26 | 221.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 220.39 | 220.26 | 221.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 222.21 | 220.72 | 221.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 222.21 | 220.72 | 221.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 222.45 | 221.06 | 221.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 221.73 | 221.06 | 221.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 220.83 | 220.86 | 221.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 221.64 | 220.86 | 221.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 220.96 | 220.88 | 221.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 220.96 | 220.88 | 221.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 222.39 | 219.77 | 220.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 222.39 | 219.77 | 220.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 221.40 | 220.10 | 220.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 220.46 | 220.06 | 220.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 221.78 | 219.78 | 219.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 221.78 | 219.78 | 219.75 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 218.57 | 220.29 | 220.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 215.50 | 219.34 | 219.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 214.40 | 213.35 | 215.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 214.84 | 213.35 | 215.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 214.60 | 213.83 | 215.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 212.85 | 213.62 | 215.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 212.85 | 213.46 | 214.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 212.80 | 213.42 | 214.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 12:15:00 | 207.89 | 207.11 | 207.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 207.89 | 207.11 | 207.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 14:15:00 | 209.05 | 207.52 | 207.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 208.68 | 209.01 | 208.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 14:15:00 | 208.68 | 209.01 | 208.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 208.68 | 209.01 | 208.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 208.68 | 209.01 | 208.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 208.65 | 208.94 | 208.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 207.94 | 208.94 | 208.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 207.19 | 208.59 | 208.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 207.05 | 208.59 | 208.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 206.76 | 208.22 | 208.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 206.58 | 208.22 | 208.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 206.70 | 207.92 | 208.00 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 208.65 | 207.83 | 207.77 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 207.32 | 207.72 | 207.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 205.51 | 207.28 | 207.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 199.13 | 198.76 | 200.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 200.45 | 199.31 | 200.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 200.45 | 199.31 | 200.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 200.45 | 199.31 | 200.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 200.88 | 199.62 | 200.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 201.03 | 199.62 | 200.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 201.40 | 199.98 | 200.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 201.40 | 199.98 | 200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 200.70 | 200.14 | 200.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 200.70 | 200.14 | 200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 200.25 | 200.16 | 200.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 198.10 | 200.16 | 200.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 199.00 | 198.01 | 197.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 199.00 | 198.01 | 197.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 200.15 | 198.60 | 198.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 198.09 | 199.28 | 198.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 198.09 | 199.28 | 198.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 198.09 | 199.28 | 198.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 197.84 | 199.28 | 198.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 197.75 | 198.98 | 198.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 197.75 | 198.98 | 198.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 197.37 | 198.66 | 198.66 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 199.25 | 198.46 | 198.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 200.41 | 199.17 | 198.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 204.00 | 204.38 | 202.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 15:15:00 | 203.49 | 203.80 | 203.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 203.49 | 203.80 | 203.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 203.37 | 203.80 | 203.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 203.10 | 203.66 | 203.09 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 202.53 | 202.96 | 202.97 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 203.10 | 202.99 | 202.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 204.22 | 203.24 | 203.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 202.80 | 203.21 | 203.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 202.80 | 203.21 | 203.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 202.80 | 203.21 | 203.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 203.08 | 203.21 | 203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 202.70 | 203.11 | 203.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 203.70 | 203.11 | 203.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 202.60 | 203.01 | 203.04 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 203.45 | 203.03 | 203.02 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 201.84 | 202.86 | 202.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 200.84 | 202.45 | 202.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 202.14 | 201.95 | 202.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 202.14 | 201.95 | 202.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 202.80 | 202.12 | 202.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 202.80 | 202.12 | 202.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 202.25 | 202.14 | 202.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 200.85 | 202.14 | 202.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 199.76 | 198.26 | 198.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 199.76 | 198.26 | 198.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 201.02 | 199.35 | 198.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 201.80 | 202.16 | 201.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 201.80 | 202.16 | 201.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 203.00 | 202.29 | 201.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 203.39 | 202.27 | 201.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 201.14 | 202.07 | 201.88 | SL hit (close<static) qty=1.00 sl=201.41 alert=retest2 |

### Cycle 168 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 201.20 | 201.66 | 201.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 13:15:00 | 199.85 | 200.74 | 201.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 201.29 | 200.64 | 200.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 201.29 | 200.64 | 200.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 201.29 | 200.64 | 200.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 201.29 | 200.64 | 200.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 200.93 | 200.70 | 200.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 200.25 | 200.74 | 200.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 200.40 | 200.51 | 200.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 199.95 | 200.52 | 200.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 199.90 | 200.40 | 200.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 199.66 | 199.92 | 200.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 199.31 | 199.92 | 200.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 10:45:00 | 199.35 | 199.18 | 199.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 200.95 | 199.60 | 199.65 | SL hit (close>static) qty=1.00 sl=200.64 alert=retest2 |

### Cycle 169 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 203.88 | 200.46 | 200.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 204.55 | 201.27 | 200.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 202.06 | 202.18 | 201.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:45:00 | 202.03 | 202.18 | 201.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 200.44 | 201.83 | 201.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 200.44 | 201.83 | 201.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 199.91 | 201.45 | 201.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 199.91 | 201.45 | 201.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 199.19 | 200.77 | 200.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 199.10 | 200.43 | 200.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 199.86 | 199.28 | 199.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 199.86 | 199.28 | 199.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 199.86 | 199.28 | 199.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 199.82 | 199.28 | 199.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 199.88 | 199.40 | 199.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:00:00 | 199.88 | 199.40 | 199.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 200.26 | 199.57 | 199.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 200.40 | 199.57 | 199.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 200.38 | 199.74 | 199.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 200.63 | 199.74 | 199.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 200.50 | 199.94 | 199.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 200.91 | 199.94 | 199.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 200.48 | 200.05 | 199.99 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 199.18 | 199.98 | 200.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 198.73 | 199.36 | 199.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 13:15:00 | 199.02 | 198.59 | 198.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 199.02 | 198.59 | 198.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 199.02 | 198.59 | 198.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 198.73 | 198.59 | 198.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 198.01 | 198.48 | 198.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 196.96 | 198.15 | 198.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 197.92 | 197.94 | 198.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 197.71 | 197.88 | 198.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 188.02 | 191.02 | 192.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 191.00 | 190.89 | 192.26 | SL hit (close>ema200) qty=0.50 sl=190.89 alert=retest2 |

### Cycle 173 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 194.39 | 191.21 | 191.12 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 191.44 | 192.38 | 192.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 191.29 | 192.16 | 192.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 191.14 | 191.13 | 191.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 191.14 | 191.13 | 191.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 191.14 | 191.13 | 191.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 191.92 | 191.13 | 191.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 191.83 | 191.27 | 191.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 191.92 | 191.27 | 191.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 191.07 | 191.23 | 191.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:30:00 | 191.80 | 191.23 | 191.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 191.62 | 189.88 | 190.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 191.62 | 189.88 | 190.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 191.64 | 190.23 | 190.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 192.12 | 190.23 | 190.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 190.90 | 190.57 | 190.57 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 13:15:00 | 190.93 | 190.64 | 190.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 194.50 | 191.41 | 190.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 191.18 | 191.86 | 191.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 191.18 | 191.86 | 191.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 191.18 | 191.86 | 191.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 190.97 | 191.86 | 191.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 191.00 | 191.69 | 191.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 191.00 | 191.69 | 191.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 191.05 | 191.56 | 191.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:30:00 | 190.87 | 191.56 | 191.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 190.75 | 191.27 | 191.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:30:00 | 190.75 | 191.27 | 191.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 191.50 | 191.36 | 191.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 191.81 | 191.36 | 191.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 192.24 | 191.54 | 191.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:30:00 | 192.46 | 191.71 | 191.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 192.50 | 191.71 | 191.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 11:45:00 | 192.49 | 191.82 | 191.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 13:15:00 | 190.87 | 191.58 | 191.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 190.87 | 191.58 | 191.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 190.29 | 191.20 | 191.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 192.19 | 191.33 | 191.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 11:15:00 | 192.19 | 191.33 | 191.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 192.19 | 191.33 | 191.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 191.90 | 191.33 | 191.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 191.80 | 191.42 | 191.45 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 192.20 | 191.58 | 191.52 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 191.06 | 191.48 | 191.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 190.90 | 191.28 | 191.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 190.96 | 190.67 | 190.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 190.96 | 190.67 | 190.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 190.96 | 190.67 | 190.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 190.96 | 190.67 | 190.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 192.00 | 190.93 | 191.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 191.41 | 190.93 | 191.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 190.89 | 190.93 | 191.05 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 191.61 | 191.14 | 191.13 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 190.70 | 191.04 | 191.09 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 192.01 | 191.24 | 191.17 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 190.62 | 191.06 | 191.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 190.22 | 190.79 | 190.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 189.12 | 188.21 | 189.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 14:15:00 | 189.12 | 188.21 | 189.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 189.12 | 188.21 | 189.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 189.12 | 188.21 | 189.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 188.65 | 188.30 | 188.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 187.84 | 188.30 | 188.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 187.89 | 188.22 | 188.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 187.90 | 188.22 | 188.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 183.52 | 183.18 | 183.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 183.52 | 183.18 | 183.16 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 09:15:00 | 182.56 | 183.06 | 183.11 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 183.29 | 182.94 | 182.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 183.73 | 183.10 | 182.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 184.01 | 185.20 | 184.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 184.01 | 185.20 | 184.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 184.01 | 185.20 | 184.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 184.01 | 185.20 | 184.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 184.25 | 185.01 | 184.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:15:00 | 184.55 | 185.01 | 184.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 185.22 | 185.06 | 184.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 186.45 | 185.06 | 184.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 189.05 | 189.75 | 189.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 189.05 | 189.75 | 189.77 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 190.15 | 189.83 | 189.80 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 14:15:00 | 188.75 | 189.61 | 189.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 15:15:00 | 188.58 | 189.41 | 189.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 189.78 | 189.39 | 189.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 189.78 | 189.39 | 189.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 189.78 | 189.39 | 189.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 190.38 | 189.39 | 189.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 189.71 | 189.45 | 189.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 189.89 | 189.45 | 189.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 189.49 | 189.46 | 189.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 189.64 | 189.46 | 189.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 189.57 | 189.48 | 189.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 189.60 | 189.48 | 189.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 192.38 | 190.06 | 189.82 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 190.52 | 191.19 | 191.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 190.44 | 190.88 | 191.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 190.10 | 189.87 | 190.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 14:15:00 | 190.10 | 189.87 | 190.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 190.10 | 189.87 | 190.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 190.10 | 189.87 | 190.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 189.70 | 189.83 | 190.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 189.52 | 189.83 | 190.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 189.03 | 189.67 | 190.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 188.70 | 189.47 | 190.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:30:00 | 188.79 | 189.31 | 189.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 188.82 | 189.19 | 189.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 188.88 | 189.09 | 189.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 187.35 | 188.41 | 188.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 185.91 | 188.08 | 188.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 186.59 | 187.63 | 187.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 185.40 | 184.39 | 184.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 185.40 | 184.39 | 184.30 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 183.85 | 184.90 | 184.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 183.70 | 184.66 | 184.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 184.00 | 183.95 | 184.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 184.00 | 183.95 | 184.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 184.00 | 183.95 | 184.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 184.43 | 183.95 | 184.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 184.30 | 184.02 | 184.33 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 185.29 | 184.59 | 184.54 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 184.21 | 184.81 | 184.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 183.28 | 184.11 | 184.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 183.50 | 183.22 | 183.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 183.50 | 183.22 | 183.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 183.50 | 183.22 | 183.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 183.50 | 183.22 | 183.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 184.28 | 183.43 | 183.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:30:00 | 184.70 | 183.43 | 183.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 185.02 | 183.75 | 183.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 185.02 | 183.75 | 183.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 184.99 | 184.12 | 184.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 188.80 | 186.24 | 185.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 186.50 | 186.57 | 185.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 185.61 | 186.57 | 185.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 186.14 | 186.48 | 185.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 186.50 | 186.48 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 185.83 | 186.35 | 185.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 185.83 | 186.35 | 185.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 185.90 | 186.26 | 185.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:15:00 | 185.62 | 186.26 | 185.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 185.35 | 186.08 | 185.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 185.35 | 186.08 | 185.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 185.60 | 185.98 | 185.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:45:00 | 185.77 | 185.98 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 185.61 | 185.91 | 185.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:15:00 | 185.30 | 185.91 | 185.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 185.30 | 185.79 | 185.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 185.55 | 185.79 | 185.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 185.20 | 185.67 | 185.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 185.20 | 185.67 | 185.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 185.62 | 185.66 | 185.59 | EMA400 retest candle locked (from upside) |

### Cycle 196 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 184.84 | 185.48 | 185.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 184.46 | 185.28 | 185.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 185.47 | 185.32 | 185.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 185.47 | 185.32 | 185.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 185.47 | 185.32 | 185.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 185.47 | 185.32 | 185.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 185.29 | 185.31 | 185.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 186.51 | 185.31 | 185.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 187.55 | 185.76 | 185.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 188.77 | 186.36 | 185.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 187.66 | 188.11 | 187.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:00:00 | 187.66 | 188.11 | 187.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 187.67 | 188.02 | 187.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 186.78 | 188.02 | 187.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 186.66 | 187.75 | 187.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 186.58 | 187.75 | 187.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 186.55 | 187.51 | 187.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 186.55 | 187.51 | 187.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 185.91 | 187.19 | 187.24 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 187.67 | 186.48 | 186.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 187.94 | 187.38 | 186.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 188.21 | 188.43 | 187.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 188.21 | 188.43 | 187.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 188.03 | 188.35 | 187.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 188.04 | 188.35 | 187.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 187.71 | 188.20 | 187.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 187.71 | 188.20 | 187.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 187.60 | 188.08 | 187.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 186.42 | 188.08 | 187.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 186.70 | 187.80 | 187.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 187.07 | 187.80 | 187.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 187.49 | 187.74 | 187.75 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 188.25 | 187.84 | 187.80 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 187.01 | 187.72 | 187.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 186.63 | 187.09 | 187.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 186.93 | 186.88 | 187.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 186.93 | 186.88 | 187.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 187.12 | 186.97 | 187.18 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 187.29 | 187.21 | 187.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 13:15:00 | 187.40 | 187.25 | 187.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 187.25 | 187.37 | 187.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 187.25 | 187.37 | 187.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 187.25 | 187.37 | 187.29 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 186.90 | 187.26 | 187.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 185.65 | 186.94 | 187.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 186.64 | 185.68 | 186.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 186.64 | 185.68 | 186.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 186.64 | 185.68 | 186.00 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 186.74 | 186.22 | 186.20 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 186.01 | 186.17 | 186.18 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 186.57 | 186.25 | 186.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 186.83 | 186.41 | 186.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 186.69 | 186.79 | 186.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 186.69 | 186.79 | 186.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 186.69 | 186.79 | 186.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 186.69 | 186.79 | 186.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 187.13 | 186.86 | 186.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 185.22 | 186.86 | 186.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 185.49 | 186.59 | 186.50 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 185.45 | 186.36 | 186.41 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 186.30 | 185.96 | 185.93 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 185.25 | 185.86 | 185.91 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 186.27 | 185.98 | 185.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 186.60 | 186.27 | 186.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 186.95 | 186.97 | 186.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 186.95 | 186.97 | 186.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 186.50 | 187.03 | 186.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 187.70 | 187.03 | 186.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 186.40 | 187.00 | 186.91 | SL hit (close<static) qty=1.00 sl=186.41 alert=retest2 |

### Cycle 212 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 180.20 | 185.73 | 186.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 179.80 | 184.55 | 185.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 177.45 | 175.48 | 178.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 10:00:00 | 177.45 | 175.48 | 178.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 178.25 | 176.39 | 178.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:45:00 | 178.03 | 176.39 | 178.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 178.25 | 176.76 | 178.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:00:00 | 178.25 | 176.76 | 178.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 178.19 | 177.05 | 178.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:00:00 | 178.19 | 177.05 | 178.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 178.15 | 177.27 | 178.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:15:00 | 178.20 | 177.27 | 178.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 178.20 | 177.45 | 178.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 176.75 | 177.45 | 178.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 09:30:00 | 176.60 | 174.76 | 175.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 177.60 | 176.04 | 175.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 177.60 | 176.04 | 175.91 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 174.24 | 175.68 | 175.76 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 176.60 | 175.78 | 175.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 176.91 | 176.01 | 175.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 176.33 | 176.63 | 176.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 176.33 | 176.63 | 176.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 176.33 | 176.63 | 176.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 176.10 | 176.63 | 176.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 176.44 | 176.59 | 176.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 176.74 | 176.59 | 176.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 176.78 | 176.76 | 176.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 12:15:00 | 178.77 | 179.23 | 179.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 178.77 | 179.23 | 179.28 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 179.79 | 179.35 | 179.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 180.35 | 179.79 | 179.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 179.39 | 179.87 | 179.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 12:15:00 | 179.39 | 179.87 | 179.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 179.39 | 179.87 | 179.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 180.34 | 180.04 | 179.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 182.65 | 183.08 | 183.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 182.65 | 183.08 | 183.09 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 184.49 | 183.25 | 183.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 184.74 | 183.55 | 183.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 183.57 | 183.90 | 183.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 12:15:00 | 183.57 | 183.90 | 183.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 183.57 | 183.90 | 183.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 183.57 | 183.90 | 183.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 184.06 | 183.93 | 183.65 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 183.16 | 183.89 | 183.94 | EMA200 below EMA400 |

### Cycle 221 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 184.64 | 184.06 | 184.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 185.17 | 184.28 | 184.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 12:15:00 | 184.87 | 185.12 | 184.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 12:15:00 | 184.87 | 185.12 | 184.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 184.87 | 185.12 | 184.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:00:00 | 184.87 | 185.12 | 184.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 185.10 | 185.12 | 184.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 185.40 | 185.12 | 184.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 185.40 | 185.47 | 184.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 185.18 | 184.83 | 184.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 184.54 | 184.78 | 184.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 184.54 | 184.78 | 184.78 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 185.00 | 184.82 | 184.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 185.50 | 184.96 | 184.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 185.43 | 185.62 | 185.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 185.43 | 185.62 | 185.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 185.18 | 185.53 | 185.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:45:00 | 186.15 | 185.68 | 185.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 186.90 | 185.64 | 185.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:45:00 | 186.27 | 185.60 | 185.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 184.80 | 185.28 | 185.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 184.80 | 185.28 | 185.34 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 09:15:00 | 114.35 | 2023-05-24 09:15:00 | 113.65 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2023-05-16 10:00:00 | 114.80 | 2023-05-24 09:15:00 | 113.65 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest1 | 2023-06-16 09:15:00 | 120.70 | 2023-06-21 09:15:00 | 126.42 | PARTIAL | 0.50 | 4.74% |
| BUY | retest1 | 2023-06-16 09:15:00 | 120.70 | 2023-06-22 09:15:00 | 124.55 | STOP_HIT | 0.50 | 3.19% |
| BUY | retest1 | 2023-06-16 10:30:00 | 120.40 | 2023-06-22 11:15:00 | 122.40 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2023-06-22 13:30:00 | 122.75 | 2023-06-22 14:15:00 | 121.95 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-07-11 09:30:00 | 127.40 | 2023-07-14 14:15:00 | 131.65 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest1 | 2023-07-25 09:15:00 | 137.55 | 2023-07-26 12:15:00 | 144.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-07-25 09:15:00 | 137.55 | 2023-07-28 09:15:00 | 143.75 | STOP_HIT | 0.50 | 4.51% |
| BUY | retest2 | 2023-07-26 11:30:00 | 137.40 | 2023-07-28 12:15:00 | 151.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2023-08-03 12:45:00 | 140.80 | 2023-08-04 09:15:00 | 144.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest1 | 2023-08-03 15:15:00 | 141.30 | 2023-08-04 09:15:00 | 144.50 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2023-08-21 15:15:00 | 146.50 | 2023-08-25 09:15:00 | 147.85 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-08-22 09:45:00 | 146.65 | 2023-08-25 09:15:00 | 147.85 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-08-23 10:45:00 | 146.65 | 2023-08-25 09:15:00 | 147.85 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-08-25 10:30:00 | 146.60 | 2023-08-25 11:15:00 | 147.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-08-31 12:45:00 | 142.90 | 2023-08-31 15:15:00 | 145.05 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-08-31 14:00:00 | 143.15 | 2023-08-31 15:15:00 | 145.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-08-31 14:45:00 | 142.95 | 2023-08-31 15:15:00 | 145.05 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2023-09-14 11:45:00 | 147.55 | 2023-09-21 14:15:00 | 140.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 13:15:00 | 147.40 | 2023-09-21 14:15:00 | 140.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 15:00:00 | 145.20 | 2023-09-21 14:15:00 | 139.94 | PARTIAL | 0.50 | 3.63% |
| SELL | retest2 | 2023-09-14 11:45:00 | 147.55 | 2023-09-22 11:15:00 | 142.20 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2023-09-14 13:15:00 | 147.40 | 2023-09-22 11:15:00 | 142.20 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2023-09-15 15:00:00 | 145.20 | 2023-09-22 11:15:00 | 142.20 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2023-09-20 09:30:00 | 147.30 | 2023-09-25 14:15:00 | 137.94 | PARTIAL | 0.50 | 6.35% |
| SELL | retest2 | 2023-09-20 09:30:00 | 147.30 | 2023-09-27 12:15:00 | 138.20 | STOP_HIT | 0.50 | 6.18% |
| SELL | retest2 | 2023-09-28 15:00:00 | 136.35 | 2023-09-29 15:15:00 | 138.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-10-04 09:30:00 | 138.35 | 2023-10-04 11:15:00 | 137.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-10-04 10:15:00 | 138.80 | 2023-10-04 11:15:00 | 137.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-10-12 09:15:00 | 139.80 | 2023-10-18 10:15:00 | 143.40 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2023-10-19 09:15:00 | 144.00 | 2023-10-25 09:15:00 | 136.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 10:30:00 | 144.20 | 2023-10-25 09:15:00 | 136.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 11:45:00 | 144.25 | 2023-10-25 09:15:00 | 137.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 12:15:00 | 143.75 | 2023-10-25 09:15:00 | 136.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:45:00 | 143.50 | 2023-10-25 09:15:00 | 136.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:30:00 | 143.10 | 2023-10-25 09:15:00 | 135.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 09:15:00 | 144.00 | 2023-10-26 12:15:00 | 135.90 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2023-10-19 10:30:00 | 144.20 | 2023-10-26 12:15:00 | 135.90 | STOP_HIT | 0.50 | 5.76% |
| SELL | retest2 | 2023-10-19 11:45:00 | 144.25 | 2023-10-26 12:15:00 | 135.90 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2023-10-19 12:15:00 | 143.75 | 2023-10-26 12:15:00 | 135.90 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2023-10-20 11:45:00 | 143.50 | 2023-10-26 12:15:00 | 135.90 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2023-10-20 12:30:00 | 143.10 | 2023-10-26 12:15:00 | 135.90 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2023-11-06 13:45:00 | 133.90 | 2023-11-08 11:15:00 | 136.05 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-11-06 14:45:00 | 133.90 | 2023-11-08 11:15:00 | 136.05 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-11-07 09:30:00 | 133.85 | 2023-11-08 11:15:00 | 136.05 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-11-23 09:15:00 | 135.40 | 2023-11-30 12:15:00 | 138.00 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2023-12-06 09:15:00 | 141.35 | 2023-12-06 14:15:00 | 140.55 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2023-12-08 12:45:00 | 139.90 | 2023-12-11 09:15:00 | 141.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-12-29 14:45:00 | 180.95 | 2024-01-05 15:15:00 | 187.05 | STOP_HIT | 1.00 | 3.37% |
| SELL | retest2 | 2024-01-11 10:30:00 | 180.05 | 2024-01-16 13:15:00 | 171.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 15:00:00 | 179.75 | 2024-01-17 09:15:00 | 170.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-12 09:30:00 | 179.50 | 2024-01-17 09:15:00 | 170.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 10:30:00 | 180.05 | 2024-01-18 10:15:00 | 175.95 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2024-01-11 15:00:00 | 179.75 | 2024-01-18 10:15:00 | 175.95 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2024-01-12 09:30:00 | 179.50 | 2024-01-18 10:15:00 | 175.95 | STOP_HIT | 0.50 | 1.98% |
| BUY | retest2 | 2024-02-06 10:30:00 | 204.10 | 2024-02-07 14:15:00 | 198.55 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-02-07 09:15:00 | 202.60 | 2024-02-07 14:15:00 | 198.55 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-02-26 09:15:00 | 201.55 | 2024-02-27 09:15:00 | 206.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-03-12 11:15:00 | 206.90 | 2024-03-13 11:15:00 | 196.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 11:15:00 | 206.90 | 2024-03-14 09:15:00 | 186.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-21 09:15:00 | 194.90 | 2024-04-01 09:15:00 | 196.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-04-04 13:30:00 | 209.40 | 2024-04-10 09:15:00 | 230.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-05 09:45:00 | 210.50 | 2024-04-15 11:15:00 | 214.30 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2024-04-09 09:15:00 | 209.80 | 2024-04-15 11:15:00 | 214.30 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2024-04-18 13:15:00 | 209.95 | 2024-04-24 09:15:00 | 215.40 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-04-19 10:15:00 | 210.70 | 2024-04-24 09:15:00 | 215.40 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-04-23 10:15:00 | 209.80 | 2024-04-24 09:15:00 | 215.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-05-07 10:15:00 | 195.80 | 2024-05-09 15:15:00 | 186.25 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2024-05-07 10:15:00 | 195.80 | 2024-05-10 09:15:00 | 193.75 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2024-05-07 11:00:00 | 196.05 | 2024-05-13 10:15:00 | 186.01 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2024-05-07 11:00:00 | 196.05 | 2024-05-13 13:15:00 | 189.90 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2024-05-10 09:30:00 | 195.60 | 2024-05-15 12:15:00 | 190.90 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2024-05-22 09:15:00 | 194.05 | 2024-05-22 10:15:00 | 192.35 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-05-30 10:00:00 | 186.05 | 2024-05-31 10:15:00 | 190.85 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-05-30 11:15:00 | 186.00 | 2024-05-31 10:15:00 | 190.85 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-06-05 14:30:00 | 185.50 | 2024-06-06 09:15:00 | 189.35 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-06-28 12:15:00 | 201.91 | 2024-07-01 09:15:00 | 209.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-07-29 09:15:00 | 267.70 | 2024-07-31 09:15:00 | 264.22 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-07-31 10:45:00 | 266.77 | 2024-07-31 12:15:00 | 265.53 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-31 11:45:00 | 266.42 | 2024-07-31 12:15:00 | 265.53 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-08-06 11:00:00 | 248.85 | 2024-08-07 14:15:00 | 255.90 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-08-16 14:15:00 | 248.85 | 2024-08-19 09:15:00 | 253.70 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-09-10 12:15:00 | 258.40 | 2024-09-12 15:15:00 | 261.20 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-09-11 12:15:00 | 258.60 | 2024-09-12 15:15:00 | 261.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-09-11 13:00:00 | 258.60 | 2024-09-12 15:15:00 | 261.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-09-12 09:30:00 | 258.20 | 2024-09-12 15:15:00 | 261.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-09-12 15:00:00 | 260.20 | 2024-09-12 15:15:00 | 261.20 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-09-17 12:00:00 | 253.60 | 2024-09-23 13:15:00 | 251.35 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2024-09-17 14:45:00 | 253.50 | 2024-09-23 13:15:00 | 251.35 | STOP_HIT | 1.00 | 0.85% |
| SELL | retest2 | 2024-09-18 12:00:00 | 254.00 | 2024-09-23 13:15:00 | 251.35 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2024-09-19 09:30:00 | 253.75 | 2024-09-23 13:15:00 | 251.35 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-09-26 13:15:00 | 246.50 | 2024-10-03 12:15:00 | 234.65 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-09-26 14:00:00 | 247.00 | 2024-10-03 12:15:00 | 234.41 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-09-27 09:45:00 | 246.75 | 2024-10-03 12:15:00 | 234.89 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-09-27 10:45:00 | 247.25 | 2024-10-03 12:15:00 | 234.56 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2024-09-27 12:15:00 | 246.90 | 2024-10-03 13:15:00 | 234.17 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2024-09-26 13:15:00 | 246.50 | 2024-10-07 10:15:00 | 221.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-26 14:00:00 | 247.00 | 2024-10-07 10:15:00 | 222.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 09:45:00 | 246.75 | 2024-10-07 10:15:00 | 222.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 10:45:00 | 247.25 | 2024-10-07 10:15:00 | 222.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 12:15:00 | 246.90 | 2024-10-07 10:15:00 | 222.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 219.79 | 2024-10-22 12:15:00 | 208.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 219.79 | 2024-10-23 10:15:00 | 210.78 | STOP_HIT | 0.50 | 4.10% |
| BUY | retest2 | 2024-11-01 18:00:00 | 214.69 | 2024-11-04 09:15:00 | 210.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-11-19 14:45:00 | 190.75 | 2024-11-22 10:15:00 | 196.20 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-11-26 09:15:00 | 195.70 | 2024-12-02 14:15:00 | 215.27 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 14:45:00 | 208.73 | 2024-12-19 09:15:00 | 198.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 208.80 | 2024-12-19 09:15:00 | 198.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:00:00 | 209.07 | 2024-12-19 09:15:00 | 198.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:45:00 | 208.73 | 2024-12-19 10:15:00 | 203.90 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2024-12-17 10:15:00 | 208.80 | 2024-12-19 10:15:00 | 203.90 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2024-12-17 11:00:00 | 209.07 | 2024-12-19 10:15:00 | 203.90 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-01-08 12:00:00 | 197.58 | 2025-01-08 14:15:00 | 200.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-08 12:45:00 | 197.40 | 2025-01-08 14:15:00 | 200.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-01-16 15:00:00 | 183.88 | 2025-01-17 09:15:00 | 186.72 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-01-23 12:45:00 | 179.69 | 2025-01-27 09:15:00 | 170.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 179.35 | 2025-01-27 09:15:00 | 170.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 179.50 | 2025-01-27 09:15:00 | 170.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:45:00 | 179.69 | 2025-01-28 11:15:00 | 170.51 | STOP_HIT | 0.50 | 5.11% |
| SELL | retest2 | 2025-01-23 14:15:00 | 179.35 | 2025-01-28 11:15:00 | 170.51 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2025-01-24 09:30:00 | 179.50 | 2025-01-28 11:15:00 | 170.51 | STOP_HIT | 0.50 | 5.01% |
| BUY | retest2 | 2025-02-01 09:15:00 | 179.11 | 2025-02-03 12:15:00 | 174.01 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-02-01 12:45:00 | 178.91 | 2025-02-03 12:15:00 | 174.01 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-02-12 11:15:00 | 208.82 | 2025-02-13 14:15:00 | 206.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-02-12 13:00:00 | 208.96 | 2025-02-13 14:15:00 | 206.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-02-13 09:15:00 | 210.17 | 2025-02-13 14:15:00 | 206.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-02-20 11:00:00 | 212.85 | 2025-02-28 12:15:00 | 210.82 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-02-24 09:45:00 | 213.09 | 2025-02-28 12:15:00 | 210.82 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-03-18 09:15:00 | 222.24 | 2025-03-25 11:15:00 | 211.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-18 09:15:00 | 222.24 | 2025-03-27 09:15:00 | 209.07 | STOP_HIT | 0.50 | 5.93% |
| BUY | retest2 | 2025-04-17 11:15:00 | 203.66 | 2025-04-17 15:15:00 | 201.99 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-04-21 09:30:00 | 203.98 | 2025-04-25 10:15:00 | 202.63 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-04-25 10:00:00 | 203.68 | 2025-04-25 10:15:00 | 202.63 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-05-21 12:00:00 | 208.00 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2025-05-22 09:15:00 | 208.27 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-05-22 13:30:00 | 208.46 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2025-05-23 09:45:00 | 208.31 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-05-27 13:30:00 | 206.87 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2025-05-28 09:45:00 | 207.04 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-05-28 12:15:00 | 206.90 | 2025-05-29 09:15:00 | 217.40 | STOP_HIT | 1.00 | -5.07% |
| BUY | retest2 | 2025-06-04 10:30:00 | 218.94 | 2025-06-06 11:15:00 | 217.35 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-04 12:00:00 | 218.78 | 2025-06-06 11:15:00 | 217.35 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-04 12:30:00 | 218.69 | 2025-06-06 11:15:00 | 217.35 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-06-04 13:45:00 | 218.54 | 2025-06-06 11:15:00 | 217.35 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-06-17 11:45:00 | 211.89 | 2025-06-24 09:15:00 | 211.99 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-07-15 09:15:00 | 226.18 | 2025-07-16 12:15:00 | 222.79 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-16 10:15:00 | 223.38 | 2025-07-16 12:15:00 | 222.79 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-16 10:45:00 | 223.28 | 2025-07-16 12:15:00 | 222.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-21 09:15:00 | 219.81 | 2025-07-22 09:15:00 | 221.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-22 14:15:00 | 219.67 | 2025-07-23 09:15:00 | 225.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-07-22 15:15:00 | 219.70 | 2025-07-23 09:15:00 | 225.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-08-01 11:30:00 | 220.46 | 2025-08-05 09:15:00 | 221.78 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-08-08 11:45:00 | 212.85 | 2025-08-20 12:15:00 | 207.89 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-08-08 13:00:00 | 212.85 | 2025-08-20 12:15:00 | 207.89 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-08-08 15:15:00 | 212.80 | 2025-08-20 12:15:00 | 207.89 | STOP_HIT | 1.00 | 2.31% |
| SELL | retest2 | 2025-09-03 09:15:00 | 198.10 | 2025-09-10 14:15:00 | 199.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-09-25 09:15:00 | 200.85 | 2025-09-30 14:15:00 | 199.76 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2025-10-08 09:15:00 | 203.39 | 2025-10-08 10:15:00 | 201.14 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-10 12:15:00 | 200.25 | 2025-10-15 13:15:00 | 200.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-10-10 15:00:00 | 200.40 | 2025-10-15 13:15:00 | 200.95 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-10-13 09:15:00 | 199.95 | 2025-10-15 14:15:00 | 203.88 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-10-13 09:45:00 | 199.90 | 2025-10-15 14:15:00 | 203.88 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-10-14 10:15:00 | 199.31 | 2025-10-15 14:15:00 | 203.88 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-10-15 10:45:00 | 199.35 | 2025-10-15 14:15:00 | 203.88 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-10-29 10:45:00 | 196.96 | 2025-11-07 09:15:00 | 188.02 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2025-10-29 10:45:00 | 196.96 | 2025-11-07 12:15:00 | 191.00 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-10-29 14:30:00 | 197.92 | 2025-11-13 09:15:00 | 194.39 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2025-10-30 09:30:00 | 197.71 | 2025-11-13 09:15:00 | 194.39 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2025-11-26 10:30:00 | 192.46 | 2025-11-27 13:15:00 | 190.87 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-11-26 11:15:00 | 192.50 | 2025-11-27 13:15:00 | 190.87 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-11-26 11:45:00 | 192.49 | 2025-11-27 13:15:00 | 190.87 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-10 09:15:00 | 187.84 | 2025-12-18 15:15:00 | 183.52 | STOP_HIT | 1.00 | 2.30% |
| SELL | retest2 | 2025-12-10 10:45:00 | 187.89 | 2025-12-18 15:15:00 | 183.52 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-12-10 11:15:00 | 187.90 | 2025-12-18 15:15:00 | 183.52 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-12-24 12:15:00 | 186.45 | 2025-12-30 12:15:00 | 189.05 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2026-01-08 10:45:00 | 188.70 | 2026-01-21 15:15:00 | 185.40 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2026-01-08 12:30:00 | 188.79 | 2026-01-21 15:15:00 | 185.40 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2026-01-08 15:00:00 | 188.82 | 2026-01-21 15:15:00 | 185.40 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2026-01-09 11:15:00 | 188.88 | 2026-01-21 15:15:00 | 185.40 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2026-01-12 11:15:00 | 185.91 | 2026-01-21 15:15:00 | 185.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2026-01-16 09:15:00 | 186.59 | 2026-01-21 15:15:00 | 185.40 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2026-03-20 09:15:00 | 187.70 | 2026-03-20 12:15:00 | 186.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-03-20 14:45:00 | 187.19 | 2026-03-23 09:15:00 | 180.20 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2026-03-27 09:15:00 | 176.75 | 2026-04-01 14:15:00 | 177.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-04-01 09:30:00 | 176.60 | 2026-04-01 14:15:00 | 177.60 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-04-06 11:15:00 | 176.74 | 2026-04-13 12:15:00 | 178.77 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-04-06 11:45:00 | 176.78 | 2026-04-13 12:15:00 | 178.77 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-04-16 14:30:00 | 180.34 | 2026-04-24 15:15:00 | 182.65 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2026-05-04 14:15:00 | 185.40 | 2026-05-05 14:15:00 | 184.54 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-05-05 09:30:00 | 185.40 | 2026-05-05 14:15:00 | 184.54 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-05-05 14:15:00 | 185.18 | 2026-05-05 14:15:00 | 184.54 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-05-07 13:45:00 | 186.15 | 2026-05-08 13:15:00 | 184.80 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-05-08 09:15:00 | 186.90 | 2026-05-08 13:15:00 | 184.80 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-05-08 10:45:00 | 186.27 | 2026-05-08 13:15:00 | 184.80 | STOP_HIT | 1.00 | -0.79% |
