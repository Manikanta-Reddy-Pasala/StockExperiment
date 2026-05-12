# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 107.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 146 |
| ALERT1 | 100 |
| ALERT2 | 99 |
| ALERT2_SKIP | 48 |
| ALERT3 | 270 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 121 |
| PARTIAL | 14 |
| TARGET_HIT | 5 |
| STOP_HIT | 119 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 88
- **Target hits / Stop hits / Partials:** 5 / 116 / 14
- **Avg / median % per leg:** 0.53% / -0.78%
- **Sum % (uncompounded):** 71.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 16 | 26.7% | 2 | 58 | 0 | -0.43% | -26.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 60 | 16 | 26.7% | 2 | 58 | 0 | -0.43% | -26.0% |
| SELL (all) | 75 | 31 | 41.3% | 3 | 58 | 14 | 1.30% | 97.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 75 | 31 | 41.3% | 3 | 58 | 14 | 1.30% | 97.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 135 | 47 | 34.8% | 5 | 116 | 14 | 0.53% | 71.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 124.65 | 123.84 | 123.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 125.70 | 124.21 | 123.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 125.30 | 125.42 | 124.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 125.30 | 125.42 | 124.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 124.50 | 125.19 | 124.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 124.50 | 125.19 | 124.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 124.15 | 124.98 | 124.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 126.10 | 124.98 | 124.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 123.50 | 124.54 | 124.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 123.50 | 124.54 | 124.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 122.75 | 124.18 | 124.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 125.05 | 124.36 | 124.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 125.05 | 124.36 | 124.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 125.05 | 124.36 | 124.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 125.05 | 124.36 | 124.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 125.35 | 124.56 | 124.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 126.20 | 125.29 | 125.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 126.30 | 126.58 | 125.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 126.30 | 126.58 | 125.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 125.30 | 126.27 | 125.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 125.30 | 126.27 | 125.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 125.60 | 126.14 | 125.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:15:00 | 125.85 | 125.94 | 125.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 125.90 | 125.97 | 125.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 127.25 | 128.40 | 128.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 127.25 | 128.40 | 128.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 126.30 | 127.88 | 128.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 127.75 | 127.52 | 127.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 12:15:00 | 127.75 | 127.52 | 127.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 127.75 | 127.52 | 127.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 127.75 | 127.52 | 127.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 129.60 | 127.94 | 128.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 129.60 | 127.94 | 128.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 129.35 | 128.22 | 128.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 135.65 | 129.88 | 128.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 129.40 | 133.80 | 132.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 129.40 | 133.80 | 132.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 129.40 | 133.80 | 132.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 127.40 | 133.80 | 132.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 123.30 | 131.70 | 131.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 123.30 | 131.70 | 131.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 112.15 | 127.79 | 129.48 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 124.55 | 123.37 | 123.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 125.10 | 124.27 | 123.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 125.50 | 125.80 | 125.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 125.76 | 125.80 | 125.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 125.33 | 125.71 | 125.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 126.26 | 125.71 | 125.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 127.03 | 128.24 | 128.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 127.03 | 128.24 | 128.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 11:15:00 | 126.69 | 127.71 | 128.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 125.24 | 124.43 | 125.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 125.24 | 124.43 | 125.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 125.24 | 124.43 | 125.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 125.24 | 124.43 | 125.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 125.70 | 124.68 | 125.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 125.87 | 124.68 | 125.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 124.90 | 124.73 | 125.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:30:00 | 124.25 | 124.74 | 125.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 14:15:00 | 118.04 | 121.40 | 123.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 124.76 | 121.72 | 122.90 | SL hit (close>ema200) qty=0.50 sl=121.72 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 122.76 | 121.80 | 121.74 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 13:15:00 | 121.38 | 122.08 | 122.10 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 123.26 | 122.12 | 122.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 123.73 | 122.44 | 122.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 14:15:00 | 122.46 | 122.78 | 122.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 14:15:00 | 122.46 | 122.78 | 122.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 122.46 | 122.78 | 122.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 122.46 | 122.78 | 122.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 122.30 | 122.68 | 122.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 121.51 | 122.68 | 122.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 119.97 | 122.14 | 122.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 12:15:00 | 119.35 | 120.90 | 121.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 118.83 | 118.55 | 119.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 10:00:00 | 118.83 | 118.55 | 119.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 119.32 | 118.70 | 119.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 119.32 | 118.70 | 119.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 119.39 | 118.84 | 119.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 119.39 | 118.84 | 119.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 119.78 | 119.03 | 119.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:45:00 | 119.67 | 119.03 | 119.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 120.86 | 119.39 | 119.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 120.86 | 119.39 | 119.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 120.91 | 119.70 | 119.60 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 119.39 | 119.91 | 119.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 118.85 | 119.58 | 119.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 117.68 | 117.49 | 118.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 117.68 | 117.49 | 118.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 117.68 | 117.49 | 118.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 117.99 | 117.49 | 118.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 118.04 | 117.60 | 118.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 118.27 | 117.60 | 118.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 118.08 | 117.69 | 118.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 118.18 | 117.69 | 118.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 118.35 | 117.83 | 118.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 118.35 | 117.83 | 118.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 118.18 | 117.90 | 118.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 118.18 | 117.90 | 118.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 118.25 | 117.97 | 118.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 118.29 | 117.97 | 118.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 118.08 | 117.99 | 118.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 118.56 | 117.99 | 118.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 117.98 | 117.99 | 118.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 115.51 | 118.28 | 118.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 116.60 | 117.94 | 118.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 117.13 | 117.88 | 118.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:00:00 | 116.91 | 117.71 | 117.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 116.05 | 117.03 | 117.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 13:30:00 | 117.72 | 117.03 | 117.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 117.64 | 116.96 | 117.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 117.64 | 116.96 | 117.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 117.25 | 117.02 | 117.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 117.79 | 117.02 | 117.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 117.31 | 117.08 | 117.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:15:00 | 117.58 | 117.08 | 117.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 117.54 | 117.17 | 117.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 117.75 | 117.46 | 117.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 117.75 | 117.46 | 117.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 118.85 | 118.10 | 117.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 125.28 | 125.63 | 123.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:00:00 | 125.28 | 125.63 | 123.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 123.81 | 125.17 | 124.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 123.81 | 125.17 | 124.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 124.73 | 125.08 | 124.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:15:00 | 124.80 | 124.98 | 124.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 13:15:00 | 123.44 | 124.57 | 124.10 | SL hit (close<static) qty=1.00 sl=123.63 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 122.81 | 123.88 | 123.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 122.54 | 123.61 | 123.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 116.49 | 115.88 | 118.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:30:00 | 116.60 | 115.88 | 118.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 115.98 | 114.98 | 115.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 115.98 | 114.98 | 115.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 116.10 | 115.20 | 115.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 115.09 | 115.20 | 115.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 115.17 | 115.20 | 115.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:30:00 | 114.43 | 114.86 | 115.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:00:00 | 114.36 | 114.63 | 115.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 114.46 | 114.92 | 115.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:45:00 | 114.09 | 114.81 | 115.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 115.40 | 114.93 | 115.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 115.68 | 114.93 | 115.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 115.12 | 114.97 | 115.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 12:15:00 | 115.03 | 114.97 | 115.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:45:00 | 114.69 | 115.01 | 115.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:45:00 | 114.88 | 115.00 | 115.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:15:00 | 115.00 | 115.00 | 115.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 115.56 | 115.11 | 115.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 115.56 | 115.11 | 115.11 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 114.57 | 115.01 | 115.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 113.99 | 114.67 | 114.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 113.23 | 113.07 | 113.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 15:00:00 | 113.23 | 113.07 | 113.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 115.04 | 113.46 | 113.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 115.26 | 113.46 | 113.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 115.25 | 114.05 | 113.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 115.40 | 114.73 | 114.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 116.02 | 116.35 | 115.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:00:00 | 116.02 | 116.35 | 115.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 115.82 | 116.22 | 115.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 115.82 | 116.22 | 115.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 115.75 | 116.12 | 115.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:45:00 | 115.95 | 116.17 | 115.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 115.88 | 116.54 | 116.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 115.88 | 116.54 | 116.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 11:15:00 | 115.76 | 115.94 | 116.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 115.22 | 115.20 | 115.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:30:00 | 115.40 | 115.20 | 115.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 115.85 | 115.34 | 115.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 116.38 | 115.34 | 115.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 117.37 | 115.75 | 115.69 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 115.67 | 116.13 | 116.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 15:15:00 | 115.34 | 115.97 | 116.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 13:15:00 | 113.65 | 113.43 | 114.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 13:30:00 | 113.56 | 113.43 | 114.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 109.78 | 109.21 | 110.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:45:00 | 109.18 | 109.57 | 109.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 109.07 | 108.56 | 108.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 10:15:00 | 109.30 | 108.56 | 108.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 110.21 | 109.13 | 109.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 110.21 | 109.13 | 109.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 112.36 | 109.78 | 109.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 110.32 | 110.71 | 110.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 12:00:00 | 110.32 | 110.71 | 110.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 109.37 | 110.50 | 110.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 109.37 | 110.50 | 110.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 108.93 | 110.18 | 110.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 108.93 | 110.18 | 110.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 108.96 | 109.94 | 110.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 13:15:00 | 108.75 | 109.58 | 109.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 11:15:00 | 109.10 | 108.83 | 109.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 11:30:00 | 108.90 | 108.83 | 109.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 109.18 | 108.90 | 109.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:45:00 | 109.40 | 108.90 | 109.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 108.39 | 108.80 | 109.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 108.60 | 108.80 | 109.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 107.96 | 108.56 | 108.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:00:00 | 107.90 | 108.43 | 108.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 107.61 | 108.01 | 108.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 107.84 | 108.11 | 108.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 109.19 | 108.41 | 108.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 109.19 | 108.41 | 108.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 111.32 | 108.99 | 108.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 108.65 | 110.19 | 109.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 108.65 | 110.19 | 109.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 108.65 | 110.19 | 109.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 108.65 | 110.19 | 109.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 108.80 | 109.92 | 109.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 11:30:00 | 109.36 | 109.77 | 109.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 108.06 | 109.16 | 109.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 108.06 | 109.16 | 109.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 105.00 | 108.13 | 108.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 107.05 | 106.01 | 106.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 10:15:00 | 107.05 | 106.01 | 106.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 107.05 | 106.01 | 106.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 106.82 | 106.01 | 106.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 106.97 | 106.20 | 106.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:30:00 | 106.47 | 106.32 | 106.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 13:15:00 | 107.21 | 106.50 | 106.97 | SL hit (close>static) qty=1.00 sl=107.15 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 108.91 | 107.20 | 107.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 14:15:00 | 109.26 | 108.21 | 107.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 108.18 | 108.29 | 107.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 10:00:00 | 108.18 | 108.29 | 107.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 108.20 | 108.24 | 107.92 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 106.99 | 107.73 | 107.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 105.59 | 107.31 | 107.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 106.14 | 105.87 | 106.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 106.14 | 105.87 | 106.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 106.14 | 105.87 | 106.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 106.14 | 105.87 | 106.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 105.37 | 105.17 | 105.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 105.37 | 105.17 | 105.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 106.16 | 105.37 | 105.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 106.16 | 105.37 | 105.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 107.24 | 105.74 | 105.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 107.24 | 105.74 | 105.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 106.16 | 105.82 | 105.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 106.00 | 105.82 | 105.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 105.83 | 105.89 | 105.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:30:00 | 105.61 | 105.69 | 105.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 12:15:00 | 104.66 | 103.79 | 103.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 104.66 | 103.79 | 103.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 105.07 | 104.19 | 103.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 104.22 | 104.32 | 104.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 104.22 | 104.32 | 104.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 104.22 | 104.32 | 104.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 104.22 | 104.32 | 104.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 103.23 | 104.10 | 104.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 103.23 | 104.10 | 104.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 103.65 | 104.01 | 103.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:15:00 | 103.75 | 104.01 | 103.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 15:15:00 | 103.75 | 103.96 | 103.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 103.75 | 103.96 | 103.96 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 104.14 | 104.00 | 103.98 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 103.71 | 103.94 | 103.96 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 12:15:00 | 104.23 | 103.97 | 103.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 104.87 | 104.21 | 104.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 104.58 | 104.85 | 104.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 104.58 | 104.85 | 104.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 104.58 | 104.85 | 104.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 104.58 | 104.85 | 104.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 104.80 | 104.84 | 104.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:15:00 | 104.79 | 104.84 | 104.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 105.02 | 104.87 | 104.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 105.10 | 104.89 | 104.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 105.10 | 104.95 | 104.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 103.84 | 104.73 | 104.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 103.84 | 104.73 | 104.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 103.26 | 104.11 | 104.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 103.90 | 103.51 | 103.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:00:00 | 103.90 | 103.51 | 103.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 103.60 | 103.53 | 103.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 13:45:00 | 103.36 | 103.48 | 103.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 103.24 | 103.43 | 103.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 103.24 | 103.31 | 103.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 98.19 | 100.96 | 102.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 98.08 | 100.96 | 102.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 98.08 | 100.96 | 102.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 93.02 | 97.14 | 99.60 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 99.28 | 97.44 | 97.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 100.03 | 98.24 | 97.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 10:15:00 | 98.19 | 98.55 | 98.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 10:15:00 | 98.19 | 98.55 | 98.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 98.19 | 98.55 | 98.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:45:00 | 98.20 | 98.55 | 98.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 98.63 | 98.57 | 98.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 14:00:00 | 99.48 | 98.74 | 98.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 13:15:00 | 97.01 | 99.04 | 99.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 97.01 | 99.04 | 99.19 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 101.11 | 99.43 | 99.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 09:15:00 | 102.06 | 99.96 | 99.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 99.59 | 99.88 | 99.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 10:15:00 | 99.59 | 99.88 | 99.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 99.59 | 99.88 | 99.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 99.59 | 99.88 | 99.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 99.71 | 99.85 | 99.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 101.54 | 100.19 | 99.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 103.71 | 105.02 | 105.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 103.71 | 105.02 | 105.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 103.50 | 104.25 | 104.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 102.26 | 102.21 | 103.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 102.26 | 102.21 | 103.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 102.06 | 100.70 | 101.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:45:00 | 101.74 | 100.70 | 101.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 101.60 | 100.88 | 101.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 101.60 | 100.88 | 101.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 103.35 | 101.28 | 101.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 103.35 | 101.28 | 101.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 103.17 | 101.65 | 101.67 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 102.96 | 101.92 | 101.79 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 100.60 | 101.61 | 101.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 95.89 | 100.47 | 101.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 99.17 | 97.95 | 99.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 99.17 | 97.95 | 99.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 99.17 | 97.95 | 99.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 98.67 | 97.95 | 99.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 99.10 | 98.18 | 99.20 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 104.71 | 100.38 | 99.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 105.37 | 101.38 | 100.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 103.96 | 104.48 | 103.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 103.96 | 104.48 | 103.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 104.77 | 105.39 | 104.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 104.77 | 105.39 | 104.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 104.14 | 105.14 | 104.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 104.14 | 105.14 | 104.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 104.85 | 104.95 | 104.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 104.55 | 104.95 | 104.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 104.64 | 104.89 | 104.78 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 104.52 | 104.73 | 104.73 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 105.01 | 104.78 | 104.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 107.22 | 105.30 | 105.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 108.47 | 108.88 | 107.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 108.47 | 108.88 | 107.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 107.97 | 108.90 | 108.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 107.97 | 108.90 | 108.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 110.35 | 109.19 | 108.53 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 108.68 | 109.26 | 109.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 107.80 | 108.85 | 109.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 107.08 | 106.81 | 107.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:45:00 | 107.14 | 106.81 | 107.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 108.08 | 107.06 | 107.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 108.08 | 107.06 | 107.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 107.76 | 107.20 | 107.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 108.04 | 107.20 | 107.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 108.00 | 107.58 | 107.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 107.90 | 107.65 | 107.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 108.14 | 107.89 | 107.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 108.14 | 107.89 | 107.86 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 106.90 | 107.69 | 107.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 106.29 | 107.29 | 107.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 104.15 | 103.78 | 104.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 11:30:00 | 103.93 | 103.78 | 104.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 102.36 | 101.68 | 101.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 102.55 | 101.68 | 101.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 102.24 | 101.79 | 102.01 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 102.93 | 102.21 | 102.14 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 101.60 | 102.09 | 102.12 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 102.85 | 102.19 | 102.15 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 101.03 | 101.94 | 102.04 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 102.81 | 102.13 | 102.04 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 101.64 | 102.26 | 102.27 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 105.02 | 102.81 | 102.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 15:15:00 | 105.65 | 103.79 | 103.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 103.04 | 105.25 | 104.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 103.04 | 105.25 | 104.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 103.04 | 105.25 | 104.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 102.05 | 105.25 | 104.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 102.02 | 104.61 | 104.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 102.02 | 104.61 | 104.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 102.52 | 103.86 | 103.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 101.63 | 103.41 | 103.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 103.75 | 103.00 | 103.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 103.75 | 103.00 | 103.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 103.75 | 103.00 | 103.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 104.31 | 103.00 | 103.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 104.03 | 103.21 | 103.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 104.03 | 103.21 | 103.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 103.86 | 103.34 | 103.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 104.29 | 103.34 | 103.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 103.52 | 103.38 | 103.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 103.70 | 103.38 | 103.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 103.23 | 103.35 | 103.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 102.96 | 103.35 | 103.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 102.76 | 103.28 | 103.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 97.81 | 99.35 | 100.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 97.62 | 99.35 | 100.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 98.11 | 97.18 | 98.51 | SL hit (close>ema200) qty=0.50 sl=97.18 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 99.52 | 98.61 | 98.59 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 97.75 | 98.45 | 98.52 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 100.60 | 98.82 | 98.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 101.36 | 99.33 | 98.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 99.63 | 100.13 | 99.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 99.63 | 100.13 | 99.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 99.63 | 100.13 | 99.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 99.63 | 100.13 | 99.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 99.73 | 100.05 | 99.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 99.71 | 100.05 | 99.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 99.94 | 100.03 | 99.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:30:00 | 100.36 | 100.02 | 99.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 15:15:00 | 100.30 | 100.02 | 99.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:30:00 | 100.40 | 100.84 | 100.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:45:00 | 100.21 | 100.75 | 100.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 100.16 | 100.63 | 100.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 98.85 | 100.63 | 100.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 98.71 | 100.25 | 100.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 98.71 | 100.25 | 100.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 97.46 | 99.45 | 99.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 98.42 | 98.40 | 99.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:45:00 | 98.15 | 98.40 | 99.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 98.76 | 98.47 | 99.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 98.88 | 98.47 | 99.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 98.97 | 98.60 | 99.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 98.93 | 98.60 | 99.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 99.25 | 98.73 | 99.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 99.67 | 98.73 | 99.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 99.48 | 98.88 | 99.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 99.48 | 98.88 | 99.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 99.25 | 98.95 | 99.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 98.93 | 98.95 | 99.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 13:15:00 | 99.80 | 97.57 | 97.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 99.80 | 97.57 | 97.56 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 10:15:00 | 97.07 | 97.82 | 97.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 11:15:00 | 96.40 | 97.54 | 97.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 97.28 | 97.03 | 97.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 97.28 | 97.03 | 97.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 97.28 | 97.03 | 97.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:15:00 | 98.26 | 97.03 | 97.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 98.54 | 97.33 | 97.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 98.54 | 97.33 | 97.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 98.56 | 97.58 | 97.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 98.81 | 97.83 | 97.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 99.85 | 100.04 | 99.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 99.87 | 100.04 | 99.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 98.65 | 99.76 | 99.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 98.65 | 99.76 | 99.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 100.15 | 99.84 | 99.15 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 98.30 | 98.87 | 98.91 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 100.25 | 98.92 | 98.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 100.82 | 99.47 | 99.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 99.76 | 100.10 | 99.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 99.76 | 100.10 | 99.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 99.76 | 100.10 | 99.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 99.76 | 100.10 | 99.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 100.06 | 100.09 | 99.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 100.05 | 100.09 | 99.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 99.96 | 100.06 | 99.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 99.96 | 100.06 | 99.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 100.14 | 100.08 | 99.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 99.82 | 100.08 | 99.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 99.82 | 100.03 | 99.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 99.62 | 100.03 | 99.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 99.69 | 99.96 | 99.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 99.98 | 99.96 | 99.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 99.70 | 99.91 | 99.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 99.31 | 99.91 | 99.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 99.52 | 99.83 | 99.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 99.52 | 99.83 | 99.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 99.85 | 99.83 | 99.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 100.10 | 99.83 | 99.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 98.93 | 99.65 | 99.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 98.93 | 99.65 | 99.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 98.58 | 99.38 | 99.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 95.81 | 95.34 | 96.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 95.87 | 95.34 | 96.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 96.20 | 95.69 | 96.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 95.35 | 95.70 | 96.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 90.58 | 92.51 | 93.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 92.57 | 92.28 | 93.29 | SL hit (close>ema200) qty=0.50 sl=92.28 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 94.20 | 92.99 | 92.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 95.27 | 93.45 | 93.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 94.53 | 94.84 | 94.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 94.53 | 94.84 | 94.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 94.21 | 94.59 | 94.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 94.05 | 94.59 | 94.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 94.31 | 94.53 | 94.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 94.18 | 94.53 | 94.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 94.14 | 94.45 | 94.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 93.82 | 94.45 | 94.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 93.57 | 94.28 | 94.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 93.10 | 94.28 | 94.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 94.07 | 94.08 | 94.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:15:00 | 93.98 | 94.08 | 94.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 93.66 | 94.00 | 94.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 93.45 | 93.90 | 93.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 87.85 | 87.63 | 89.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 87.85 | 87.63 | 89.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 88.94 | 87.67 | 88.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 89.03 | 87.67 | 88.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 89.52 | 88.04 | 88.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 89.52 | 88.04 | 88.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 89.27 | 88.53 | 88.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 89.78 | 88.91 | 88.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 90.89 | 90.90 | 90.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 90.89 | 90.90 | 90.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 90.31 | 90.79 | 90.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:15:00 | 90.34 | 90.79 | 90.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 90.16 | 90.67 | 90.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 90.16 | 90.67 | 90.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 89.67 | 90.47 | 90.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 89.67 | 90.47 | 90.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 89.91 | 90.36 | 90.29 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 89.57 | 90.20 | 90.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 88.66 | 89.89 | 90.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 88.50 | 88.47 | 89.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 88.21 | 88.47 | 89.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 87.39 | 88.25 | 88.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 87.25 | 88.25 | 88.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 86.58 | 87.99 | 88.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 87.21 | 87.22 | 87.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 87.17 | 87.34 | 87.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 87.19 | 87.26 | 87.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 88.70 | 87.76 | 87.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 88.70 | 87.76 | 87.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 89.32 | 88.07 | 87.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 94.72 | 95.10 | 93.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:00:00 | 94.72 | 95.10 | 93.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 94.30 | 94.94 | 93.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 94.44 | 94.94 | 93.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 94.20 | 94.66 | 93.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 94.41 | 94.33 | 93.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 94.40 | 94.29 | 93.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 93.56 | 94.06 | 93.91 | SL hit (close<static) qty=1.00 sl=93.60 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 92.85 | 93.68 | 93.76 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 95.00 | 93.81 | 93.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 96.55 | 94.82 | 94.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 95.78 | 95.88 | 95.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 95.78 | 95.88 | 95.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 96.68 | 96.06 | 95.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:00:00 | 97.23 | 96.36 | 95.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 15:00:00 | 97.40 | 96.71 | 96.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:00:00 | 97.30 | 96.92 | 96.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 97.45 | 97.95 | 97.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 97.42 | 97.85 | 97.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 97.42 | 97.85 | 97.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 96.86 | 97.65 | 97.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 96.86 | 97.65 | 97.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 96.65 | 97.45 | 97.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 96.64 | 97.45 | 97.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 93.50 | 96.47 | 96.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 93.50 | 96.47 | 96.85 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 96.94 | 96.24 | 96.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 15:15:00 | 97.08 | 96.41 | 96.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 95.67 | 96.26 | 96.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 95.67 | 96.26 | 96.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 95.67 | 96.26 | 96.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 95.67 | 96.26 | 96.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 10:15:00 | 94.96 | 96.00 | 96.09 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 96.59 | 96.12 | 96.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 96.80 | 96.26 | 96.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 96.02 | 96.29 | 96.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 13:15:00 | 96.02 | 96.29 | 96.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 96.02 | 96.29 | 96.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:00:00 | 96.02 | 96.29 | 96.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 96.13 | 96.26 | 96.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 97.30 | 96.22 | 96.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 12:45:00 | 96.49 | 96.36 | 96.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 100.43 | 102.39 | 102.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 100.43 | 102.39 | 102.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 99.17 | 101.75 | 102.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 101.07 | 100.39 | 101.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 101.07 | 100.39 | 101.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 101.07 | 100.39 | 101.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 100.81 | 100.39 | 101.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 100.77 | 100.46 | 101.09 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 102.15 | 101.48 | 101.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 103.31 | 101.95 | 101.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 101.43 | 102.30 | 102.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 101.43 | 102.30 | 102.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 101.43 | 102.30 | 102.07 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 100.90 | 101.81 | 101.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 100.26 | 101.37 | 101.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 101.30 | 100.99 | 101.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 101.30 | 100.99 | 101.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 101.30 | 100.99 | 101.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 101.23 | 100.99 | 101.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 100.94 | 100.98 | 101.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 101.30 | 100.98 | 101.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 97.69 | 99.27 | 99.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 97.14 | 98.84 | 99.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 13:15:00 | 92.28 | 93.78 | 95.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 92.20 | 92.04 | 93.22 | SL hit (close>ema200) qty=0.50 sl=92.04 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 95.29 | 93.96 | 93.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 95.81 | 94.33 | 94.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 100.80 | 101.10 | 100.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:00:00 | 100.80 | 101.10 | 100.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 100.50 | 100.98 | 100.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 101.14 | 101.09 | 100.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 101.18 | 101.19 | 100.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 99.81 | 100.75 | 100.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 99.81 | 100.75 | 100.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 99.74 | 100.48 | 100.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 100.52 | 100.48 | 100.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 100.52 | 100.48 | 100.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 100.52 | 100.48 | 100.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 100.46 | 100.48 | 100.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 100.57 | 100.50 | 100.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 100.68 | 100.50 | 100.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 100.60 | 100.52 | 100.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:45:00 | 100.70 | 100.52 | 100.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 100.39 | 100.50 | 100.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:30:00 | 100.31 | 100.56 | 100.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 101.08 | 100.67 | 100.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 101.08 | 100.67 | 100.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 101.50 | 100.89 | 100.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 101.96 | 102.05 | 101.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:30:00 | 101.96 | 102.05 | 101.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 102.05 | 102.05 | 101.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 101.91 | 102.05 | 101.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 101.60 | 101.96 | 101.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 101.60 | 101.96 | 101.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 102.31 | 102.03 | 101.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 102.48 | 102.03 | 101.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 102.40 | 102.23 | 101.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-09 09:15:00 | 112.73 | 110.32 | 109.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 110.47 | 110.94 | 111.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 109.65 | 110.38 | 110.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 106.99 | 106.66 | 107.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 106.99 | 106.66 | 107.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 107.38 | 106.88 | 107.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 107.90 | 106.88 | 107.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 107.10 | 106.93 | 107.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 106.84 | 106.89 | 107.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 101.50 | 103.38 | 104.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 103.67 | 103.44 | 104.51 | SL hit (close>ema200) qty=0.50 sl=103.44 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 105.94 | 104.05 | 103.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 107.12 | 106.11 | 105.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 106.41 | 106.67 | 106.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 106.41 | 106.67 | 106.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 106.41 | 106.67 | 106.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 106.41 | 106.67 | 106.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 111.05 | 112.60 | 111.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 111.05 | 112.60 | 111.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 111.20 | 112.32 | 111.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 111.20 | 112.32 | 111.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 111.26 | 112.11 | 111.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 111.10 | 112.11 | 111.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 110.10 | 111.16 | 111.19 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 112.01 | 111.05 | 111.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 10:15:00 | 112.40 | 111.32 | 111.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 11:15:00 | 111.79 | 112.08 | 111.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 111.79 | 112.08 | 111.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 111.79 | 112.08 | 111.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 111.79 | 112.08 | 111.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 111.73 | 112.01 | 111.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:45:00 | 112.07 | 111.96 | 111.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 112.10 | 111.97 | 111.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 111.20 | 111.70 | 111.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 111.20 | 111.70 | 111.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 110.88 | 111.44 | 111.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 12:15:00 | 110.51 | 109.82 | 110.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 12:15:00 | 110.51 | 109.82 | 110.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 110.51 | 109.82 | 110.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 110.51 | 109.82 | 110.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 110.64 | 109.98 | 110.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 110.90 | 109.98 | 110.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 110.32 | 110.05 | 110.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 109.95 | 110.05 | 110.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 111.61 | 110.35 | 110.46 | SL hit (close>static) qty=1.00 sl=110.70 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 110.67 | 110.53 | 110.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 112.88 | 111.03 | 110.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 10:15:00 | 113.81 | 113.99 | 113.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 10:30:00 | 113.88 | 113.99 | 113.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 113.17 | 113.73 | 113.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 113.17 | 113.73 | 113.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 113.75 | 113.73 | 113.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 113.87 | 113.75 | 113.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 114.49 | 113.72 | 113.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 113.94 | 113.78 | 113.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:00:00 | 113.90 | 113.80 | 113.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 113.37 | 113.72 | 113.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 113.37 | 113.72 | 113.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 113.37 | 113.65 | 113.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:30:00 | 113.41 | 113.65 | 113.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 113.37 | 113.59 | 113.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 112.41 | 113.32 | 113.31 | SL hit (close<static) qty=1.00 sl=113.10 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 112.61 | 113.18 | 113.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 110.93 | 112.49 | 112.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 109.89 | 109.61 | 110.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:45:00 | 110.05 | 109.61 | 110.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 110.37 | 109.39 | 110.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 110.26 | 109.39 | 110.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 110.57 | 109.63 | 110.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 110.60 | 109.63 | 110.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 108.55 | 108.88 | 109.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 108.35 | 108.78 | 109.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 108.32 | 108.55 | 109.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 109.40 | 108.42 | 108.49 | SL hit (close>static) qty=1.00 sl=109.38 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 109.15 | 108.56 | 108.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 110.76 | 109.00 | 108.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 107.98 | 109.25 | 109.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 107.98 | 109.25 | 109.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 107.98 | 109.25 | 109.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 108.12 | 109.25 | 109.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 108.11 | 109.02 | 108.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 107.18 | 109.02 | 108.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 104.22 | 108.06 | 108.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 103.16 | 104.65 | 105.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 104.17 | 104.10 | 105.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:30:00 | 103.99 | 104.10 | 105.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 104.23 | 104.34 | 104.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:15:00 | 103.79 | 104.37 | 104.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:45:00 | 104.09 | 103.79 | 104.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:00:00 | 104.10 | 103.88 | 104.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 104.00 | 104.09 | 104.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 104.06 | 104.08 | 104.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:15:00 | 104.11 | 104.08 | 104.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 104.24 | 104.11 | 104.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 104.24 | 104.11 | 104.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 12:15:00 | 104.47 | 104.18 | 104.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 104.07 | 104.18 | 104.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 104.07 | 104.18 | 104.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 104.07 | 104.18 | 104.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 104.07 | 104.18 | 104.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 103.60 | 104.06 | 104.10 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 105.45 | 104.34 | 104.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 106.20 | 105.20 | 104.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 106.64 | 106.66 | 106.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 106.64 | 106.66 | 106.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 106.37 | 106.60 | 106.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 106.37 | 106.60 | 106.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 106.39 | 106.52 | 106.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:45:00 | 106.42 | 106.52 | 106.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 106.25 | 106.46 | 106.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:45:00 | 106.35 | 106.46 | 106.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 106.29 | 106.43 | 106.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 106.48 | 106.40 | 106.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 106.90 | 106.37 | 106.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:45:00 | 106.74 | 106.88 | 106.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 106.71 | 107.18 | 107.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 106.71 | 107.18 | 107.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 106.16 | 106.87 | 107.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 102.09 | 101.91 | 102.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:30:00 | 102.34 | 101.91 | 102.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 102.24 | 101.78 | 102.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 102.24 | 101.78 | 102.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 102.23 | 101.87 | 102.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 102.23 | 101.87 | 102.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 102.30 | 101.95 | 102.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 102.04 | 101.95 | 102.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 103.23 | 102.40 | 102.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 103.75 | 102.67 | 102.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 102.75 | 102.82 | 102.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 103.66 | 102.82 | 102.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 103.79 | 103.01 | 102.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 104.00 | 103.27 | 102.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 104.05 | 103.43 | 103.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:45:00 | 104.00 | 103.91 | 103.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 103.10 | 103.39 | 103.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 103.10 | 103.39 | 103.42 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 103.67 | 103.45 | 103.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 104.47 | 103.71 | 103.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 13:15:00 | 104.24 | 104.25 | 104.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 14:00:00 | 104.24 | 104.25 | 104.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 109.86 | 108.67 | 108.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:00:00 | 110.75 | 109.32 | 108.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 112.27 | 112.96 | 112.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 112.27 | 112.96 | 112.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 111.88 | 112.75 | 112.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 112.04 | 112.00 | 112.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:45:00 | 111.90 | 112.00 | 112.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 109.77 | 109.35 | 109.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 109.77 | 109.35 | 109.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 109.52 | 109.38 | 109.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 109.79 | 109.38 | 109.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 109.90 | 109.49 | 109.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 111.22 | 109.49 | 109.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 111.41 | 109.87 | 110.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 111.80 | 109.87 | 110.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 111.60 | 110.22 | 110.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 113.47 | 110.87 | 110.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 112.15 | 112.25 | 111.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:45:00 | 111.96 | 112.25 | 111.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 111.39 | 112.08 | 111.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 111.39 | 112.08 | 111.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 112.05 | 112.08 | 111.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 112.59 | 112.14 | 111.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:15:00 | 112.65 | 112.22 | 111.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 112.44 | 113.76 | 113.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 112.44 | 113.76 | 113.77 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 114.14 | 113.64 | 113.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 116.50 | 114.51 | 114.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 116.42 | 116.77 | 116.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 116.42 | 116.77 | 116.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 116.42 | 116.77 | 116.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 116.53 | 116.77 | 116.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 115.70 | 116.55 | 116.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 115.70 | 116.55 | 116.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 115.34 | 116.31 | 116.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 115.34 | 116.31 | 116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 115.17 | 115.91 | 115.91 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 116.48 | 115.89 | 115.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 116.82 | 116.07 | 115.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 14:15:00 | 116.06 | 116.66 | 116.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 14:15:00 | 116.06 | 116.66 | 116.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 116.06 | 116.66 | 116.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 116.06 | 116.66 | 116.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 116.13 | 116.56 | 116.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 116.33 | 116.56 | 116.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 116.24 | 116.33 | 116.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 113.47 | 115.60 | 115.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 114.98 | 114.75 | 115.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 114.98 | 114.75 | 115.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 114.98 | 114.75 | 115.39 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 118.29 | 115.81 | 115.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 118.81 | 116.41 | 116.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 117.46 | 117.48 | 116.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:30:00 | 117.46 | 117.48 | 116.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 117.85 | 118.39 | 117.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 117.94 | 118.39 | 117.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 118.27 | 118.36 | 117.83 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 116.95 | 117.56 | 117.60 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 118.16 | 117.68 | 117.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 118.70 | 118.18 | 117.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 120.88 | 121.04 | 120.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 120.88 | 121.04 | 120.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 120.88 | 121.04 | 120.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 120.88 | 121.04 | 120.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 121.38 | 121.09 | 120.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 123.11 | 120.97 | 120.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 121.44 | 122.50 | 122.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 121.44 | 122.50 | 122.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 120.68 | 121.89 | 122.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 122.24 | 121.26 | 121.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 122.24 | 121.26 | 121.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 122.24 | 121.26 | 121.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 122.24 | 121.26 | 121.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 121.96 | 121.40 | 121.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 122.82 | 121.40 | 121.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 121.90 | 121.79 | 121.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:15:00 | 122.17 | 121.79 | 121.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 121.63 | 121.76 | 121.83 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 122.33 | 121.93 | 121.90 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 120.26 | 121.67 | 121.79 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 122.72 | 121.87 | 121.79 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 121.05 | 121.90 | 121.98 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 124.65 | 122.47 | 122.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 125.03 | 123.53 | 123.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 13:15:00 | 124.31 | 124.50 | 123.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 124.31 | 124.50 | 123.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 123.98 | 124.39 | 123.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:45:00 | 123.53 | 124.39 | 123.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 123.79 | 124.27 | 123.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 123.53 | 124.27 | 123.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 122.98 | 124.01 | 123.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 122.98 | 124.01 | 123.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 122.48 | 123.71 | 123.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 122.23 | 123.71 | 123.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 122.77 | 123.52 | 123.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 121.86 | 122.71 | 123.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 122.64 | 122.50 | 122.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 11:15:00 | 122.64 | 122.50 | 122.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 122.64 | 122.50 | 122.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 122.64 | 122.50 | 122.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 123.08 | 122.63 | 122.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 123.08 | 122.63 | 122.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 123.06 | 122.72 | 122.84 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 125.33 | 123.28 | 123.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 12:15:00 | 126.09 | 125.21 | 124.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 15:15:00 | 125.30 | 125.32 | 124.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 127.41 | 125.32 | 124.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 127.33 | 125.72 | 125.17 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 121.69 | 125.23 | 125.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 120.00 | 124.19 | 124.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 121.09 | 120.20 | 121.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 121.09 | 120.20 | 121.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 121.60 | 120.48 | 121.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 121.60 | 120.48 | 121.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 121.05 | 120.60 | 121.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 121.63 | 120.60 | 121.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 121.20 | 120.72 | 121.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 121.57 | 120.72 | 121.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 121.52 | 120.88 | 121.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 121.52 | 120.88 | 121.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 121.77 | 121.06 | 121.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 121.77 | 121.06 | 121.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 121.65 | 121.17 | 121.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 121.19 | 121.17 | 121.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 115.13 | 117.51 | 119.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 117.59 | 117.46 | 118.78 | SL hit (close>ema200) qty=0.50 sl=117.46 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 118.43 | 117.78 | 117.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 118.61 | 118.03 | 117.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 117.82 | 118.15 | 117.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 117.82 | 118.15 | 117.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 117.82 | 118.15 | 117.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 117.82 | 118.15 | 117.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 117.54 | 118.03 | 117.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 117.54 | 118.03 | 117.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 117.20 | 117.86 | 117.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 117.23 | 117.86 | 117.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 116.77 | 117.65 | 117.74 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 119.07 | 117.80 | 117.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 11:15:00 | 119.16 | 118.07 | 117.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 118.34 | 118.60 | 118.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 118.34 | 118.60 | 118.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 118.34 | 118.60 | 118.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 118.34 | 118.60 | 118.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 118.67 | 118.61 | 118.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 119.19 | 118.74 | 118.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 15:00:00 | 119.00 | 118.88 | 118.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 119.63 | 118.84 | 118.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 119.17 | 118.85 | 118.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 121.05 | 121.09 | 120.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 120.78 | 121.09 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 120.26 | 120.93 | 120.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 120.26 | 120.93 | 120.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 119.88 | 120.72 | 120.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 119.88 | 120.72 | 120.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 120.40 | 120.65 | 120.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 120.40 | 120.65 | 120.68 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 120.88 | 120.47 | 120.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 121.37 | 120.65 | 120.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 123.72 | 123.77 | 122.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:00:00 | 123.72 | 123.77 | 122.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 124.68 | 125.45 | 124.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 124.68 | 125.45 | 124.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 125.29 | 125.42 | 124.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 125.70 | 125.40 | 124.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:30:00 | 125.62 | 125.49 | 125.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 125.58 | 125.45 | 125.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 125.86 | 125.46 | 125.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 125.17 | 125.73 | 125.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 125.90 | 125.75 | 125.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 123.65 | 125.21 | 125.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 123.65 | 125.21 | 125.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 123.45 | 124.60 | 124.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 123.95 | 123.75 | 124.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 123.95 | 123.75 | 124.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 123.86 | 123.77 | 124.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 124.19 | 123.77 | 124.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 123.10 | 122.80 | 123.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 123.40 | 122.80 | 123.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 123.47 | 122.94 | 123.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 123.50 | 122.94 | 123.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 123.10 | 122.97 | 123.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 122.80 | 122.96 | 123.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 122.66 | 122.94 | 123.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 124.55 | 123.26 | 123.30 | SL hit (close>static) qty=1.00 sl=124.09 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 124.35 | 123.48 | 123.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 124.89 | 123.76 | 123.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 128.38 | 131.69 | 130.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 128.38 | 131.69 | 130.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 128.38 | 131.69 | 130.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 128.38 | 131.69 | 130.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 127.87 | 130.92 | 129.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 126.82 | 130.92 | 129.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 125.84 | 128.77 | 129.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 125.45 | 127.67 | 128.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 125.76 | 124.80 | 125.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 125.76 | 124.80 | 125.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 125.76 | 124.80 | 125.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 126.16 | 124.80 | 125.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 125.08 | 124.85 | 125.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 125.31 | 124.85 | 125.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 123.75 | 124.78 | 125.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:45:00 | 123.32 | 124.35 | 125.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:45:00 | 123.36 | 122.77 | 122.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:30:00 | 123.36 | 122.90 | 122.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:00:00 | 123.43 | 122.90 | 122.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 124.48 | 123.22 | 123.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 124.48 | 123.22 | 123.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 125.10 | 123.82 | 123.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 124.65 | 124.72 | 124.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 124.65 | 124.72 | 124.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 124.65 | 124.72 | 124.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 125.80 | 124.73 | 124.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 125.49 | 124.98 | 124.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 125.34 | 125.03 | 124.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 121.50 | 123.94 | 124.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 121.50 | 123.94 | 124.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 118.79 | 121.66 | 122.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 122.21 | 121.34 | 122.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 122.21 | 121.34 | 122.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 121.74 | 121.42 | 122.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 123.54 | 121.42 | 122.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 123.25 | 121.79 | 122.30 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 124.18 | 122.90 | 122.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 09:15:00 | 124.44 | 123.74 | 123.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 123.78 | 123.79 | 123.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:30:00 | 123.75 | 123.79 | 123.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 124.15 | 123.85 | 123.57 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 122.08 | 123.39 | 123.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 121.44 | 122.80 | 123.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 123.00 | 122.72 | 123.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 123.00 | 122.72 | 123.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 123.00 | 122.72 | 123.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 123.00 | 122.72 | 123.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 122.49 | 122.67 | 122.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 124.71 | 122.67 | 122.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 124.15 | 122.97 | 123.09 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 124.27 | 123.23 | 123.20 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 122.92 | 123.30 | 123.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 121.30 | 122.90 | 123.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 12:15:00 | 122.85 | 122.79 | 123.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:00:00 | 122.85 | 122.79 | 123.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 123.37 | 122.91 | 123.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 123.37 | 122.91 | 123.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 122.93 | 122.91 | 123.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 121.87 | 122.89 | 123.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 124.45 | 121.03 | 120.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 124.45 | 121.03 | 120.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 128.00 | 124.48 | 122.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 126.94 | 127.09 | 125.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:00:00 | 126.94 | 127.09 | 125.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 126.20 | 126.76 | 125.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 127.12 | 126.76 | 125.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 127.91 | 126.99 | 126.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 129.10 | 127.29 | 126.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 129.63 | 130.13 | 130.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 129.63 | 130.13 | 130.14 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 10:15:00 | 130.27 | 130.16 | 130.15 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 129.76 | 130.08 | 130.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 129.40 | 129.90 | 130.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 121.92 | 121.76 | 123.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 121.92 | 121.76 | 123.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 117.70 | 116.43 | 117.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 117.70 | 116.43 | 117.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 117.34 | 116.61 | 117.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:15:00 | 117.70 | 116.61 | 117.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 117.70 | 116.83 | 117.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 117.90 | 116.83 | 117.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 117.65 | 116.99 | 117.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 117.38 | 117.03 | 117.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:00:00 | 117.20 | 117.03 | 117.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 117.37 | 116.81 | 116.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 15:15:00 | 111.51 | 113.28 | 114.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 15:15:00 | 111.50 | 113.28 | 114.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 111.34 | 112.80 | 114.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 111.10 | 110.71 | 112.60 | SL hit (close>ema200) qty=0.50 sl=110.71 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 113.67 | 112.33 | 112.25 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 110.24 | 112.33 | 112.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 109.28 | 110.92 | 111.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 112.73 | 110.95 | 111.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 112.73 | 110.95 | 111.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 112.73 | 110.95 | 111.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 113.14 | 110.95 | 111.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 112.90 | 111.34 | 111.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 112.70 | 111.34 | 111.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 111.89 | 111.65 | 111.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 111.47 | 111.65 | 111.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 14:15:00 | 105.90 | 107.72 | 109.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 107.75 | 107.09 | 108.34 | SL hit (close>ema200) qty=0.50 sl=107.09 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 110.22 | 108.72 | 108.61 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 106.35 | 108.33 | 108.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 105.67 | 107.80 | 108.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 103.78 | 102.67 | 104.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 103.78 | 102.67 | 104.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 103.78 | 102.67 | 104.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 100.60 | 104.02 | 104.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 102.90 | 103.04 | 103.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 103.38 | 103.28 | 103.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 104.90 | 103.96 | 103.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 104.90 | 103.96 | 103.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 105.54 | 104.28 | 104.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 104.55 | 105.22 | 104.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 104.55 | 105.22 | 104.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 104.55 | 105.22 | 104.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 108.87 | 104.57 | 104.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 113.46 | 114.27 | 114.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 113.46 | 114.27 | 114.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 112.43 | 113.81 | 114.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 113.06 | 112.63 | 113.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 113.06 | 112.63 | 113.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 113.06 | 112.63 | 113.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 113.37 | 112.63 | 113.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 113.15 | 112.73 | 113.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 113.87 | 112.73 | 113.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 113.82 | 112.95 | 113.18 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 113.92 | 113.38 | 113.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 113.97 | 113.50 | 113.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 112.55 | 113.42 | 113.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 112.55 | 113.42 | 113.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 112.55 | 113.42 | 113.40 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 112.28 | 113.20 | 113.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 111.86 | 112.93 | 113.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 112.45 | 112.05 | 112.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 11:15:00 | 112.45 | 112.05 | 112.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 112.45 | 112.05 | 112.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:45:00 | 112.44 | 112.05 | 112.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 112.39 | 112.11 | 112.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 112.39 | 112.11 | 112.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 111.70 | 112.03 | 112.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:30:00 | 112.11 | 112.03 | 112.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 110.55 | 109.91 | 110.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 111.01 | 109.91 | 110.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 108.03 | 108.96 | 109.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 107.79 | 108.96 | 109.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 107.80 | 108.73 | 109.46 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 109.49 | 109.34 | 109.33 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 109.22 | 109.32 | 109.32 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 109.60 | 109.37 | 109.35 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 109.10 | 109.33 | 109.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 108.42 | 109.11 | 109.23 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 126.10 | 2024-05-16 12:15:00 | 123.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-05-22 13:15:00 | 125.85 | 2024-05-30 14:15:00 | 127.25 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2024-05-22 15:00:00 | 125.90 | 2024-05-30 14:15:00 | 127.25 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-06-11 09:15:00 | 126.26 | 2024-06-21 09:15:00 | 127.03 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2024-06-26 14:30:00 | 124.25 | 2024-06-27 14:15:00 | 118.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 14:30:00 | 124.25 | 2024-06-28 09:15:00 | 124.76 | STOP_HIT | 0.50 | -0.41% |
| SELL | retest2 | 2024-06-28 10:15:00 | 123.85 | 2024-07-05 09:15:00 | 122.76 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-07-23 12:15:00 | 115.51 | 2024-07-26 09:15:00 | 117.75 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-07-23 13:00:00 | 116.60 | 2024-07-26 09:15:00 | 117.75 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-23 14:15:00 | 117.13 | 2024-07-26 09:15:00 | 117.75 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-07-24 11:00:00 | 116.91 | 2024-07-26 09:15:00 | 117.75 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-07-31 12:15:00 | 124.80 | 2024-07-31 13:15:00 | 123.44 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-01 09:15:00 | 124.85 | 2024-08-01 11:15:00 | 123.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-08-08 14:30:00 | 114.43 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-08-09 10:00:00 | 114.36 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-12 09:15:00 | 114.46 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-08-12 09:45:00 | 114.09 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-08-12 12:15:00 | 115.03 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-08-12 14:45:00 | 114.69 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-08-13 09:45:00 | 114.88 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-08-13 10:15:00 | 115.00 | 2024-08-13 10:15:00 | 115.56 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-08-21 14:45:00 | 115.95 | 2024-08-26 10:15:00 | 115.88 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-09-11 11:45:00 | 109.18 | 2024-09-13 11:15:00 | 110.21 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-13 09:30:00 | 109.07 | 2024-09-13 11:15:00 | 110.21 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-09-13 10:15:00 | 109.30 | 2024-09-13 11:15:00 | 110.21 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-19 11:00:00 | 107.90 | 2024-09-23 09:15:00 | 109.19 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-09-20 13:30:00 | 107.61 | 2024-09-23 09:15:00 | 109.19 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-09-20 14:30:00 | 107.84 | 2024-09-23 09:15:00 | 109.19 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-09-24 11:30:00 | 109.36 | 2024-09-24 14:15:00 | 108.06 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-09-26 12:30:00 | 106.47 | 2024-09-26 13:15:00 | 107.21 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-04 13:15:00 | 106.00 | 2024-10-09 12:15:00 | 104.66 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2024-10-04 14:15:00 | 105.83 | 2024-10-09 12:15:00 | 104.66 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2024-10-07 09:30:00 | 105.61 | 2024-10-09 12:15:00 | 104.66 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-10-10 15:15:00 | 103.75 | 2024-10-10 15:15:00 | 103.75 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-10-16 09:15:00 | 105.10 | 2024-10-17 09:15:00 | 103.84 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-10-17 09:15:00 | 105.10 | 2024-10-17 09:15:00 | 103.84 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-10-18 13:45:00 | 103.36 | 2024-10-22 11:15:00 | 98.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 103.24 | 2024-10-22 11:15:00 | 98.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 103.24 | 2024-10-22 11:15:00 | 98.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 13:45:00 | 103.36 | 2024-10-23 09:15:00 | 93.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 103.24 | 2024-10-23 09:15:00 | 92.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 103.24 | 2024-10-23 09:15:00 | 92.92 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-29 14:00:00 | 99.48 | 2024-10-31 13:15:00 | 97.01 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-11-04 13:00:00 | 101.54 | 2024-11-11 09:15:00 | 103.71 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2024-12-16 12:00:00 | 107.90 | 2024-12-16 14:15:00 | 108.14 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-01-07 15:15:00 | 102.96 | 2025-01-13 09:15:00 | 97.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 102.76 | 2025-01-13 09:15:00 | 97.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 15:15:00 | 102.96 | 2025-01-14 10:15:00 | 98.11 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-01-08 10:15:00 | 102.76 | 2025-01-14 10:15:00 | 98.11 | STOP_HIT | 0.50 | 4.53% |
| BUY | retest2 | 2025-01-17 14:30:00 | 100.36 | 2025-01-22 09:15:00 | 98.71 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-01-17 15:15:00 | 100.30 | 2025-01-22 09:15:00 | 98.71 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-21 11:30:00 | 100.40 | 2025-01-22 09:15:00 | 98.71 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-01-21 14:45:00 | 100.21 | 2025-01-22 09:15:00 | 98.71 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-01-24 09:15:00 | 98.93 | 2025-01-28 13:15:00 | 99.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-07 11:15:00 | 100.10 | 2025-02-07 11:15:00 | 98.93 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-02-13 13:00:00 | 95.35 | 2025-02-17 09:15:00 | 90.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 95.35 | 2025-02-17 13:15:00 | 92.57 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2025-03-12 10:15:00 | 87.25 | 2025-03-18 10:15:00 | 88.70 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-03-12 10:45:00 | 86.58 | 2025-03-18 10:15:00 | 88.70 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-03-13 10:45:00 | 87.21 | 2025-03-18 10:15:00 | 88.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-13 15:00:00 | 87.17 | 2025-03-18 10:15:00 | 88.70 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-03-26 09:15:00 | 94.41 | 2025-03-26 12:15:00 | 93.56 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-03-26 10:15:00 | 94.40 | 2025-03-26 12:15:00 | 93.56 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-04-02 12:00:00 | 97.23 | 2025-04-07 09:15:00 | 93.50 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-04-02 15:00:00 | 97.40 | 2025-04-07 09:15:00 | 93.50 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-04-03 10:00:00 | 97.30 | 2025-04-07 09:15:00 | 93.50 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-04-04 10:30:00 | 97.45 | 2025-04-07 09:15:00 | 93.50 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-04-15 09:15:00 | 97.30 | 2025-04-25 09:15:00 | 100.43 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-04-15 12:45:00 | 96.49 | 2025-04-25 09:15:00 | 100.43 | STOP_HIT | 1.00 | 4.08% |
| SELL | retest2 | 2025-05-06 11:00:00 | 97.14 | 2025-05-08 13:15:00 | 92.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 97.14 | 2025-05-09 15:15:00 | 92.20 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2025-05-21 09:45:00 | 101.14 | 2025-05-22 13:15:00 | 99.81 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-05-21 12:15:00 | 101.18 | 2025-05-22 13:15:00 | 99.81 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-05-26 12:30:00 | 100.31 | 2025-05-26 13:15:00 | 101.08 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-29 15:15:00 | 102.48 | 2025-06-09 09:15:00 | 112.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-30 12:45:00 | 102.40 | 2025-06-09 09:15:00 | 112.64 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 106.84 | 2025-06-20 09:15:00 | 101.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 106.84 | 2025-06-20 10:15:00 | 103.67 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest2 | 2025-07-08 14:45:00 | 112.07 | 2025-07-09 13:15:00 | 111.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-09 09:15:00 | 112.10 | 2025-07-09 13:15:00 | 111.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-11 15:15:00 | 109.95 | 2025-07-14 09:15:00 | 111.61 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-07-17 14:30:00 | 113.87 | 2025-07-21 09:15:00 | 112.41 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-18 09:15:00 | 114.49 | 2025-07-21 09:15:00 | 112.41 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-07-18 11:15:00 | 113.94 | 2025-07-21 09:15:00 | 112.41 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-18 12:00:00 | 113.90 | 2025-07-21 09:15:00 | 112.41 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-28 12:00:00 | 108.35 | 2025-07-29 15:15:00 | 109.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-28 12:30:00 | 108.32 | 2025-07-29 15:15:00 | 109.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-08-05 13:15:00 | 103.79 | 2025-08-08 11:15:00 | 104.24 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-06 13:45:00 | 104.09 | 2025-08-08 11:15:00 | 104.24 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-08-07 10:00:00 | 104.10 | 2025-08-08 11:15:00 | 104.24 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-08-08 09:45:00 | 104.00 | 2025-08-08 11:15:00 | 104.24 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-08-14 14:45:00 | 106.48 | 2025-08-22 09:15:00 | 106.71 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-08-18 09:15:00 | 106.90 | 2025-08-22 09:15:00 | 106.71 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-08-19 09:45:00 | 106.74 | 2025-08-22 09:15:00 | 106.71 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-09-03 12:15:00 | 104.00 | 2025-09-05 12:15:00 | 103.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-03 13:00:00 | 104.05 | 2025-09-05 12:15:00 | 103.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-04 10:45:00 | 104.00 | 2025-09-05 12:15:00 | 103.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-17 13:00:00 | 110.75 | 2025-09-24 12:15:00 | 112.27 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2025-10-01 13:45:00 | 112.59 | 2025-10-08 10:15:00 | 112.44 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-10-01 15:15:00 | 112.65 | 2025-10-08 10:15:00 | 112.44 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-10-31 11:15:00 | 123.11 | 2025-11-06 11:15:00 | 121.44 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-08 09:15:00 | 121.19 | 2025-12-09 09:15:00 | 115.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 121.19 | 2025-12-09 11:15:00 | 117.59 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest2 | 2025-12-18 11:45:00 | 119.19 | 2025-12-26 11:15:00 | 120.40 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-12-18 15:00:00 | 119.00 | 2025-12-26 11:15:00 | 120.40 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-12-19 09:15:00 | 119.63 | 2025-12-26 11:15:00 | 120.40 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-12-19 10:15:00 | 119.17 | 2025-12-26 11:15:00 | 120.40 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2026-01-06 11:30:00 | 125.70 | 2026-01-08 10:15:00 | 123.65 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-01-06 12:30:00 | 125.62 | 2026-01-08 10:15:00 | 123.65 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-01-06 15:00:00 | 125.58 | 2026-01-08 10:15:00 | 123.65 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-07 09:15:00 | 125.86 | 2026-01-08 10:15:00 | 123.65 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-07 14:45:00 | 125.90 | 2026-01-08 10:15:00 | 123.65 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-01-13 12:00:00 | 122.80 | 2026-01-13 14:15:00 | 124.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-13 12:45:00 | 122.66 | 2026-01-13 14:15:00 | 124.55 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-01-23 11:45:00 | 123.32 | 2026-01-28 14:15:00 | 124.48 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-01-28 12:45:00 | 123.36 | 2026-01-28 14:15:00 | 124.48 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-28 13:30:00 | 123.36 | 2026-01-28 14:15:00 | 124.48 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-28 14:00:00 | 123.43 | 2026-01-28 14:15:00 | 124.48 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-30 11:15:00 | 125.80 | 2026-02-01 11:15:00 | 121.50 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-01-30 13:30:00 | 125.49 | 2026-02-01 11:15:00 | 121.50 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-01-30 14:45:00 | 125.34 | 2026-02-01 11:15:00 | 121.50 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-02-12 09:15:00 | 121.87 | 2026-02-17 10:15:00 | 124.45 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-02-20 11:15:00 | 129.10 | 2026-02-27 09:15:00 | 129.63 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2026-03-11 10:30:00 | 117.38 | 2026-03-13 15:15:00 | 111.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:00:00 | 117.20 | 2026-03-13 15:15:00 | 111.50 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2026-03-12 14:15:00 | 117.37 | 2026-03-16 09:15:00 | 111.34 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2026-03-11 10:30:00 | 117.38 | 2026-03-16 14:15:00 | 111.10 | STOP_HIT | 0.50 | 5.35% |
| SELL | retest2 | 2026-03-11 11:00:00 | 117.20 | 2026-03-16 14:15:00 | 111.10 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2026-03-12 14:15:00 | 117.37 | 2026-03-16 14:15:00 | 111.10 | STOP_HIT | 0.50 | 5.34% |
| SELL | retest2 | 2026-03-20 14:15:00 | 111.47 | 2026-03-23 14:15:00 | 105.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:15:00 | 111.47 | 2026-03-24 12:15:00 | 107.75 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-04-02 09:15:00 | 100.60 | 2026-04-06 11:15:00 | 104.90 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2026-04-02 14:30:00 | 102.90 | 2026-04-06 11:15:00 | 104.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-06 09:15:00 | 103.38 | 2026-04-06 11:15:00 | 104.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-04-08 09:15:00 | 108.87 | 2026-04-23 10:15:00 | 113.46 | STOP_HIT | 1.00 | 4.22% |
