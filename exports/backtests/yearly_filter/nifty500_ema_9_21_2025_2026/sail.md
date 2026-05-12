# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 184.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 51 |
| ALERT2 | 49 |
| ALERT2_SKIP | 23 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 29
- **Target hits / Stop hits / Partials:** 2 / 42 / 2
- **Avg / median % per leg:** -0.06% / -0.76%
- **Sum % (uncompounded):** -2.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 9 | 34.6% | 2 | 24 | 0 | -0.24% | -6.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 2 | 24 | 0 | -0.24% | -6.3% |
| SELL (all) | 20 | 8 | 40.0% | 0 | 18 | 2 | 0.16% | 3.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 8 | 40.0% | 0 | 18 | 2 | 0.16% | 3.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 17 | 37.0% | 2 | 42 | 2 | -0.06% | -3.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 114.73 | 111.86 | 111.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 117.38 | 112.96 | 112.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 116.45 | 116.50 | 114.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 116.38 | 116.50 | 114.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 122.09 | 122.22 | 121.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 121.60 | 122.22 | 121.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 122.63 | 123.27 | 122.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 122.63 | 123.27 | 122.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 122.68 | 123.15 | 122.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 123.03 | 123.33 | 122.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 127.10 | 128.95 | 128.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 127.10 | 128.95 | 128.97 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 129.69 | 128.95 | 128.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 130.54 | 129.42 | 129.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 130.55 | 130.77 | 130.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 130.55 | 130.77 | 130.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 130.33 | 130.68 | 130.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 130.18 | 130.68 | 130.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 129.90 | 130.53 | 130.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 129.90 | 130.53 | 130.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 130.19 | 130.46 | 130.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 130.55 | 130.62 | 130.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 133.65 | 134.12 | 133.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 134.19 | 134.12 | 133.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 134.11 | 134.18 | 133.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 133.10 | 133.87 | 133.68 | SL hit (close<static) qty=1.00 sl=133.55 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 132.81 | 133.43 | 133.50 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 134.04 | 133.54 | 133.53 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 133.25 | 133.50 | 133.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 132.86 | 133.37 | 133.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 129.65 | 129.12 | 130.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 129.65 | 129.12 | 130.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 129.01 | 129.03 | 129.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 129.85 | 129.03 | 129.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 126.05 | 125.55 | 126.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 126.00 | 125.55 | 126.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 127.21 | 125.88 | 126.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 127.21 | 125.88 | 126.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 127.11 | 126.13 | 126.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 127.15 | 126.13 | 126.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 126.55 | 126.26 | 126.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 126.61 | 126.26 | 126.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 127.53 | 126.52 | 126.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 127.53 | 126.52 | 126.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 127.27 | 126.67 | 126.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 126.71 | 126.67 | 126.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 126.94 | 126.72 | 126.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 126.94 | 126.72 | 126.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 128.22 | 127.02 | 126.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 126.71 | 127.05 | 126.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 126.71 | 127.05 | 126.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 126.71 | 127.05 | 126.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 126.80 | 127.05 | 126.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 126.54 | 126.95 | 126.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 128.87 | 126.95 | 126.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 130.56 | 131.54 | 131.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 130.56 | 131.54 | 131.57 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 131.79 | 131.59 | 131.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 12:15:00 | 132.28 | 131.73 | 131.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 136.48 | 136.71 | 135.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:30:00 | 136.85 | 136.71 | 135.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 135.93 | 136.46 | 135.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:45:00 | 136.56 | 136.49 | 135.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 134.33 | 136.06 | 135.49 | SL hit (close<static) qty=1.00 sl=135.07 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 134.80 | 135.17 | 135.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 133.50 | 134.84 | 135.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 133.66 | 133.62 | 134.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 134.58 | 133.62 | 134.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 134.49 | 133.80 | 134.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 134.47 | 133.80 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 134.26 | 133.89 | 134.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 134.64 | 133.89 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 134.16 | 133.94 | 134.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 134.20 | 133.94 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 134.78 | 134.11 | 134.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 134.78 | 134.11 | 134.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 134.53 | 134.19 | 134.30 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 135.11 | 134.51 | 134.44 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 132.44 | 134.10 | 134.25 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 134.32 | 134.02 | 133.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 15:15:00 | 134.90 | 134.40 | 134.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 134.41 | 134.71 | 134.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 134.41 | 134.71 | 134.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 134.41 | 134.71 | 134.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 134.41 | 134.71 | 134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 134.50 | 134.67 | 134.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 134.50 | 134.67 | 134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 133.59 | 134.46 | 134.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 134.19 | 134.46 | 134.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 135.00 | 134.56 | 134.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 135.30 | 134.56 | 134.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 133.36 | 134.23 | 134.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 133.36 | 134.23 | 134.32 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 135.00 | 134.31 | 134.27 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 132.94 | 134.09 | 134.18 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 134.94 | 133.80 | 133.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 136.59 | 134.88 | 134.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 136.71 | 137.09 | 136.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 136.71 | 137.09 | 136.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 136.71 | 137.09 | 136.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 137.05 | 137.09 | 136.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 136.81 | 137.04 | 136.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 136.92 | 137.04 | 136.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 136.10 | 136.85 | 136.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 136.10 | 136.85 | 136.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 135.77 | 136.63 | 136.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 135.72 | 136.63 | 136.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 135.30 | 136.15 | 136.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 135.30 | 136.15 | 136.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 134.41 | 135.80 | 135.94 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 136.38 | 135.74 | 135.70 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 132.18 | 135.10 | 135.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 130.79 | 134.24 | 135.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 126.14 | 126.03 | 127.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 125.19 | 126.03 | 127.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 123.95 | 122.12 | 123.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 123.43 | 122.12 | 123.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 123.51 | 122.40 | 123.28 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 125.44 | 123.97 | 123.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 125.70 | 124.31 | 123.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 124.89 | 124.97 | 124.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 124.89 | 124.97 | 124.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 124.89 | 124.97 | 124.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 124.38 | 124.74 | 124.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 123.03 | 124.40 | 124.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 123.03 | 124.40 | 124.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 123.01 | 124.12 | 124.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 121.65 | 123.24 | 123.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 123.35 | 122.37 | 123.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 123.35 | 122.37 | 123.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 123.35 | 122.37 | 123.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 123.35 | 122.37 | 123.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 123.75 | 122.65 | 123.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 123.20 | 122.65 | 123.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 123.05 | 122.00 | 121.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 123.05 | 122.00 | 121.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 123.80 | 122.57 | 122.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 122.48 | 122.71 | 122.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 122.48 | 122.71 | 122.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 122.48 | 122.71 | 122.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 122.48 | 122.71 | 122.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 122.41 | 122.65 | 122.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 122.43 | 122.65 | 122.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 122.35 | 122.59 | 122.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 122.35 | 122.59 | 122.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 124.15 | 122.90 | 122.57 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 121.17 | 122.29 | 122.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 121.05 | 122.04 | 122.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 122.72 | 121.56 | 121.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 122.72 | 121.56 | 121.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 122.72 | 121.56 | 121.90 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 122.72 | 122.12 | 122.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 123.44 | 122.48 | 122.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 122.86 | 122.90 | 122.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 122.86 | 122.90 | 122.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 122.86 | 122.90 | 122.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 122.86 | 122.90 | 122.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 125.18 | 123.36 | 122.87 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 122.05 | 123.27 | 123.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 120.87 | 122.37 | 122.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 119.85 | 119.46 | 120.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 120.02 | 119.46 | 120.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 119.60 | 119.22 | 119.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 119.60 | 119.22 | 119.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 119.90 | 119.35 | 119.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 119.90 | 119.35 | 119.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 120.40 | 119.56 | 119.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 120.40 | 119.56 | 119.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 121.12 | 120.17 | 120.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 122.50 | 120.79 | 120.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 127.34 | 127.77 | 126.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 127.34 | 127.77 | 126.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 130.47 | 131.14 | 129.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 130.73 | 131.14 | 129.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 129.81 | 130.71 | 129.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 129.81 | 130.71 | 129.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 130.16 | 130.60 | 129.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 131.15 | 130.31 | 129.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 130.75 | 130.21 | 130.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 132.01 | 132.46 | 132.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 132.01 | 132.46 | 132.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 131.60 | 132.19 | 132.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 13:15:00 | 132.27 | 132.20 | 132.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 13:15:00 | 132.27 | 132.20 | 132.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 132.27 | 132.20 | 132.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 132.54 | 132.20 | 132.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 132.75 | 132.31 | 132.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 132.75 | 132.31 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 133.54 | 132.56 | 132.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 135.49 | 133.15 | 132.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 135.50 | 136.09 | 135.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 135.32 | 136.09 | 135.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 136.24 | 136.03 | 135.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 137.83 | 136.90 | 136.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:30:00 | 137.85 | 137.20 | 136.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 138.37 | 137.09 | 136.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 137.71 | 137.20 | 136.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 136.49 | 137.09 | 136.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 136.49 | 137.09 | 136.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 136.52 | 136.98 | 136.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 134.76 | 136.98 | 136.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 134.31 | 136.45 | 136.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 134.31 | 136.45 | 136.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 133.18 | 135.79 | 136.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 133.22 | 132.93 | 134.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 133.22 | 132.93 | 134.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 134.55 | 132.97 | 133.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 134.55 | 132.97 | 133.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 133.93 | 133.16 | 133.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:00:00 | 133.28 | 133.19 | 133.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 135.55 | 134.04 | 133.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 135.55 | 134.04 | 133.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 137.22 | 135.37 | 134.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 134.95 | 135.72 | 135.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 134.95 | 135.72 | 135.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 134.95 | 135.72 | 135.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 134.95 | 135.72 | 135.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 134.95 | 135.57 | 135.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 134.00 | 135.57 | 135.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 136.02 | 135.66 | 135.15 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 132.17 | 134.52 | 134.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 131.76 | 132.72 | 133.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 132.76 | 132.46 | 133.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 132.76 | 132.46 | 133.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 132.60 | 132.48 | 133.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 131.90 | 132.48 | 133.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 134.79 | 132.28 | 132.51 | SL hit (close>static) qty=1.00 sl=133.13 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 136.28 | 133.08 | 132.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 136.63 | 134.34 | 133.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 133.56 | 134.53 | 133.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 133.56 | 134.53 | 133.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 133.56 | 134.53 | 133.82 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 132.27 | 133.45 | 133.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 132.20 | 133.20 | 133.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 132.50 | 131.90 | 132.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 132.50 | 131.90 | 132.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 132.50 | 131.90 | 132.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 132.50 | 131.90 | 132.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 132.65 | 132.05 | 132.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 133.41 | 132.05 | 132.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 131.97 | 132.03 | 132.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 131.70 | 132.03 | 132.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 131.79 | 130.92 | 130.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 131.79 | 130.92 | 130.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 132.42 | 131.22 | 131.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 131.20 | 131.26 | 131.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 15:15:00 | 131.20 | 131.26 | 131.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 131.20 | 131.26 | 131.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 131.63 | 131.26 | 131.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 130.90 | 131.19 | 131.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:15:00 | 130.43 | 131.19 | 131.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 131.34 | 131.22 | 131.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 131.34 | 131.22 | 131.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 129.38 | 130.97 | 131.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 129.16 | 130.61 | 130.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 130.10 | 129.60 | 130.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 130.10 | 129.60 | 130.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 130.10 | 129.60 | 130.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 130.10 | 129.60 | 130.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 129.91 | 129.66 | 130.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 130.15 | 129.66 | 130.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 129.83 | 129.70 | 130.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 129.00 | 129.72 | 129.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 131.44 | 129.95 | 129.97 | SL hit (close>static) qty=1.00 sl=130.12 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 11:15:00 | 130.38 | 130.04 | 130.01 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 129.90 | 129.97 | 129.98 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 130.49 | 130.01 | 129.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 131.79 | 130.37 | 130.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 137.89 | 138.61 | 135.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 10:15:00 | 136.70 | 138.22 | 136.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 136.70 | 138.22 | 136.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 136.45 | 138.22 | 136.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 136.15 | 137.81 | 136.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 135.83 | 137.81 | 136.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 136.27 | 137.50 | 136.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:15:00 | 136.05 | 137.50 | 136.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 136.34 | 137.27 | 136.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 136.02 | 137.27 | 136.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 137.76 | 137.32 | 136.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:30:00 | 138.53 | 137.49 | 136.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 139.62 | 138.24 | 137.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 139.05 | 138.13 | 137.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 136.78 | 137.67 | 137.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 136.78 | 137.67 | 137.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 135.77 | 137.31 | 137.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 137.96 | 137.42 | 137.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 137.96 | 137.42 | 137.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 137.96 | 137.42 | 137.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 138.12 | 137.42 | 137.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 141.30 | 138.20 | 137.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 142.20 | 139.96 | 138.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 142.15 | 142.88 | 141.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:45:00 | 142.90 | 142.88 | 141.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 143.35 | 143.67 | 143.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 143.35 | 143.67 | 143.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 143.00 | 143.53 | 143.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 144.25 | 143.53 | 143.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 144.02 | 143.63 | 143.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 143.87 | 143.94 | 143.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 142.72 | 143.70 | 143.54 | SL hit (close<static) qty=1.00 sl=142.90 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 141.84 | 143.28 | 143.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 141.29 | 141.89 | 142.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 139.84 | 139.29 | 140.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:45:00 | 139.50 | 139.29 | 140.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 140.15 | 139.46 | 140.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 140.31 | 139.46 | 140.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 140.15 | 139.60 | 140.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 140.30 | 139.60 | 140.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 139.95 | 139.67 | 140.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 140.60 | 139.67 | 140.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 140.12 | 139.76 | 140.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 140.12 | 139.76 | 140.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 139.80 | 139.77 | 140.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 139.50 | 139.77 | 140.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 138.57 | 139.53 | 140.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:00:00 | 138.18 | 139.02 | 139.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 138.40 | 138.84 | 139.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 13:15:00 | 131.27 | 132.52 | 134.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 13:15:00 | 131.48 | 132.52 | 134.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 132.60 | 132.48 | 133.73 | SL hit (close>ema200) qty=0.50 sl=132.48 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 136.80 | 134.51 | 134.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 137.31 | 135.65 | 135.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 136.13 | 136.28 | 135.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 136.13 | 136.28 | 135.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 135.10 | 136.11 | 135.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 135.10 | 136.11 | 135.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 136.86 | 136.26 | 135.87 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 134.80 | 135.71 | 135.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 133.61 | 134.96 | 135.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 132.07 | 131.93 | 132.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 132.07 | 131.93 | 132.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 133.81 | 132.32 | 132.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 133.81 | 132.32 | 132.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 133.39 | 132.53 | 133.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 133.50 | 132.53 | 133.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 132.26 | 132.58 | 132.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 131.88 | 132.56 | 132.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 131.61 | 132.42 | 132.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 131.51 | 132.15 | 132.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 131.16 | 129.58 | 129.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 131.16 | 129.58 | 129.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 132.44 | 130.96 | 130.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 130.65 | 131.27 | 130.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 130.65 | 131.27 | 130.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 130.65 | 131.27 | 130.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 130.65 | 131.27 | 130.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 131.17 | 131.25 | 130.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 131.80 | 131.25 | 130.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 130.25 | 131.45 | 131.07 | SL hit (close<static) qty=1.00 sl=130.50 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 128.95 | 130.66 | 130.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 128.50 | 129.93 | 130.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 129.79 | 129.71 | 130.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 13:15:00 | 129.85 | 129.73 | 130.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 129.85 | 129.73 | 130.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 129.85 | 129.73 | 130.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 130.28 | 129.84 | 130.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 130.28 | 129.84 | 130.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 129.81 | 129.84 | 130.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 129.22 | 129.84 | 130.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 129.54 | 129.64 | 129.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 129.47 | 126.61 | 127.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 130.46 | 127.38 | 127.72 | SL hit (close>static) qty=1.00 sl=130.32 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 130.53 | 128.01 | 127.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 131.11 | 128.63 | 128.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 132.52 | 132.58 | 131.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 132.52 | 132.58 | 131.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 131.28 | 132.26 | 131.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 132.55 | 132.26 | 131.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 132.00 | 132.20 | 131.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 137.23 | 132.21 | 131.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-05 14:15:00 | 150.95 | 148.79 | 147.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 146.46 | 147.80 | 147.93 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 147.89 | 146.46 | 146.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 149.30 | 147.03 | 146.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 147.32 | 147.87 | 147.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:45:00 | 147.59 | 147.87 | 147.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 147.39 | 147.77 | 147.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 147.39 | 147.77 | 147.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 147.00 | 147.62 | 147.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 147.38 | 147.62 | 147.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 147.70 | 147.63 | 147.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 148.52 | 147.65 | 147.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 148.69 | 149.60 | 149.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 148.69 | 149.60 | 149.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 147.59 | 148.69 | 149.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 147.41 | 147.23 | 148.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 09:45:00 | 147.32 | 147.23 | 148.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 151.55 | 147.36 | 147.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 153.29 | 147.36 | 147.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 151.17 | 148.12 | 147.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 156.04 | 151.30 | 149.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 150.43 | 151.77 | 150.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 150.43 | 151.77 | 150.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 150.43 | 151.77 | 150.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 150.43 | 151.77 | 150.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 149.12 | 151.24 | 150.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 149.14 | 151.24 | 150.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 148.89 | 150.77 | 150.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 149.94 | 150.77 | 150.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 150.64 | 154.06 | 154.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 150.64 | 154.06 | 154.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 148.19 | 152.88 | 153.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 147.95 | 146.61 | 148.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 147.95 | 146.61 | 148.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 147.95 | 146.61 | 148.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 147.95 | 146.61 | 148.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 149.49 | 147.19 | 148.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 150.75 | 147.19 | 148.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 150.23 | 147.79 | 148.82 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 154.84 | 150.34 | 149.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 157.07 | 153.27 | 151.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 155.11 | 155.79 | 153.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 155.11 | 155.79 | 153.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 155.11 | 155.79 | 153.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:15:00 | 156.00 | 155.81 | 154.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 157.08 | 160.28 | 160.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 157.08 | 160.28 | 160.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 154.25 | 157.38 | 158.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 157.26 | 156.87 | 157.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 157.26 | 156.87 | 157.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 160.32 | 157.60 | 158.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 159.96 | 157.60 | 158.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 159.75 | 158.03 | 158.18 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 159.75 | 158.37 | 158.32 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 156.20 | 158.16 | 158.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 155.80 | 157.69 | 158.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 158.09 | 157.56 | 157.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 158.09 | 157.56 | 157.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 158.09 | 157.56 | 157.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 158.26 | 157.56 | 157.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 159.01 | 157.85 | 158.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 158.84 | 157.85 | 158.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 158.69 | 158.02 | 158.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 159.41 | 158.02 | 158.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 158.93 | 158.26 | 158.23 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 156.70 | 157.93 | 158.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 156.11 | 157.23 | 157.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 09:15:00 | 157.73 | 157.21 | 157.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 157.73 | 157.21 | 157.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 157.73 | 157.21 | 157.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 157.81 | 157.21 | 157.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 157.00 | 157.17 | 157.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 158.32 | 157.17 | 157.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 156.28 | 156.99 | 157.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:30:00 | 156.03 | 156.58 | 157.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 160.43 | 157.33 | 157.41 | SL hit (close>static) qty=1.00 sl=157.41 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 160.01 | 157.86 | 157.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 166.66 | 159.62 | 158.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 163.19 | 163.85 | 161.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:00:00 | 163.19 | 163.85 | 161.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 165.60 | 165.71 | 164.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 165.60 | 165.71 | 164.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 166.82 | 165.82 | 164.66 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 157.88 | 163.92 | 164.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 154.54 | 162.05 | 163.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 158.68 | 157.49 | 160.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 158.68 | 157.49 | 160.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 159.29 | 157.85 | 159.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 159.75 | 157.85 | 159.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 149.62 | 149.79 | 152.36 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 154.22 | 152.38 | 152.20 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 151.54 | 152.17 | 152.20 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 152.67 | 152.27 | 152.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 153.58 | 152.55 | 152.37 | Break + close above crossover candle high |

### Cycle 64 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 147.44 | 151.79 | 152.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 145.88 | 150.61 | 151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 149.60 | 148.95 | 150.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 15:00:00 | 149.60 | 148.95 | 150.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 149.10 | 148.98 | 150.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 147.96 | 148.98 | 150.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 148.45 | 147.15 | 148.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 151.68 | 149.07 | 148.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 151.68 | 149.07 | 148.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 153.49 | 149.96 | 149.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 153.45 | 153.57 | 151.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 152.80 | 153.57 | 151.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 152.85 | 153.42 | 152.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 153.94 | 153.49 | 152.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 155.88 | 153.16 | 152.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 147.36 | 153.91 | 153.77 | SL hit (close<static) qty=1.00 sl=151.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 147.24 | 152.58 | 153.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 145.34 | 151.13 | 152.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 146.88 | 145.95 | 148.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 146.88 | 145.95 | 148.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 149.90 | 146.76 | 147.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 150.40 | 146.76 | 147.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 149.71 | 147.35 | 148.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 150.21 | 147.35 | 148.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 150.81 | 148.61 | 148.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 151.91 | 149.27 | 148.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 149.43 | 149.62 | 149.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 149.42 | 149.62 | 149.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 148.48 | 149.39 | 149.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 148.48 | 149.39 | 149.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 146.52 | 148.82 | 148.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 146.52 | 148.82 | 148.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 146.58 | 148.37 | 148.60 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 10:15:00 | 152.54 | 149.19 | 148.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 11:15:00 | 156.15 | 150.58 | 149.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 14:15:00 | 151.20 | 151.53 | 150.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:45:00 | 151.56 | 151.53 | 150.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 151.75 | 154.22 | 152.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 151.75 | 154.22 | 152.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 151.15 | 153.61 | 152.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 150.76 | 153.61 | 152.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 155.70 | 154.46 | 153.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 157.17 | 155.22 | 154.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 14:15:00 | 172.89 | 171.46 | 169.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 186.19 | 186.93 | 186.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 184.69 | 186.30 | 186.67 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:30:00 | 114.46 | 2025-05-12 11:15:00 | 114.73 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-21 09:30:00 | 123.03 | 2025-06-02 09:15:00 | 127.10 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2025-06-10 09:15:00 | 134.19 | 2025-06-10 12:15:00 | 133.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-10 11:00:00 | 134.11 | 2025-06-10 12:15:00 | 133.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-23 09:15:00 | 126.71 | 2025-06-23 11:15:00 | 126.94 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-06-24 09:15:00 | 128.87 | 2025-07-01 10:15:00 | 130.56 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-07-04 10:45:00 | 136.56 | 2025-07-04 11:15:00 | 134.33 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-14 10:15:00 | 135.30 | 2025-07-14 12:15:00 | 133.36 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-08-08 09:15:00 | 123.20 | 2025-08-12 11:15:00 | 123.05 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-09-10 10:00:00 | 131.15 | 2025-09-18 10:15:00 | 132.01 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-09-11 09:30:00 | 130.75 | 2025-09-18 10:15:00 | 132.01 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-09-24 09:30:00 | 137.83 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-09-24 13:30:00 | 137.85 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-09-25 09:45:00 | 138.37 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-09-25 11:15:00 | 137.71 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-09-30 14:00:00 | 133.28 | 2025-10-01 09:15:00 | 135.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-10-08 09:15:00 | 131.90 | 2025-10-09 09:15:00 | 134.79 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-10-14 10:15:00 | 131.70 | 2025-10-16 12:15:00 | 131.79 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-23 15:15:00 | 129.00 | 2025-10-24 09:15:00 | 131.44 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-11-03 10:30:00 | 138.53 | 2025-11-06 11:15:00 | 136.78 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-04 09:15:00 | 139.62 | 2025-11-06 11:15:00 | 136.78 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-11-06 09:15:00 | 139.05 | 2025-11-06 11:15:00 | 136.78 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-13 09:15:00 | 144.25 | 2025-11-14 11:15:00 | 142.72 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-13 10:00:00 | 144.02 | 2025-11-14 11:15:00 | 142.72 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-13 13:00:00 | 143.87 | 2025-11-14 11:15:00 | 142.72 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-20 13:00:00 | 138.18 | 2025-11-25 13:15:00 | 131.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 138.40 | 2025-11-25 13:15:00 | 131.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:00:00 | 138.18 | 2025-11-25 15:15:00 | 132.60 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-11-20 14:30:00 | 138.40 | 2025-11-25 15:15:00 | 132.60 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2025-12-04 13:45:00 | 131.88 | 2025-12-12 09:15:00 | 131.16 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-12-05 09:15:00 | 131.61 | 2025-12-12 09:15:00 | 131.16 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-12-08 09:45:00 | 131.51 | 2025-12-12 09:15:00 | 131.16 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-12-15 13:15:00 | 131.80 | 2025-12-16 09:15:00 | 130.25 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-18 09:15:00 | 129.22 | 2025-12-22 10:15:00 | 130.46 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-12-18 13:00:00 | 129.54 | 2025-12-22 10:15:00 | 130.46 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-22 09:30:00 | 129.47 | 2025-12-22 10:15:00 | 130.46 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-29 09:15:00 | 137.23 | 2026-01-05 14:15:00 | 150.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-14 09:15:00 | 148.52 | 2026-01-19 10:15:00 | 148.69 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-01-27 09:15:00 | 149.94 | 2026-01-30 11:15:00 | 150.64 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2026-02-05 11:15:00 | 156.00 | 2026-02-13 09:15:00 | 157.08 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-02-24 12:30:00 | 156.03 | 2026-02-24 14:15:00 | 160.43 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-03-16 09:15:00 | 147.96 | 2026-03-17 13:15:00 | 151.68 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-03-17 11:15:00 | 148.45 | 2026-03-17 13:15:00 | 151.68 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-03-19 10:30:00 | 153.94 | 2026-03-23 09:15:00 | 147.36 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-03-20 09:15:00 | 155.88 | 2026-03-23 09:15:00 | 147.36 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2026-04-06 12:00:00 | 157.17 | 2026-04-17 14:15:00 | 172.89 | TARGET_HIT | 1.00 | 10.00% |
