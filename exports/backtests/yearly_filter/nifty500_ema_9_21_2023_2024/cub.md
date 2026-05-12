# City Union Bank Ltd. (CUB)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 258.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 200 |
| ALERT1 | 144 |
| ALERT2 | 142 |
| ALERT2_SKIP | 67 |
| ALERT3 | 394 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 163 |
| PARTIAL | 17 |
| TARGET_HIT | 2 |
| STOP_HIT | 167 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 186 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 63 / 123
- **Target hits / Stop hits / Partials:** 2 / 167 / 17
- **Avg / median % per leg:** 0.28% / -0.56%
- **Sum % (uncompounded):** 52.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 23 | 26.7% | 2 | 84 | 0 | -0.26% | -22.1% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 8 | 0 | -0.95% | -7.6% |
| BUY @ 3rd Alert (retest2) | 78 | 21 | 26.9% | 2 | 76 | 0 | -0.19% | -14.4% |
| SELL (all) | 100 | 40 | 40.0% | 0 | 83 | 17 | 0.75% | 75.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 100 | 40 | 40.0% | 0 | 83 | 17 | 0.75% | 75.1% |
| retest1 (combined) | 8 | 2 | 25.0% | 0 | 8 | 0 | -0.95% | -7.6% |
| retest2 (combined) | 178 | 61 | 34.3% | 2 | 159 | 17 | 0.34% | 60.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 12:15:00 | 139.70 | 138.73 | 138.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 13:15:00 | 140.25 | 139.04 | 138.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 10:15:00 | 139.95 | 140.05 | 139.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 10:30:00 | 140.00 | 140.05 | 139.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 11:15:00 | 139.65 | 139.97 | 139.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 11:30:00 | 139.50 | 139.97 | 139.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 12:15:00 | 139.55 | 139.89 | 139.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-16 13:45:00 | 139.95 | 139.85 | 139.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-16 14:15:00 | 138.75 | 139.63 | 139.42 | SL hit (close<static) qty=1.00 sl=139.20 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 15:15:00 | 138.85 | 139.85 | 139.94 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 14:15:00 | 140.70 | 139.94 | 139.92 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 09:15:00 | 138.50 | 139.69 | 139.82 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 140.65 | 140.00 | 139.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 15:15:00 | 140.90 | 140.18 | 140.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 15:15:00 | 140.85 | 140.96 | 140.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 09:15:00 | 140.55 | 140.96 | 140.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 141.00 | 140.97 | 140.59 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 14:15:00 | 140.00 | 140.45 | 140.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 139.75 | 140.31 | 140.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 09:15:00 | 138.70 | 138.41 | 139.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 12:15:00 | 139.50 | 138.63 | 139.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 12:15:00 | 139.50 | 138.63 | 139.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 13:00:00 | 139.50 | 138.63 | 139.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 13:15:00 | 139.70 | 138.85 | 139.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 13:45:00 | 139.50 | 138.85 | 139.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 139.35 | 138.95 | 139.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 14:30:00 | 139.70 | 138.95 | 139.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 124.95 | 124.04 | 124.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:30:00 | 125.70 | 124.04 | 124.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 124.20 | 124.07 | 124.63 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 15:15:00 | 125.30 | 124.95 | 124.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 125.75 | 125.11 | 125.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 10:15:00 | 124.95 | 125.08 | 124.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 10:15:00 | 124.95 | 125.08 | 124.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 124.95 | 125.08 | 124.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:00:00 | 124.95 | 125.08 | 124.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 124.75 | 125.01 | 124.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 12:45:00 | 125.10 | 125.01 | 124.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 13:30:00 | 125.10 | 124.99 | 124.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 15:00:00 | 125.20 | 125.03 | 124.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 12:45:00 | 125.65 | 125.33 | 125.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 126.80 | 125.82 | 125.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-09 11:15:00 | 124.90 | 125.50 | 125.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 11:15:00 | 124.90 | 125.50 | 125.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 124.75 | 125.35 | 125.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 124.65 | 124.59 | 124.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 124.65 | 124.59 | 124.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 124.65 | 124.59 | 124.84 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 11:15:00 | 125.10 | 124.80 | 124.80 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 12:15:00 | 124.55 | 124.80 | 124.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 13:15:00 | 124.45 | 124.73 | 124.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 125.25 | 124.65 | 124.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 125.25 | 124.65 | 124.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 125.25 | 124.65 | 124.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:00:00 | 125.25 | 124.65 | 124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 125.40 | 124.80 | 124.78 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 10:15:00 | 124.30 | 124.77 | 124.81 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 124.85 | 124.57 | 124.54 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 123.70 | 124.54 | 124.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 122.10 | 123.85 | 124.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 121.90 | 121.58 | 122.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 11:30:00 | 121.80 | 121.58 | 122.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 122.15 | 121.83 | 122.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 122.15 | 121.83 | 122.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 122.00 | 121.87 | 122.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 123.15 | 121.87 | 122.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 123.00 | 122.09 | 122.41 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 123.40 | 122.56 | 122.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 124.55 | 122.96 | 122.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 123.85 | 123.99 | 123.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 13:45:00 | 123.85 | 123.99 | 123.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 123.55 | 123.90 | 123.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 123.55 | 123.90 | 123.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 123.60 | 123.84 | 123.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:15:00 | 123.50 | 123.84 | 123.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 123.45 | 123.76 | 123.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 13:30:00 | 126.35 | 124.30 | 123.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 11:15:00 | 127.65 | 129.81 | 129.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 127.65 | 129.81 | 129.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 126.75 | 128.50 | 128.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 127.80 | 127.54 | 128.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 14:15:00 | 127.80 | 127.54 | 128.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 127.80 | 127.54 | 128.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 127.80 | 127.54 | 128.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 129.05 | 127.89 | 128.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:00:00 | 129.05 | 127.89 | 128.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 128.90 | 128.09 | 128.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:30:00 | 128.80 | 128.09 | 128.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 130.40 | 128.55 | 128.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 133.35 | 130.66 | 130.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 14:15:00 | 131.55 | 131.68 | 130.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 15:00:00 | 131.55 | 131.68 | 130.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 131.85 | 131.62 | 131.07 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 130.30 | 130.90 | 130.90 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 131.40 | 131.00 | 130.95 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 130.45 | 130.89 | 130.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 11:15:00 | 130.35 | 130.78 | 130.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 131.45 | 130.44 | 130.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 131.45 | 130.44 | 130.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 131.45 | 130.44 | 130.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:00:00 | 131.45 | 130.44 | 130.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 131.10 | 130.57 | 130.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:30:00 | 131.25 | 130.57 | 130.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 132.00 | 130.86 | 130.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 12:15:00 | 132.30 | 131.15 | 130.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 13:15:00 | 134.00 | 134.04 | 133.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-28 13:30:00 | 133.60 | 134.04 | 133.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 133.75 | 133.94 | 133.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:30:00 | 133.40 | 133.94 | 133.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 133.45 | 133.84 | 133.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:45:00 | 133.60 | 133.84 | 133.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 133.70 | 133.81 | 133.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 14:15:00 | 133.80 | 133.74 | 133.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 09:15:00 | 134.90 | 133.60 | 133.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 10:15:00 | 131.95 | 133.69 | 133.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 131.95 | 133.69 | 133.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 12:15:00 | 131.80 | 133.06 | 133.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 15:15:00 | 131.40 | 131.04 | 131.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 09:15:00 | 132.25 | 131.04 | 131.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 131.95 | 131.22 | 131.80 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 14:15:00 | 132.70 | 132.19 | 132.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 12:15:00 | 133.15 | 132.59 | 132.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 12:15:00 | 133.00 | 133.39 | 132.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 12:15:00 | 133.00 | 133.39 | 132.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 133.00 | 133.39 | 132.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 12:45:00 | 132.90 | 133.39 | 132.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 133.50 | 133.41 | 133.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 14:15:00 | 133.80 | 133.41 | 133.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 09:15:00 | 133.90 | 133.35 | 133.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 10:00:00 | 134.50 | 133.58 | 133.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 11:15:00 | 133.85 | 133.57 | 133.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 133.55 | 133.54 | 133.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 13:15:00 | 133.10 | 133.54 | 133.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 132.80 | 133.39 | 133.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-10 13:15:00 | 132.80 | 133.39 | 133.30 | SL hit (close<static) qty=1.00 sl=132.90 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 15:15:00 | 132.90 | 133.23 | 133.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 131.45 | 132.88 | 133.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 123.25 | 122.66 | 124.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 09:45:00 | 123.65 | 122.66 | 124.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 121.90 | 121.78 | 122.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 13:45:00 | 122.20 | 121.78 | 122.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 122.35 | 121.90 | 122.34 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 123.10 | 122.56 | 122.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 12:15:00 | 123.50 | 122.75 | 122.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 14:15:00 | 125.00 | 125.45 | 124.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 14:15:00 | 125.00 | 125.45 | 124.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 125.00 | 125.45 | 124.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 14:45:00 | 125.05 | 125.45 | 124.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 124.80 | 125.32 | 124.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 124.65 | 125.32 | 124.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 124.25 | 125.11 | 124.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 123.60 | 125.11 | 124.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 123.25 | 124.73 | 124.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 11:00:00 | 123.25 | 124.73 | 124.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 123.95 | 124.34 | 124.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 123.75 | 124.17 | 124.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 124.40 | 124.13 | 124.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 13:15:00 | 124.40 | 124.13 | 124.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 124.40 | 124.13 | 124.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:00:00 | 124.40 | 124.13 | 124.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 124.70 | 124.24 | 124.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 15:00:00 | 124.70 | 124.24 | 124.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 15:15:00 | 124.45 | 124.28 | 124.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 124.95 | 124.42 | 124.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 124.90 | 125.06 | 124.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 124.90 | 125.06 | 124.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 124.90 | 125.06 | 124.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 124.90 | 125.06 | 124.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 124.75 | 125.00 | 124.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 124.55 | 125.00 | 124.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 124.50 | 124.90 | 124.81 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 10:15:00 | 124.05 | 124.73 | 124.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 123.65 | 124.41 | 124.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 124.70 | 124.14 | 124.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 10:15:00 | 124.70 | 124.14 | 124.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 124.70 | 124.14 | 124.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:00:00 | 124.70 | 124.14 | 124.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 124.65 | 124.25 | 124.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:15:00 | 124.95 | 124.25 | 124.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 12:15:00 | 125.75 | 124.55 | 124.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 127.20 | 125.49 | 124.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 12:15:00 | 128.90 | 128.93 | 128.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 12:45:00 | 128.65 | 128.93 | 128.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 131.75 | 134.44 | 133.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 131.90 | 134.44 | 133.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 131.85 | 133.92 | 133.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 132.10 | 133.92 | 133.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 130.55 | 132.75 | 132.80 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 133.20 | 132.13 | 132.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 09:15:00 | 134.20 | 132.86 | 132.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 12:15:00 | 133.05 | 133.13 | 132.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 13:00:00 | 133.05 | 133.13 | 132.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 132.70 | 133.05 | 132.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 133.15 | 133.05 | 132.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 132.60 | 132.96 | 132.79 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 131.30 | 132.43 | 132.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 130.95 | 131.97 | 132.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 10:15:00 | 130.40 | 130.16 | 130.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 11:00:00 | 130.40 | 130.16 | 130.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 130.70 | 130.29 | 130.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:45:00 | 130.60 | 130.29 | 130.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 129.35 | 129.42 | 129.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 129.35 | 129.42 | 129.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 129.10 | 128.79 | 129.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 13:00:00 | 129.10 | 128.79 | 129.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 13:15:00 | 128.60 | 128.75 | 129.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 13:30:00 | 128.85 | 128.75 | 129.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 15:15:00 | 129.25 | 128.88 | 129.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:15:00 | 129.65 | 128.88 | 129.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 129.30 | 128.97 | 129.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:15:00 | 128.60 | 128.94 | 129.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 14:15:00 | 127.95 | 126.25 | 126.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 127.95 | 126.25 | 126.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 15:15:00 | 129.60 | 126.92 | 126.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 126.65 | 126.86 | 126.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 126.65 | 126.86 | 126.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 126.65 | 126.86 | 126.40 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 125.85 | 126.20 | 126.21 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 129.20 | 126.70 | 126.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 10:15:00 | 131.50 | 127.66 | 126.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 12:15:00 | 139.20 | 139.30 | 137.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 12:45:00 | 139.15 | 139.30 | 137.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 137.45 | 138.86 | 137.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:45:00 | 137.25 | 138.86 | 137.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 137.00 | 138.49 | 137.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 138.15 | 138.49 | 137.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 10:00:00 | 138.10 | 138.41 | 137.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 12:15:00 | 138.40 | 139.17 | 139.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 12:15:00 | 138.40 | 139.17 | 139.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 14:15:00 | 138.20 | 138.91 | 139.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 133.20 | 132.84 | 134.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 09:45:00 | 133.75 | 132.84 | 134.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 131.60 | 130.34 | 131.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 131.60 | 130.34 | 131.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 133.75 | 131.02 | 131.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 133.75 | 131.02 | 131.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 134.45 | 131.70 | 131.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 134.45 | 131.70 | 131.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 133.00 | 132.19 | 132.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 134.50 | 132.75 | 132.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 10:15:00 | 132.70 | 132.76 | 132.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 10:15:00 | 132.70 | 132.76 | 132.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 132.70 | 132.76 | 132.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 15:15:00 | 134.00 | 132.71 | 132.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 11:15:00 | 137.30 | 138.36 | 138.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 11:15:00 | 137.30 | 138.36 | 138.39 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 12:15:00 | 139.70 | 138.49 | 138.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 141.20 | 139.24 | 138.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 11:15:00 | 142.00 | 142.52 | 141.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-10 11:45:00 | 142.00 | 142.52 | 141.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 142.05 | 142.35 | 141.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 11:15:00 | 142.60 | 142.35 | 141.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 12:30:00 | 142.40 | 142.32 | 141.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 148.70 | 149.17 | 149.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 09:15:00 | 148.70 | 149.17 | 149.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 12:15:00 | 147.85 | 148.83 | 149.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 12:15:00 | 147.40 | 147.23 | 147.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 13:00:00 | 147.40 | 147.23 | 147.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 145.75 | 146.96 | 147.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 09:15:00 | 145.15 | 146.70 | 147.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 12:15:00 | 147.90 | 146.78 | 147.24 | SL hit (close>static) qty=1.00 sl=147.70 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 147.85 | 147.44 | 147.41 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 146.45 | 147.21 | 147.31 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 150.10 | 147.58 | 147.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 151.70 | 149.26 | 148.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 09:15:00 | 152.50 | 152.64 | 151.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-06 09:30:00 | 152.50 | 152.64 | 151.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 155.50 | 153.85 | 152.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 11:00:00 | 156.35 | 154.35 | 153.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 157.45 | 155.21 | 154.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 157.75 | 161.99 | 162.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 157.75 | 161.99 | 162.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 155.40 | 160.67 | 161.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-26 09:15:00 | 149.00 | 148.38 | 151.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-26 09:45:00 | 149.65 | 148.38 | 151.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 148.15 | 148.28 | 149.21 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 12:15:00 | 150.05 | 149.25 | 149.21 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 148.60 | 149.12 | 149.16 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 09:15:00 | 152.05 | 149.63 | 149.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 10:15:00 | 153.40 | 151.65 | 150.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 10:15:00 | 153.05 | 153.07 | 152.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-04 11:00:00 | 153.05 | 153.07 | 152.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 153.20 | 153.66 | 153.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:45:00 | 153.10 | 153.66 | 153.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 154.05 | 153.74 | 153.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 13:00:00 | 154.30 | 153.85 | 153.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 13:15:00 | 152.50 | 153.58 | 153.16 | SL hit (close<static) qty=1.00 sl=152.85 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 151.65 | 152.97 | 153.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 150.40 | 152.45 | 152.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 147.30 | 146.63 | 148.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 11:15:00 | 147.90 | 147.04 | 148.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 147.90 | 147.04 | 148.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:45:00 | 148.00 | 147.04 | 148.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 148.35 | 147.30 | 148.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 12:45:00 | 148.80 | 147.30 | 148.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 147.50 | 147.34 | 148.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 13:30:00 | 148.55 | 147.34 | 148.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 147.80 | 147.43 | 148.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:30:00 | 148.25 | 147.43 | 148.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 147.90 | 147.53 | 148.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 09:15:00 | 148.50 | 147.53 | 148.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 148.35 | 147.69 | 148.04 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 11:15:00 | 149.60 | 148.45 | 148.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 14:15:00 | 151.80 | 150.02 | 149.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 150.65 | 150.96 | 150.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 11:45:00 | 151.30 | 150.96 | 150.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 149.55 | 150.67 | 150.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 149.55 | 150.67 | 150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 150.00 | 150.54 | 150.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 149.55 | 150.54 | 150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 150.15 | 150.46 | 150.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:30:00 | 150.05 | 150.46 | 150.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 150.45 | 150.46 | 150.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 147.50 | 150.46 | 150.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 148.65 | 150.10 | 149.95 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 147.90 | 149.66 | 149.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 145.85 | 148.90 | 149.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 145.40 | 144.62 | 145.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:30:00 | 145.30 | 144.62 | 145.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 144.65 | 144.63 | 145.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:30:00 | 145.20 | 144.63 | 145.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 144.00 | 143.95 | 144.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 09:30:00 | 143.35 | 143.77 | 144.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 10:15:00 | 142.80 | 141.53 | 141.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 142.80 | 141.53 | 141.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 13:15:00 | 143.60 | 142.36 | 141.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 142.85 | 143.66 | 142.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 142.85 | 143.66 | 142.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 142.85 | 143.66 | 142.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 142.85 | 143.66 | 142.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 142.75 | 143.48 | 142.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:15:00 | 143.80 | 143.48 | 142.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 12:00:00 | 143.20 | 143.89 | 143.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:45:00 | 143.10 | 143.76 | 143.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 09:15:00 | 141.50 | 143.36 | 143.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 09:15:00 | 141.50 | 143.36 | 143.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 10:15:00 | 139.90 | 142.67 | 143.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 135.25 | 132.72 | 134.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 135.25 | 132.72 | 134.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 135.25 | 132.72 | 134.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 135.25 | 132.72 | 134.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 133.75 | 132.93 | 134.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 12:15:00 | 133.05 | 133.05 | 134.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 14:00:00 | 133.00 | 133.13 | 133.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 09:30:00 | 131.90 | 132.84 | 133.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 09:30:00 | 132.45 | 132.27 | 132.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 133.35 | 131.78 | 132.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:00:00 | 133.35 | 131.78 | 132.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 133.95 | 132.22 | 132.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:45:00 | 134.25 | 132.22 | 132.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-13 12:15:00 | 133.95 | 132.68 | 132.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 12:15:00 | 133.95 | 132.68 | 132.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 13:15:00 | 134.40 | 133.02 | 132.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 138.55 | 139.09 | 137.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-19 09:45:00 | 138.60 | 139.09 | 137.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 139.00 | 139.07 | 137.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:30:00 | 138.10 | 139.07 | 137.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 137.90 | 138.75 | 137.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 13:00:00 | 137.90 | 138.75 | 137.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 13:15:00 | 138.40 | 138.68 | 137.96 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 136.25 | 137.55 | 137.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 12:15:00 | 136.05 | 137.25 | 137.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 135.05 | 134.79 | 135.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 11:45:00 | 134.95 | 134.79 | 135.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 135.35 | 134.93 | 135.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 136.15 | 134.93 | 135.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 136.10 | 135.17 | 135.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:30:00 | 135.50 | 135.29 | 135.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 13:15:00 | 135.50 | 135.49 | 135.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 14:00:00 | 135.50 | 135.49 | 135.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 15:00:00 | 135.35 | 135.46 | 135.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 135.00 | 135.26 | 135.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 135.60 | 135.38 | 135.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 13:15:00 | 135.60 | 135.38 | 135.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 09:15:00 | 137.55 | 136.00 | 135.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 135.00 | 135.97 | 135.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 12:15:00 | 135.00 | 135.97 | 135.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 135.00 | 135.97 | 135.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 135.00 | 135.97 | 135.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 136.15 | 136.00 | 135.79 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 133.90 | 135.58 | 135.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 133.00 | 134.81 | 135.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 135.20 | 134.21 | 134.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 135.20 | 134.21 | 134.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 135.20 | 134.21 | 134.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 135.20 | 134.21 | 134.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 135.50 | 134.47 | 134.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 136.00 | 134.47 | 134.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 136.30 | 135.08 | 135.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 136.80 | 135.42 | 135.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 138.45 | 138.81 | 137.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 10:00:00 | 138.45 | 138.81 | 137.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 138.50 | 138.75 | 137.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:15:00 | 138.00 | 138.75 | 137.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 137.70 | 138.54 | 137.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:15:00 | 137.25 | 138.54 | 137.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 138.35 | 138.50 | 137.92 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 136.50 | 137.54 | 137.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 135.65 | 137.17 | 137.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 12:15:00 | 136.90 | 136.85 | 137.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 12:45:00 | 136.95 | 136.85 | 137.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 136.90 | 136.79 | 137.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:30:00 | 137.00 | 136.79 | 137.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 135.00 | 136.40 | 136.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 10:30:00 | 134.50 | 135.99 | 136.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 127.77 | 130.56 | 132.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-14 10:15:00 | 129.95 | 128.87 | 130.58 | SL hit (close>ema200) qty=0.50 sl=128.87 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 15:15:00 | 129.10 | 128.33 | 128.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 132.30 | 129.12 | 128.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 134.90 | 135.13 | 133.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 11:45:00 | 136.50 | 135.46 | 133.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 12:45:00 | 136.45 | 135.61 | 134.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 14:15:00 | 136.70 | 135.76 | 134.29 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 09:15:00 | 137.15 | 135.54 | 135.02 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 134.90 | 135.81 | 135.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 134.90 | 135.81 | 135.45 | SL hit (close<ema400) qty=1.00 sl=135.45 alert=retest1 |

### Cycle 60 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 153.35 | 155.97 | 156.08 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 155.55 | 154.21 | 154.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 156.85 | 155.64 | 154.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 155.05 | 155.78 | 155.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 10:15:00 | 155.05 | 155.78 | 155.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 155.05 | 155.78 | 155.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:00:00 | 155.05 | 155.78 | 155.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 154.95 | 155.61 | 155.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:30:00 | 154.60 | 155.61 | 155.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 156.30 | 155.75 | 155.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 13:15:00 | 156.35 | 155.75 | 155.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 10:00:00 | 156.70 | 156.00 | 155.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 13:15:00 | 156.50 | 156.22 | 155.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:15:00 | 156.55 | 155.90 | 155.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 156.65 | 156.05 | 155.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:00:00 | 157.60 | 156.59 | 156.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:45:00 | 157.20 | 156.72 | 156.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 15:15:00 | 159.65 | 160.81 | 160.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 15:15:00 | 159.65 | 160.81 | 160.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 157.50 | 160.15 | 160.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 10:15:00 | 149.55 | 149.42 | 151.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 10:30:00 | 149.95 | 149.42 | 151.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 147.70 | 148.93 | 150.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 146.95 | 148.66 | 150.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 153.45 | 150.48 | 150.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 153.45 | 150.48 | 150.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 154.55 | 151.29 | 150.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 152.95 | 153.33 | 152.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 12:00:00 | 152.95 | 153.33 | 152.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 153.05 | 153.24 | 152.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:30:00 | 152.45 | 153.24 | 152.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 153.15 | 153.19 | 152.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:30:00 | 153.00 | 153.19 | 152.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 151.80 | 152.88 | 152.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 152.40 | 152.88 | 152.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 153.55 | 153.02 | 152.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 154.25 | 153.02 | 152.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:30:00 | 154.10 | 153.44 | 153.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 154.30 | 153.44 | 153.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:15:00 | 153.75 | 153.54 | 153.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 151.80 | 153.22 | 153.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 151.80 | 153.22 | 153.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 152.00 | 152.97 | 152.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 15:15:00 | 152.00 | 152.97 | 152.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 145.90 | 151.54 | 152.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 15:15:00 | 144.10 | 144.04 | 145.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:15:00 | 144.70 | 144.04 | 145.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 145.60 | 144.55 | 145.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:30:00 | 145.70 | 144.55 | 145.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 145.10 | 144.66 | 145.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 144.55 | 144.66 | 145.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:30:00 | 144.55 | 144.66 | 145.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 15:15:00 | 144.70 | 144.66 | 145.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:45:00 | 144.30 | 144.70 | 145.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 144.30 | 144.62 | 145.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:00:00 | 144.00 | 144.62 | 144.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 144.05 | 144.60 | 144.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 143.90 | 144.44 | 144.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 143.35 | 144.23 | 144.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 142.55 | 142.29 | 143.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 142.55 | 142.29 | 143.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 142.05 | 142.24 | 142.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:30:00 | 142.75 | 142.24 | 142.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 142.80 | 142.44 | 142.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 146.10 | 142.44 | 142.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 145.00 | 142.95 | 143.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 144.40 | 143.24 | 143.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 144.40 | 143.24 | 143.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 146.50 | 144.15 | 143.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 142.30 | 145.40 | 144.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 142.30 | 145.40 | 144.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 142.30 | 145.40 | 144.80 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 137.00 | 143.72 | 144.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 133.10 | 141.60 | 143.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 140.45 | 138.99 | 140.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:15:00 | 141.35 | 138.99 | 140.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 143.45 | 139.88 | 141.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 143.45 | 139.88 | 141.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 143.40 | 140.58 | 141.38 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 144.40 | 141.87 | 141.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 146.45 | 143.48 | 142.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 149.43 | 149.67 | 148.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:45:00 | 149.38 | 149.67 | 148.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 149.07 | 149.57 | 149.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 149.07 | 149.57 | 149.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 149.12 | 149.48 | 149.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:30:00 | 150.39 | 149.41 | 149.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-20 09:15:00 | 165.43 | 158.69 | 156.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 166.25 | 166.93 | 167.02 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 168.40 | 167.19 | 167.09 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 166.42 | 166.95 | 167.00 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 169.57 | 167.38 | 167.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 170.66 | 169.50 | 168.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 170.45 | 170.96 | 169.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 170.45 | 170.96 | 169.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 170.45 | 170.96 | 169.82 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 167.30 | 169.29 | 169.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 166.53 | 168.74 | 169.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 166.94 | 166.61 | 167.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:00:00 | 166.94 | 166.61 | 167.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 167.75 | 166.84 | 167.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 167.65 | 166.84 | 167.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 166.90 | 166.85 | 167.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 12:15:00 | 166.72 | 166.85 | 167.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:15:00 | 158.38 | 160.61 | 161.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 160.62 | 160.61 | 161.36 | SL hit (close>static) qty=0.50 sl=160.61 alert=retest2 |

### Cycle 73 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 158.60 | 157.49 | 157.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 160.74 | 158.26 | 157.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 158.80 | 159.10 | 158.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 158.80 | 159.10 | 158.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 158.80 | 159.10 | 158.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 158.80 | 159.10 | 158.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 158.51 | 159.00 | 158.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 158.60 | 159.00 | 158.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 159.14 | 159.03 | 158.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 158.30 | 159.03 | 158.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 158.60 | 158.94 | 158.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 158.60 | 158.94 | 158.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 158.80 | 158.91 | 158.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:45:00 | 158.88 | 158.91 | 158.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 159.75 | 159.08 | 158.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 156.06 | 159.08 | 158.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 157.30 | 158.72 | 158.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 170.50 | 159.25 | 158.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 168.43 | 170.85 | 170.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 168.43 | 170.85 | 170.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 167.82 | 170.24 | 170.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 165.42 | 164.26 | 165.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 11:15:00 | 165.42 | 164.26 | 165.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 165.42 | 164.26 | 165.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 165.42 | 164.26 | 165.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 164.80 | 164.37 | 165.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 163.98 | 164.15 | 165.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 166.12 | 164.39 | 164.51 | SL hit (close>static) qty=1.00 sl=165.77 alert=retest2 |

### Cycle 75 — BUY (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 13:15:00 | 165.69 | 164.37 | 164.25 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 163.52 | 164.21 | 164.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 162.19 | 163.81 | 164.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 162.67 | 161.57 | 162.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 162.67 | 161.57 | 162.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 162.67 | 161.57 | 162.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:15:00 | 163.13 | 161.57 | 162.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 163.63 | 161.98 | 162.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 163.63 | 161.98 | 162.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 163.50 | 162.29 | 162.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 163.50 | 162.29 | 162.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 164.31 | 162.90 | 162.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 166.95 | 164.12 | 163.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 168.66 | 168.69 | 167.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:00:00 | 168.66 | 168.69 | 167.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 168.97 | 169.85 | 168.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 168.97 | 169.85 | 168.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 169.20 | 169.72 | 168.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 168.53 | 169.72 | 168.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 167.53 | 169.28 | 168.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 167.51 | 169.28 | 168.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 167.63 | 168.95 | 168.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 167.63 | 168.95 | 168.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 167.81 | 168.59 | 168.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 167.21 | 168.31 | 168.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 167.93 | 167.62 | 168.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 167.93 | 167.62 | 168.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 167.93 | 167.62 | 168.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 168.06 | 167.62 | 168.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 168.02 | 167.56 | 167.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 167.86 | 167.56 | 167.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 167.84 | 167.61 | 167.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 167.38 | 167.61 | 167.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 169.38 | 167.97 | 167.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 169.38 | 167.97 | 167.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 168.69 | 168.11 | 168.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 13:15:00 | 170.20 | 168.79 | 168.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 15:15:00 | 169.50 | 169.90 | 169.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 15:15:00 | 169.50 | 169.90 | 169.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 169.50 | 169.90 | 169.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 168.90 | 169.90 | 169.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 169.10 | 169.74 | 169.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:45:00 | 169.06 | 169.74 | 169.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 168.66 | 169.52 | 169.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 168.66 | 169.52 | 169.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 167.24 | 168.85 | 169.04 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 173.42 | 169.65 | 169.34 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 169.38 | 170.11 | 170.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 15:15:00 | 167.91 | 169.36 | 169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 170.50 | 169.59 | 169.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 170.50 | 169.59 | 169.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 170.50 | 169.59 | 169.75 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 173.95 | 170.49 | 170.13 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 166.59 | 169.72 | 170.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 164.86 | 166.85 | 167.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 13:15:00 | 167.10 | 165.76 | 166.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 13:15:00 | 167.10 | 165.76 | 166.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 167.10 | 165.76 | 166.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 167.10 | 165.76 | 166.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 165.54 | 165.72 | 166.31 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 171.45 | 167.42 | 166.97 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 168.19 | 169.29 | 169.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 166.75 | 168.56 | 169.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 167.72 | 166.90 | 167.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 167.72 | 166.90 | 167.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 167.45 | 167.01 | 167.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 166.96 | 167.01 | 167.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 166.19 | 166.84 | 167.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:15:00 | 165.80 | 166.84 | 167.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 170.45 | 168.19 | 167.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 170.45 | 168.19 | 167.96 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 167.48 | 168.36 | 168.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 166.75 | 168.04 | 168.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 10:15:00 | 168.00 | 167.76 | 168.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 10:15:00 | 168.00 | 167.76 | 168.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 168.00 | 167.76 | 168.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 168.00 | 167.76 | 168.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 166.57 | 167.52 | 167.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:30:00 | 165.74 | 166.94 | 167.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:00:00 | 166.12 | 166.73 | 167.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:15:00 | 166.14 | 166.38 | 166.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 165.75 | 166.23 | 166.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 164.73 | 164.75 | 165.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 164.62 | 164.75 | 165.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 164.85 | 164.34 | 164.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:45:00 | 165.16 | 164.34 | 164.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 164.28 | 164.32 | 164.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 162.80 | 164.26 | 164.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 157.81 | 160.30 | 161.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 157.83 | 160.30 | 161.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 157.45 | 159.36 | 160.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 157.46 | 159.36 | 160.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 154.66 | 158.50 | 160.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 156.17 | 155.52 | 157.78 | SL hit (close>ema200) qty=0.50 sl=155.52 alert=retest2 |

### Cycle 89 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 155.07 | 154.21 | 154.19 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 153.50 | 154.37 | 154.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 153.29 | 154.15 | 154.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 154.21 | 153.66 | 153.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 154.21 | 153.66 | 153.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 154.21 | 153.66 | 153.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 154.21 | 153.66 | 153.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 153.67 | 153.66 | 153.87 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 154.42 | 154.02 | 154.01 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 153.16 | 153.89 | 153.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 151.80 | 153.43 | 153.73 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 09:15:00 | 168.11 | 155.15 | 154.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-22 10:15:00 | 172.17 | 158.55 | 155.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 10:15:00 | 165.81 | 165.91 | 161.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:45:00 | 166.19 | 165.91 | 161.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 174.22 | 175.81 | 174.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:30:00 | 173.98 | 175.81 | 174.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 174.86 | 175.62 | 174.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 173.75 | 175.62 | 174.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 175.54 | 175.60 | 174.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 178.68 | 175.94 | 175.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 176.29 | 176.14 | 175.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 176.40 | 175.98 | 175.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 176.00 | 178.80 | 179.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 176.00 | 178.80 | 179.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 175.43 | 177.47 | 178.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 172.00 | 171.75 | 173.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:30:00 | 171.45 | 171.75 | 173.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 172.24 | 171.97 | 172.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:30:00 | 172.62 | 171.97 | 172.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 173.05 | 172.19 | 172.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 170.25 | 172.19 | 172.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 171.00 | 171.78 | 172.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 174.27 | 172.24 | 172.44 | SL hit (close>static) qty=1.00 sl=173.05 alert=retest2 |

### Cycle 95 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 172.51 | 171.43 | 171.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 173.70 | 172.21 | 171.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 15:15:00 | 173.11 | 173.15 | 172.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:15:00 | 175.36 | 173.15 | 172.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 178.77 | 179.28 | 178.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 178.36 | 179.28 | 178.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 178.52 | 179.13 | 178.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 178.52 | 179.13 | 178.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 179.64 | 179.23 | 178.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 177.07 | 178.84 | 178.57 | SL hit (close<ema400) qty=1.00 sl=178.57 alert=retest1 |

### Cycle 96 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 184.80 | 185.89 | 185.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 183.21 | 185.15 | 185.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 183.35 | 182.45 | 183.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 183.35 | 182.45 | 183.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 183.35 | 182.45 | 183.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 183.35 | 182.45 | 183.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 183.50 | 182.66 | 183.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 184.65 | 182.66 | 183.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 185.15 | 183.16 | 183.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 185.41 | 183.16 | 183.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 184.32 | 183.39 | 183.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 184.20 | 183.39 | 183.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 12:15:00 | 185.03 | 183.93 | 183.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 185.03 | 183.93 | 183.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 186.09 | 184.36 | 184.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 184.34 | 185.12 | 184.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 184.34 | 185.12 | 184.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 184.34 | 185.12 | 184.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 184.34 | 185.12 | 184.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 183.09 | 184.71 | 184.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 183.09 | 184.71 | 184.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 183.41 | 184.45 | 184.37 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 182.84 | 184.13 | 184.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 182.42 | 183.79 | 184.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 180.19 | 180.14 | 181.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 180.19 | 180.14 | 181.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 178.90 | 179.95 | 180.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 180.72 | 179.95 | 180.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 180.85 | 180.13 | 180.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 180.85 | 180.13 | 180.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 179.72 | 180.05 | 180.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 179.42 | 180.05 | 180.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 09:15:00 | 170.45 | 172.43 | 173.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 13:15:00 | 172.03 | 171.85 | 172.69 | SL hit (close>ema200) qty=0.50 sl=171.85 alert=retest2 |

### Cycle 99 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 173.97 | 172.52 | 172.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 175.03 | 173.30 | 172.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 174.92 | 175.37 | 174.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 13:15:00 | 174.92 | 175.37 | 174.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 174.92 | 175.37 | 174.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 174.92 | 175.37 | 174.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 172.14 | 174.56 | 174.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 171.63 | 174.56 | 174.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 169.95 | 173.64 | 173.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 169.38 | 172.26 | 173.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 170.62 | 170.62 | 171.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 170.62 | 170.62 | 171.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 170.62 | 170.62 | 171.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 172.05 | 170.62 | 171.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 169.72 | 170.31 | 171.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 168.89 | 170.31 | 171.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:00:00 | 168.01 | 169.44 | 170.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 14:45:00 | 168.63 | 169.27 | 170.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 168.90 | 169.30 | 170.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 169.03 | 168.27 | 169.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:45:00 | 169.50 | 168.27 | 169.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 168.38 | 168.29 | 169.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 167.14 | 168.29 | 169.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 168.05 | 166.07 | 165.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 168.05 | 166.07 | 165.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 169.00 | 166.65 | 166.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 171.61 | 171.70 | 170.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 171.61 | 171.70 | 170.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 172.73 | 174.03 | 173.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 172.73 | 174.03 | 173.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 172.64 | 173.75 | 173.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 171.91 | 173.75 | 173.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 171.65 | 173.33 | 173.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 172.09 | 173.33 | 173.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 170.14 | 172.30 | 172.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 10:15:00 | 169.34 | 171.15 | 171.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 10:15:00 | 170.01 | 169.69 | 170.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:00:00 | 170.01 | 169.69 | 170.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 163.11 | 164.81 | 166.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:30:00 | 162.47 | 164.32 | 166.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 167.41 | 165.15 | 166.11 | SL hit (close>static) qty=1.00 sl=166.95 alert=retest2 |

### Cycle 103 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 172.70 | 167.82 | 167.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 15:15:00 | 173.40 | 171.22 | 169.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 169.82 | 171.51 | 170.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 169.82 | 171.51 | 170.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 169.82 | 171.51 | 170.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 169.89 | 171.51 | 170.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 171.77 | 171.56 | 170.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 15:15:00 | 175.50 | 171.56 | 170.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 177.56 | 173.42 | 172.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 174.47 | 173.57 | 173.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:30:00 | 174.37 | 173.45 | 173.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 172.34 | 173.23 | 173.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:45:00 | 172.10 | 173.23 | 173.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-03 13:15:00 | 172.13 | 173.01 | 173.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 172.13 | 173.01 | 173.03 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 176.98 | 173.72 | 173.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 177.50 | 176.05 | 174.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 177.08 | 178.08 | 176.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 177.08 | 178.08 | 176.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 176.27 | 177.71 | 176.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 176.13 | 177.71 | 176.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 175.00 | 177.17 | 176.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 175.00 | 177.17 | 176.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 172.85 | 175.65 | 175.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 170.64 | 172.82 | 173.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 167.05 | 166.33 | 168.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 166.95 | 166.33 | 168.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 169.82 | 167.10 | 168.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 169.95 | 167.10 | 168.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 170.83 | 167.84 | 168.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 170.83 | 167.84 | 168.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 157.19 | 157.65 | 159.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:45:00 | 156.35 | 157.65 | 159.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 150.63 | 148.77 | 149.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:30:00 | 147.26 | 148.46 | 149.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:15:00 | 146.90 | 148.32 | 149.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:30:00 | 147.57 | 148.09 | 148.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 147.51 | 148.99 | 149.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 147.51 | 148.69 | 149.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:45:00 | 146.61 | 148.08 | 148.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 14:15:00 | 150.02 | 148.05 | 148.35 | SL hit (close>static) qty=1.00 sl=149.62 alert=retest2 |

### Cycle 107 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 150.65 | 148.59 | 148.54 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 147.40 | 148.52 | 148.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 14:15:00 | 147.00 | 147.98 | 148.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 149.56 | 148.14 | 148.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 149.56 | 148.14 | 148.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 149.56 | 148.14 | 148.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 150.16 | 148.14 | 148.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 150.15 | 148.54 | 148.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 152.73 | 149.90 | 149.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 155.24 | 155.49 | 153.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:45:00 | 154.93 | 155.49 | 153.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 154.81 | 155.46 | 154.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 154.81 | 155.46 | 154.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 155.55 | 155.48 | 154.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:30:00 | 156.14 | 155.50 | 154.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 12:15:00 | 155.96 | 155.50 | 154.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:30:00 | 155.96 | 155.57 | 154.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 152.33 | 154.76 | 154.49 | SL hit (close<static) qty=1.00 sl=153.62 alert=retest2 |

### Cycle 110 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 148.79 | 153.57 | 153.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 147.19 | 150.90 | 152.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 148.69 | 148.49 | 150.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 14:00:00 | 148.69 | 148.49 | 150.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 149.34 | 148.68 | 149.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:15:00 | 150.14 | 148.68 | 149.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 149.37 | 148.82 | 149.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 149.99 | 148.82 | 149.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 149.25 | 148.91 | 149.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:30:00 | 149.34 | 148.91 | 149.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 149.10 | 148.91 | 149.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 147.52 | 148.91 | 149.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 148.69 | 148.87 | 149.35 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 152.62 | 149.99 | 149.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 14:15:00 | 153.41 | 150.68 | 150.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 154.86 | 155.04 | 153.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 156.91 | 155.04 | 153.74 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 159.23 | 160.25 | 159.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 159.23 | 160.25 | 159.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 158.73 | 159.95 | 159.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-24 14:15:00 | 158.73 | 159.95 | 159.08 | SL hit (close<ema400) qty=1.00 sl=159.08 alert=retest1 |

### Cycle 112 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 158.34 | 158.76 | 158.78 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 11:15:00 | 160.06 | 159.02 | 158.89 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 157.82 | 158.86 | 158.88 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 159.52 | 158.99 | 158.94 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 154.30 | 158.41 | 158.74 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 160.07 | 158.11 | 158.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 161.28 | 159.40 | 158.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 162.70 | 162.92 | 161.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:15:00 | 161.88 | 162.92 | 161.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 162.55 | 162.85 | 161.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 163.07 | 162.85 | 161.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 162.95 | 162.88 | 162.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:45:00 | 162.44 | 162.88 | 162.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 162.70 | 162.84 | 162.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 156.88 | 162.84 | 162.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 151.84 | 160.64 | 161.14 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 161.55 | 159.62 | 159.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 165.10 | 161.85 | 160.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 181.27 | 182.10 | 179.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:30:00 | 181.39 | 182.10 | 179.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 179.39 | 181.56 | 179.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:00:00 | 179.39 | 181.56 | 179.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 177.32 | 180.71 | 179.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:00:00 | 177.32 | 180.71 | 179.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 177.62 | 180.09 | 179.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 179.28 | 178.69 | 178.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 178.07 | 178.56 | 178.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 09:15:00 | 178.07 | 178.56 | 178.59 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 14:15:00 | 182.48 | 179.24 | 178.84 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 175.37 | 178.60 | 178.67 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 178.65 | 178.00 | 177.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 180.17 | 178.44 | 178.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 178.73 | 179.85 | 179.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 178.73 | 179.85 | 179.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 178.73 | 179.85 | 179.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:15:00 | 177.71 | 179.85 | 179.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 177.29 | 179.34 | 179.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 177.29 | 179.34 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 177.02 | 178.87 | 178.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 175.59 | 177.93 | 178.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 177.72 | 174.94 | 175.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 177.72 | 174.94 | 175.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 177.72 | 174.94 | 175.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:30:00 | 176.53 | 175.69 | 176.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 12:15:00 | 180.64 | 177.32 | 176.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 180.64 | 177.32 | 176.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 182.40 | 178.33 | 177.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 181.09 | 181.21 | 179.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 14:45:00 | 180.52 | 181.21 | 179.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 180.00 | 180.85 | 179.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 180.49 | 180.78 | 179.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:00:00 | 180.62 | 180.75 | 180.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 176.25 | 180.14 | 180.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 176.25 | 180.14 | 180.50 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 183.17 | 180.31 | 179.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 184.01 | 181.05 | 180.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 192.31 | 192.92 | 189.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:00:00 | 192.31 | 192.92 | 189.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 193.28 | 193.03 | 191.96 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 189.83 | 191.53 | 191.57 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 192.90 | 191.61 | 191.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 12:15:00 | 193.64 | 192.01 | 191.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 191.33 | 191.88 | 191.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 13:15:00 | 191.33 | 191.88 | 191.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 191.33 | 191.88 | 191.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 191.33 | 191.88 | 191.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 193.33 | 192.17 | 191.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 194.00 | 192.17 | 191.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:15:00 | 194.00 | 195.34 | 194.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 194.21 | 195.12 | 194.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 10:15:00 | 193.69 | 195.20 | 195.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 193.69 | 195.20 | 195.38 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 196.02 | 194.53 | 194.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 197.63 | 195.71 | 195.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 200.62 | 200.97 | 199.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 200.62 | 200.97 | 199.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 200.62 | 200.97 | 199.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 198.55 | 200.97 | 199.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 200.18 | 200.64 | 199.32 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 198.20 | 198.80 | 198.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 197.22 | 198.40 | 198.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 14:15:00 | 198.79 | 198.32 | 198.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 14:15:00 | 198.79 | 198.32 | 198.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 198.79 | 198.32 | 198.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 198.79 | 198.32 | 198.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 197.85 | 198.22 | 198.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 198.91 | 198.22 | 198.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 200.00 | 198.58 | 198.62 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 199.98 | 198.86 | 198.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 202.00 | 199.98 | 199.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 203.70 | 205.43 | 204.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 203.70 | 205.43 | 204.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 203.70 | 205.43 | 204.06 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 200.06 | 203.55 | 203.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 199.10 | 201.28 | 202.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 196.10 | 195.80 | 198.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 196.10 | 195.80 | 198.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 194.80 | 194.55 | 195.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 195.67 | 194.55 | 195.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 196.72 | 194.98 | 195.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 198.00 | 194.98 | 195.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 193.89 | 194.76 | 195.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 193.28 | 194.76 | 195.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 193.01 | 194.29 | 195.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 193.44 | 193.37 | 194.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:00:00 | 193.58 | 193.51 | 194.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 193.19 | 193.44 | 194.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 193.50 | 193.44 | 194.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 192.33 | 193.30 | 193.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 191.13 | 193.09 | 193.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 191.52 | 192.36 | 193.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:45:00 | 190.78 | 192.08 | 192.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 191.04 | 192.03 | 192.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 187.50 | 190.49 | 191.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 194.09 | 192.19 | 191.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 194.09 | 192.19 | 191.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 195.00 | 193.99 | 193.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 203.09 | 203.28 | 200.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 203.09 | 203.28 | 200.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 219.94 | 224.24 | 220.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 222.15 | 224.24 | 220.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 223.30 | 224.06 | 220.48 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 216.46 | 218.86 | 219.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 214.75 | 217.45 | 218.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 217.65 | 216.39 | 217.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 217.65 | 216.39 | 217.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 217.65 | 216.39 | 217.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 217.65 | 216.39 | 217.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 216.82 | 216.48 | 217.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 217.65 | 216.48 | 217.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 217.17 | 216.62 | 217.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 217.17 | 216.62 | 217.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 218.64 | 217.02 | 217.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 218.64 | 217.02 | 217.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 217.88 | 217.19 | 217.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 218.54 | 217.19 | 217.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 218.71 | 217.50 | 217.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 218.71 | 217.50 | 217.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 218.12 | 217.62 | 217.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 218.64 | 217.62 | 217.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 219.00 | 217.90 | 217.86 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 216.04 | 217.61 | 217.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 213.61 | 216.81 | 217.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 216.00 | 215.44 | 216.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:00:00 | 216.00 | 215.44 | 216.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 215.50 | 215.45 | 216.18 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 218.25 | 216.81 | 216.64 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 214.75 | 216.53 | 216.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 212.80 | 214.91 | 215.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 215.60 | 212.73 | 213.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 215.60 | 212.73 | 213.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 215.60 | 212.73 | 213.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 214.54 | 212.73 | 213.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 213.37 | 212.86 | 213.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 213.27 | 212.86 | 213.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 213.23 | 212.89 | 213.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 216.90 | 213.97 | 213.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 216.90 | 213.97 | 213.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 218.62 | 216.47 | 215.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 217.00 | 217.27 | 216.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 217.00 | 217.27 | 216.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 217.00 | 217.27 | 216.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 215.83 | 217.27 | 216.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 217.01 | 217.22 | 216.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:45:00 | 217.06 | 217.22 | 216.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 216.42 | 217.06 | 216.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 216.84 | 217.06 | 216.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 216.49 | 216.95 | 216.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:30:00 | 216.14 | 216.95 | 216.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 216.46 | 216.85 | 216.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 216.07 | 216.85 | 216.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 216.15 | 216.71 | 216.55 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 215.10 | 216.39 | 216.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 214.52 | 215.85 | 216.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 213.25 | 212.56 | 213.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 213.03 | 212.56 | 213.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 212.20 | 212.49 | 213.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 211.89 | 212.41 | 213.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:00:00 | 211.50 | 212.23 | 213.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 211.64 | 211.77 | 212.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 15:15:00 | 214.19 | 212.98 | 212.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 214.19 | 212.98 | 212.88 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 210.82 | 212.55 | 212.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 210.12 | 211.82 | 212.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 205.92 | 205.34 | 206.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 207.60 | 205.34 | 206.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 206.15 | 205.50 | 206.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 207.30 | 205.50 | 206.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 206.87 | 205.77 | 206.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 206.87 | 205.77 | 206.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 208.00 | 206.22 | 206.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:15:00 | 207.01 | 206.86 | 207.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 208.06 | 207.29 | 207.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 208.06 | 207.29 | 207.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 210.24 | 208.02 | 207.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 213.43 | 214.85 | 212.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 14:00:00 | 213.43 | 214.85 | 212.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 213.62 | 214.60 | 212.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 216.34 | 214.67 | 212.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 216.00 | 214.91 | 213.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:00:00 | 215.65 | 215.06 | 213.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 215.87 | 216.45 | 215.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 216.01 | 216.36 | 215.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 214.01 | 215.30 | 215.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 214.01 | 215.30 | 215.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 213.43 | 214.54 | 214.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 12:15:00 | 209.34 | 209.11 | 210.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:30:00 | 209.65 | 209.11 | 210.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 213.27 | 209.72 | 210.35 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 212.16 | 210.80 | 210.74 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 209.90 | 210.67 | 210.74 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 212.00 | 211.02 | 210.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 212.24 | 211.26 | 211.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 211.95 | 212.78 | 212.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 211.95 | 212.78 | 212.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 211.95 | 212.78 | 212.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 211.95 | 212.78 | 212.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 211.60 | 212.55 | 212.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:45:00 | 212.72 | 212.75 | 212.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 213.00 | 214.16 | 214.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 213.00 | 214.16 | 214.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 211.63 | 213.47 | 213.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 197.36 | 197.28 | 198.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:30:00 | 197.19 | 197.28 | 198.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 199.79 | 197.76 | 198.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 199.79 | 197.76 | 198.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 200.12 | 198.23 | 198.94 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 199.96 | 199.34 | 199.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 15:15:00 | 201.55 | 199.78 | 199.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 198.20 | 199.83 | 199.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 12:15:00 | 198.20 | 199.83 | 199.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 198.20 | 199.83 | 199.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 198.20 | 199.83 | 199.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 199.45 | 199.75 | 199.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 196.46 | 199.75 | 199.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 197.90 | 199.38 | 199.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 196.68 | 198.64 | 199.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 198.00 | 196.52 | 197.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 198.00 | 196.52 | 197.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 198.00 | 196.52 | 197.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 197.81 | 196.52 | 197.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 198.00 | 196.82 | 197.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:45:00 | 197.34 | 196.62 | 197.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 15:15:00 | 200.03 | 198.01 | 197.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 200.03 | 198.01 | 197.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 202.59 | 198.92 | 198.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 11:15:00 | 198.80 | 199.09 | 198.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 198.80 | 199.09 | 198.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 200.21 | 199.31 | 198.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 199.86 | 199.31 | 198.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 199.25 | 199.30 | 198.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 199.25 | 199.30 | 198.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 196.66 | 198.77 | 198.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:45:00 | 196.41 | 198.77 | 198.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 197.26 | 198.47 | 198.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 199.74 | 198.47 | 198.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 203.26 | 204.37 | 204.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 11:15:00 | 203.26 | 204.37 | 204.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 203.01 | 204.01 | 204.25 | Break + close below crossover candle low |

### Cycle 155 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 207.70 | 204.65 | 204.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 211.67 | 208.06 | 206.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 208.70 | 209.66 | 208.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 10:00:00 | 208.70 | 209.66 | 208.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 208.03 | 209.33 | 208.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 207.54 | 209.33 | 208.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 207.84 | 209.03 | 208.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:45:00 | 207.43 | 209.03 | 208.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 207.74 | 208.77 | 207.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 207.74 | 208.77 | 207.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 207.13 | 208.44 | 207.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 207.13 | 208.44 | 207.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 206.00 | 207.96 | 207.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 206.00 | 207.96 | 207.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 206.41 | 207.65 | 207.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 205.70 | 207.65 | 207.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 207.04 | 207.55 | 207.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 205.28 | 207.10 | 207.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 210.64 | 206.97 | 207.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 210.64 | 206.97 | 207.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 210.64 | 206.97 | 207.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 210.64 | 206.97 | 207.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 211.22 | 207.82 | 207.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 12:15:00 | 212.71 | 211.08 | 209.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 208.00 | 210.97 | 210.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 208.00 | 210.97 | 210.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 208.00 | 210.97 | 210.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 207.03 | 210.97 | 210.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 209.57 | 210.69 | 210.11 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 207.90 | 209.58 | 209.67 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 210.14 | 209.27 | 209.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 213.10 | 210.03 | 209.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 211.78 | 212.36 | 211.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 211.78 | 212.36 | 211.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 211.78 | 212.36 | 211.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 211.78 | 212.36 | 211.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 211.15 | 212.12 | 211.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 211.50 | 212.12 | 211.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 209.65 | 211.63 | 211.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:45:00 | 209.80 | 211.63 | 211.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 214.04 | 212.11 | 211.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:30:00 | 215.51 | 213.16 | 212.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 216.00 | 217.24 | 217.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 216.00 | 217.24 | 217.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 214.53 | 216.70 | 217.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 214.70 | 213.93 | 215.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 214.15 | 213.93 | 215.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 215.04 | 214.15 | 215.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 215.04 | 214.15 | 215.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 215.11 | 214.34 | 215.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 215.11 | 214.34 | 215.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 215.39 | 214.55 | 215.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 215.39 | 214.55 | 215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 215.73 | 214.79 | 215.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 215.73 | 214.79 | 215.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 215.10 | 215.04 | 215.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 216.09 | 215.04 | 215.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 216.08 | 215.25 | 215.27 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 216.03 | 215.40 | 215.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 216.76 | 215.68 | 215.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 214.68 | 216.32 | 215.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 214.68 | 216.32 | 215.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 214.68 | 216.32 | 215.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 215.75 | 216.32 | 215.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 213.30 | 215.72 | 215.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 212.39 | 214.47 | 215.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 215.29 | 214.08 | 214.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 215.29 | 214.08 | 214.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 215.29 | 214.08 | 214.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 215.29 | 214.08 | 214.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 217.44 | 214.76 | 214.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 217.04 | 214.76 | 214.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 217.79 | 215.36 | 215.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 218.61 | 216.51 | 215.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 219.70 | 220.08 | 218.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 219.70 | 220.08 | 218.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 219.31 | 219.93 | 218.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 218.93 | 219.93 | 218.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 218.67 | 219.68 | 218.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 218.67 | 219.68 | 218.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 219.62 | 219.67 | 218.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 219.14 | 219.67 | 218.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 222.40 | 220.21 | 219.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 223.70 | 220.21 | 219.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 222.93 | 225.97 | 226.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 222.93 | 225.97 | 226.19 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 230.41 | 226.53 | 226.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 233.40 | 229.02 | 227.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 230.17 | 230.27 | 228.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 230.17 | 230.27 | 228.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 230.27 | 230.27 | 228.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 228.63 | 230.27 | 228.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 232.87 | 235.25 | 233.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 232.31 | 235.25 | 233.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 232.99 | 234.80 | 233.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:45:00 | 234.88 | 234.33 | 233.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 231.71 | 234.25 | 233.99 | SL hit (close<static) qty=1.00 sl=232.41 alert=retest2 |

### Cycle 166 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 230.50 | 233.50 | 233.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 230.11 | 232.60 | 233.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 235.42 | 231.99 | 232.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 235.42 | 231.99 | 232.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 235.42 | 231.99 | 232.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 235.42 | 231.99 | 232.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 233.92 | 232.38 | 232.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 235.83 | 232.38 | 232.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 235.70 | 233.43 | 233.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 236.30 | 234.01 | 233.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 13:15:00 | 252.96 | 254.25 | 248.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 14:00:00 | 252.96 | 254.25 | 248.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 250.80 | 253.66 | 249.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:15:00 | 252.40 | 253.66 | 249.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 250.95 | 254.91 | 254.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 14:15:00 | 250.95 | 254.91 | 254.95 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 13:15:00 | 257.12 | 255.04 | 254.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 14:15:00 | 259.71 | 255.98 | 255.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 277.40 | 277.45 | 273.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 273.84 | 277.45 | 273.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 268.99 | 275.76 | 273.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 268.99 | 275.76 | 273.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 268.00 | 274.21 | 272.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 268.24 | 274.21 | 272.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 271.64 | 272.70 | 272.36 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 270.20 | 271.93 | 272.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 266.31 | 270.80 | 271.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 271.40 | 265.57 | 267.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 271.40 | 265.57 | 267.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 271.40 | 265.57 | 267.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 271.40 | 265.57 | 267.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 271.00 | 266.66 | 267.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 271.00 | 266.66 | 267.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 272.00 | 268.42 | 268.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 10:15:00 | 275.19 | 271.37 | 269.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 273.84 | 276.22 | 274.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 273.84 | 276.22 | 274.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 273.84 | 276.22 | 274.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 273.25 | 276.22 | 274.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 274.00 | 275.78 | 274.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:45:00 | 275.00 | 274.78 | 274.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 274.86 | 274.78 | 274.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 273.30 | 274.73 | 274.44 | SL hit (close<static) qty=1.00 sl=273.82 alert=retest2 |

### Cycle 172 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 270.75 | 273.68 | 274.01 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 276.45 | 274.17 | 274.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 277.50 | 274.84 | 274.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 276.95 | 278.34 | 276.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 276.95 | 278.34 | 276.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 276.95 | 278.34 | 276.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 276.95 | 278.34 | 276.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 276.45 | 277.96 | 276.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 276.35 | 277.96 | 276.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 274.00 | 277.17 | 276.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 274.00 | 277.17 | 276.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 273.65 | 276.46 | 276.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 273.65 | 276.46 | 276.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 273.30 | 275.83 | 275.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 15:15:00 | 272.00 | 274.66 | 275.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 273.40 | 273.25 | 274.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 11:00:00 | 273.40 | 273.25 | 274.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 275.05 | 271.01 | 272.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 275.05 | 271.01 | 272.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 275.00 | 271.81 | 272.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 269.80 | 271.81 | 272.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 271.25 | 271.37 | 272.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 272.25 | 271.37 | 272.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 271.40 | 271.37 | 272.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 271.40 | 271.37 | 272.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 270.40 | 270.49 | 271.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 270.00 | 270.49 | 271.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 270.60 | 270.51 | 271.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 271.15 | 270.51 | 271.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 271.95 | 270.80 | 271.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 271.95 | 270.80 | 271.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 272.50 | 271.14 | 271.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 270.35 | 271.23 | 271.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 271.00 | 266.28 | 265.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 271.00 | 266.28 | 265.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 275.40 | 269.90 | 267.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 273.30 | 273.72 | 271.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 275.00 | 273.72 | 271.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 274.55 | 273.88 | 271.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 278.75 | 275.87 | 274.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 286.60 | 292.26 | 293.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 286.60 | 292.26 | 293.01 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 293.50 | 290.21 | 289.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 294.65 | 291.10 | 290.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 289.45 | 292.28 | 291.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 289.45 | 292.28 | 291.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 289.45 | 292.28 | 291.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 289.45 | 292.28 | 291.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 286.90 | 291.20 | 291.07 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 290.00 | 290.96 | 290.97 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 292.00 | 291.06 | 290.99 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 289.15 | 290.68 | 290.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 282.60 | 289.06 | 290.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 283.95 | 283.10 | 285.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:00:00 | 283.95 | 283.10 | 285.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 274.00 | 271.00 | 274.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 274.35 | 271.00 | 274.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 274.20 | 271.64 | 274.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 274.95 | 271.64 | 274.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 276.70 | 272.65 | 274.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 276.70 | 272.65 | 274.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 275.35 | 273.19 | 274.73 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 276.95 | 275.59 | 275.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 278.50 | 276.65 | 276.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 275.10 | 276.53 | 276.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 275.10 | 276.53 | 276.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 275.10 | 276.53 | 276.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 275.10 | 276.53 | 276.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 273.70 | 275.97 | 275.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 274.00 | 275.97 | 275.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 272.80 | 275.33 | 275.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 270.75 | 274.42 | 275.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 12:15:00 | 271.90 | 269.88 | 271.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 12:15:00 | 271.90 | 269.88 | 271.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 271.90 | 269.88 | 271.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 271.90 | 269.88 | 271.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 272.05 | 270.31 | 271.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:15:00 | 273.00 | 270.31 | 271.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 270.05 | 270.26 | 271.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:45:00 | 272.10 | 270.26 | 271.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 272.05 | 270.62 | 271.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 273.20 | 270.62 | 271.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 270.10 | 270.51 | 271.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 269.10 | 270.18 | 271.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 278.35 | 272.56 | 272.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 278.35 | 272.56 | 272.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 15:15:00 | 282.05 | 274.46 | 272.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 281.10 | 281.20 | 278.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 281.10 | 281.20 | 278.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 278.00 | 280.56 | 278.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:45:00 | 278.90 | 280.56 | 278.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 275.95 | 279.64 | 278.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 275.15 | 279.64 | 278.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 275.10 | 278.73 | 277.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 275.10 | 278.73 | 277.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 275.50 | 278.09 | 277.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 277.60 | 278.09 | 277.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 286.25 | 288.40 | 285.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 286.25 | 288.40 | 285.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 284.50 | 287.62 | 285.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 284.70 | 287.62 | 285.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 283.70 | 286.83 | 285.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 283.70 | 286.83 | 285.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 286.30 | 286.73 | 285.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 287.60 | 287.17 | 285.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:45:00 | 289.95 | 292.93 | 292.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 285.75 | 291.36 | 291.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 285.75 | 291.36 | 291.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 280.35 | 287.53 | 289.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 286.60 | 283.79 | 286.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 286.60 | 283.79 | 286.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 286.60 | 283.79 | 286.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 286.60 | 283.79 | 286.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 286.70 | 284.37 | 286.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 296.75 | 284.37 | 286.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 306.35 | 288.77 | 288.50 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 291.45 | 293.82 | 293.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 290.00 | 292.21 | 293.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 293.95 | 292.34 | 292.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 293.95 | 292.34 | 292.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 293.95 | 292.34 | 292.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 293.95 | 292.34 | 292.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 291.60 | 292.19 | 292.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 295.25 | 292.19 | 292.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 292.55 | 292.26 | 292.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 292.70 | 292.26 | 292.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 290.60 | 291.93 | 292.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 289.60 | 291.93 | 292.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 289.40 | 291.35 | 292.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:00:00 | 289.50 | 288.20 | 289.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:30:00 | 289.75 | 288.41 | 289.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 289.25 | 288.58 | 289.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 289.65 | 288.58 | 289.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 290.00 | 288.86 | 289.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 290.00 | 288.86 | 289.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 288.95 | 288.88 | 289.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 282.50 | 289.64 | 289.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 15:15:00 | 275.12 | 278.39 | 281.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 15:15:00 | 275.26 | 278.39 | 281.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:15:00 | 274.93 | 276.71 | 280.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:15:00 | 275.02 | 276.71 | 280.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:15:00 | 268.38 | 275.81 | 279.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 276.70 | 275.87 | 278.74 | SL hit (close>ema200) qty=0.50 sl=275.87 alert=retest2 |

### Cycle 187 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 286.05 | 280.69 | 280.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 287.85 | 285.42 | 283.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 284.85 | 286.11 | 284.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:30:00 | 285.35 | 286.11 | 284.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 283.50 | 285.59 | 284.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 282.15 | 285.59 | 284.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 279.50 | 284.37 | 283.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:00:00 | 279.50 | 284.37 | 283.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 277.70 | 283.03 | 283.20 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 289.10 | 283.11 | 282.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 13:15:00 | 292.30 | 286.64 | 284.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 286.70 | 287.93 | 285.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 286.70 | 287.93 | 285.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 286.70 | 287.93 | 285.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 285.00 | 287.93 | 285.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 286.65 | 287.73 | 286.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 286.65 | 287.73 | 286.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 289.20 | 287.97 | 286.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 286.80 | 287.97 | 286.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 287.70 | 288.08 | 286.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 290.30 | 288.08 | 286.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 290.05 | 289.96 | 288.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 290.60 | 289.81 | 288.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 290.80 | 289.37 | 288.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 287.50 | 289.00 | 288.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 285.20 | 289.00 | 288.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 288.55 | 288.91 | 288.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 287.75 | 288.91 | 288.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 285.25 | 288.18 | 288.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 285.25 | 288.18 | 288.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 284.40 | 287.42 | 288.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 267.25 | 266.95 | 272.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:45:00 | 267.80 | 266.95 | 272.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 254.95 | 250.97 | 254.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 254.95 | 250.97 | 254.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 251.05 | 250.99 | 254.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 249.90 | 251.30 | 253.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 12:15:00 | 249.10 | 251.30 | 253.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 237.41 | 242.57 | 246.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 236.64 | 242.57 | 246.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 234.35 | 234.24 | 238.65 | SL hit (close>ema200) qty=0.50 sl=234.24 alert=retest2 |

### Cycle 191 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 242.00 | 240.30 | 240.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 248.00 | 241.85 | 241.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 243.85 | 247.43 | 244.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 243.85 | 247.43 | 244.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 243.85 | 247.43 | 244.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 243.85 | 247.43 | 244.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 243.45 | 246.64 | 244.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 243.45 | 246.64 | 244.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 243.50 | 244.50 | 244.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 245.85 | 244.50 | 244.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 242.90 | 244.01 | 244.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 242.90 | 244.01 | 244.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 241.80 | 243.49 | 243.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 15:15:00 | 238.05 | 236.45 | 239.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:15:00 | 240.00 | 236.45 | 239.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 236.85 | 236.53 | 238.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 235.50 | 236.53 | 238.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 235.30 | 235.40 | 237.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 243.35 | 236.92 | 237.67 | SL hit (close>static) qty=1.00 sl=241.80 alert=retest2 |

### Cycle 193 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 245.30 | 238.60 | 238.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 247.80 | 240.44 | 239.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 245.35 | 247.01 | 244.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:00:00 | 245.35 | 247.01 | 244.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 238.80 | 246.30 | 244.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 238.80 | 246.30 | 244.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 239.90 | 245.02 | 244.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:30:00 | 237.90 | 245.02 | 244.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 239.60 | 243.94 | 244.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 09:15:00 | 237.43 | 240.70 | 242.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 239.98 | 239.93 | 241.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 239.98 | 239.93 | 241.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 242.53 | 240.45 | 241.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 242.53 | 240.45 | 241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 242.20 | 240.80 | 241.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 241.31 | 240.54 | 241.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 241.00 | 239.06 | 239.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 240.66 | 239.06 | 239.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 244.72 | 240.19 | 239.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 244.72 | 240.19 | 239.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 245.57 | 241.27 | 240.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 253.02 | 254.66 | 251.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:00:00 | 253.02 | 254.66 | 251.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 252.87 | 253.75 | 251.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 255.84 | 253.28 | 251.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 253.50 | 255.24 | 254.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 12:15:00 | 250.32 | 253.28 | 253.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 250.32 | 253.28 | 253.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 249.28 | 252.00 | 252.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 10:15:00 | 251.57 | 251.43 | 252.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 10:15:00 | 251.57 | 251.43 | 252.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 251.57 | 251.43 | 252.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:00:00 | 250.19 | 251.63 | 252.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 249.71 | 250.43 | 251.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 255.40 | 251.73 | 251.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 12:15:00 | 255.40 | 251.73 | 251.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 256.25 | 252.63 | 251.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 264.80 | 265.07 | 260.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 10:15:00 | 263.90 | 265.07 | 260.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 265.52 | 267.60 | 265.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 265.52 | 267.60 | 265.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 265.97 | 267.27 | 265.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 265.11 | 267.27 | 265.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 264.59 | 266.74 | 265.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:45:00 | 264.12 | 266.74 | 265.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 265.10 | 266.41 | 265.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:45:00 | 264.51 | 266.41 | 265.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 265.69 | 266.13 | 265.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 263.95 | 266.13 | 265.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 263.45 | 265.60 | 265.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 261.76 | 265.60 | 265.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 262.59 | 265.00 | 265.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 261.03 | 263.83 | 264.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 265.62 | 263.46 | 264.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 265.62 | 263.46 | 264.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 265.62 | 263.46 | 264.06 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 267.65 | 264.85 | 264.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 269.68 | 266.24 | 265.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 15:15:00 | 275.00 | 275.55 | 272.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 277.52 | 275.94 | 272.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:45:00 | 280.16 | 277.08 | 273.76 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 272.55 | 276.80 | 275.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 272.55 | 276.80 | 275.05 | SL hit (close<ema400) qty=1.00 sl=275.05 alert=retest1 |

### Cycle 200 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 271.50 | 274.04 | 274.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 270.46 | 273.32 | 273.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 274.75 | 273.13 | 273.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 274.75 | 273.13 | 273.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 274.75 | 273.13 | 273.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 274.75 | 273.13 | 273.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 274.30 | 273.36 | 273.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 271.75 | 273.09 | 273.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 272.25 | 270.96 | 271.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 15:15:00 | 258.16 | 263.14 | 266.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 15:15:00 | 258.64 | 263.14 | 266.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 09:15:00 | 136.85 | 2023-05-15 12:15:00 | 139.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2023-05-16 13:45:00 | 139.95 | 2023-05-16 14:15:00 | 138.75 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-05-17 10:15:00 | 139.95 | 2023-05-18 13:15:00 | 138.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-06-06 12:45:00 | 125.10 | 2023-06-09 11:15:00 | 124.90 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-06-06 13:30:00 | 125.10 | 2023-06-09 11:15:00 | 124.90 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-06-06 15:00:00 | 125.20 | 2023-06-09 11:15:00 | 124.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-06-07 12:45:00 | 125.65 | 2023-06-09 11:15:00 | 124.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-06-30 13:30:00 | 126.35 | 2023-07-10 11:15:00 | 127.65 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2023-07-31 14:15:00 | 133.80 | 2023-08-02 10:15:00 | 131.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-08-01 09:15:00 | 134.90 | 2023-08-02 10:15:00 | 131.95 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2023-08-08 14:15:00 | 133.80 | 2023-08-10 13:15:00 | 132.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-08-10 09:15:00 | 133.90 | 2023-08-10 13:15:00 | 132.80 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-08-10 10:00:00 | 134.50 | 2023-08-10 13:15:00 | 132.80 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-08-10 11:15:00 | 133.85 | 2023-08-10 13:15:00 | 132.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-09-28 11:15:00 | 128.60 | 2023-10-06 14:15:00 | 127.95 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2023-10-16 09:15:00 | 138.15 | 2023-10-19 12:15:00 | 138.40 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2023-10-16 10:00:00 | 138.10 | 2023-10-19 12:15:00 | 138.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2023-10-30 15:15:00 | 134.00 | 2023-11-07 11:15:00 | 137.30 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2023-11-13 11:15:00 | 142.60 | 2023-11-23 09:15:00 | 148.70 | STOP_HIT | 1.00 | 4.28% |
| BUY | retest2 | 2023-11-13 12:30:00 | 142.40 | 2023-11-23 09:15:00 | 148.70 | STOP_HIT | 1.00 | 4.42% |
| SELL | retest2 | 2023-11-28 09:15:00 | 145.15 | 2023-11-28 12:15:00 | 147.90 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2023-12-07 11:00:00 | 156.35 | 2023-12-20 13:15:00 | 157.75 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2023-12-11 09:15:00 | 157.45 | 2023-12-20 13:15:00 | 157.75 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-01-05 13:00:00 | 154.30 | 2024-01-05 13:15:00 | 152.50 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-01-23 09:30:00 | 143.35 | 2024-01-29 10:15:00 | 142.80 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-01-31 09:15:00 | 143.80 | 2024-02-02 09:15:00 | 141.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-02-01 12:00:00 | 143.20 | 2024-02-02 09:15:00 | 141.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-02-01 13:45:00 | 143.10 | 2024-02-02 09:15:00 | 141.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-02-08 12:15:00 | 133.05 | 2024-02-13 12:15:00 | 133.95 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-02-08 14:00:00 | 133.00 | 2024-02-13 12:15:00 | 133.95 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-02-09 09:30:00 | 131.90 | 2024-02-13 12:15:00 | 133.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-02-12 09:30:00 | 132.45 | 2024-02-13 12:15:00 | 133.95 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-02-23 10:30:00 | 135.50 | 2024-02-27 13:15:00 | 135.60 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-02-23 13:15:00 | 135.50 | 2024-02-27 13:15:00 | 135.60 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-02-23 14:00:00 | 135.50 | 2024-02-27 13:15:00 | 135.60 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-02-23 15:00:00 | 135.35 | 2024-02-27 13:15:00 | 135.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-03-11 10:30:00 | 134.50 | 2024-03-13 11:15:00 | 127.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 10:30:00 | 134.50 | 2024-03-14 10:15:00 | 129.95 | STOP_HIT | 0.50 | 3.38% |
| BUY | retest1 | 2024-03-26 11:45:00 | 136.50 | 2024-03-28 14:15:00 | 134.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2024-03-26 12:45:00 | 136.45 | 2024-03-28 14:15:00 | 134.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2024-03-26 14:15:00 | 136.70 | 2024-03-28 14:15:00 | 134.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2024-03-28 09:15:00 | 137.15 | 2024-03-28 14:15:00 | 134.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-04-01 09:15:00 | 137.80 | 2024-04-04 09:15:00 | 151.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 13:15:00 | 156.35 | 2024-05-06 15:15:00 | 159.65 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2024-04-24 10:00:00 | 156.70 | 2024-05-06 15:15:00 | 159.65 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2024-04-24 13:15:00 | 156.50 | 2024-05-06 15:15:00 | 159.65 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2024-04-25 09:15:00 | 156.55 | 2024-05-06 15:15:00 | 159.65 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2024-04-25 14:00:00 | 157.60 | 2024-05-06 15:15:00 | 159.65 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-04-25 14:45:00 | 157.20 | 2024-05-06 15:15:00 | 159.65 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-05-13 10:30:00 | 146.95 | 2024-05-14 10:15:00 | 153.45 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2024-05-16 15:15:00 | 154.25 | 2024-05-17 15:15:00 | 152.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-05-17 10:30:00 | 154.10 | 2024-05-17 15:15:00 | 152.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-05-17 11:00:00 | 154.30 | 2024-05-17 15:15:00 | 152.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-17 13:15:00 | 153.75 | 2024-05-17 15:15:00 | 152.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-24 13:15:00 | 144.55 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-05-24 14:30:00 | 144.55 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-05-24 15:15:00 | 144.70 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-05-27 09:45:00 | 144.30 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-05-28 10:00:00 | 144.00 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-05-28 10:30:00 | 144.05 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-05-28 11:30:00 | 143.90 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-05-29 09:15:00 | 143.35 | 2024-05-31 10:15:00 | 144.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-06-14 10:30:00 | 150.39 | 2024-06-20 09:15:00 | 165.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-09 12:15:00 | 166.72 | 2024-07-18 09:15:00 | 158.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-09 12:15:00 | 166.72 | 2024-07-18 09:15:00 | 160.62 | STOP_HIT | 0.50 | 3.66% |
| BUY | retest2 | 2024-07-29 09:15:00 | 170.50 | 2024-08-02 11:15:00 | 168.43 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-08-06 13:30:00 | 163.98 | 2024-08-08 11:15:00 | 166.12 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-08 14:15:00 | 164.48 | 2024-08-12 12:15:00 | 165.11 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-08-08 14:45:00 | 164.22 | 2024-08-12 12:15:00 | 165.11 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-08-09 10:15:00 | 163.50 | 2024-08-12 13:15:00 | 165.69 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-08-09 14:30:00 | 163.02 | 2024-08-12 13:15:00 | 165.69 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-08-12 09:45:00 | 162.64 | 2024-08-12 13:15:00 | 165.69 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-09-20 10:15:00 | 165.80 | 2024-09-23 10:15:00 | 170.45 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-09-26 09:30:00 | 165.74 | 2024-10-04 14:15:00 | 157.81 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2024-09-26 12:00:00 | 166.12 | 2024-10-04 14:15:00 | 157.83 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2024-09-27 12:15:00 | 166.14 | 2024-10-07 09:15:00 | 157.45 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2024-09-27 12:45:00 | 165.75 | 2024-10-07 09:15:00 | 157.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 162.80 | 2024-10-07 10:15:00 | 154.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 09:30:00 | 165.74 | 2024-10-08 09:15:00 | 156.17 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2024-09-26 12:00:00 | 166.12 | 2024-10-08 09:15:00 | 156.17 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2024-09-27 12:15:00 | 166.14 | 2024-10-08 09:15:00 | 156.17 | STOP_HIT | 0.50 | 6.00% |
| SELL | retest2 | 2024-09-27 12:45:00 | 165.75 | 2024-10-08 09:15:00 | 156.17 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2024-10-03 09:15:00 | 162.80 | 2024-10-08 09:15:00 | 156.17 | STOP_HIT | 0.50 | 4.07% |
| BUY | retest2 | 2024-11-01 18:00:00 | 178.68 | 2024-11-11 09:15:00 | 176.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-11-04 10:15:00 | 176.29 | 2024-11-11 09:15:00 | 176.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-11-04 11:15:00 | 176.40 | 2024-11-11 09:15:00 | 176.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-11-18 09:15:00 | 170.25 | 2024-11-19 09:15:00 | 174.27 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-11-18 14:15:00 | 171.00 | 2024-11-19 09:15:00 | 174.27 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-11-19 14:45:00 | 171.25 | 2024-11-22 11:15:00 | 172.51 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2024-11-26 09:15:00 | 175.36 | 2024-12-02 09:15:00 | 177.07 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2024-12-02 14:15:00 | 180.59 | 2024-12-12 09:15:00 | 184.80 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2024-12-16 11:15:00 | 184.20 | 2024-12-16 12:15:00 | 185.03 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-12-20 12:15:00 | 179.42 | 2024-12-31 09:15:00 | 170.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 179.42 | 2024-12-31 13:15:00 | 172.03 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2025-01-08 10:15:00 | 168.89 | 2025-01-14 15:15:00 | 168.05 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-01-08 13:00:00 | 168.01 | 2025-01-14 15:15:00 | 168.05 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-01-08 14:45:00 | 168.63 | 2025-01-14 15:15:00 | 168.05 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-01-09 09:15:00 | 168.90 | 2025-01-14 15:15:00 | 168.05 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-01-10 09:15:00 | 167.14 | 2025-01-14 15:15:00 | 168.05 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-01-28 10:30:00 | 162.47 | 2025-01-28 14:15:00 | 167.41 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-01-30 15:15:00 | 175.50 | 2025-02-03 13:15:00 | 172.13 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-02-01 09:15:00 | 177.56 | 2025-02-03 13:15:00 | 172.13 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-02-03 10:15:00 | 174.47 | 2025-02-03 13:15:00 | 172.13 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-02-03 11:30:00 | 174.37 | 2025-02-03 13:15:00 | 172.13 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-02-27 13:30:00 | 147.26 | 2025-03-03 14:15:00 | 150.02 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-02-27 15:15:00 | 146.90 | 2025-03-04 09:15:00 | 150.65 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-02-28 09:30:00 | 147.57 | 2025-03-04 09:15:00 | 150.65 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-02-28 15:15:00 | 147.51 | 2025-03-04 09:15:00 | 150.65 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-03-03 09:45:00 | 146.61 | 2025-03-04 09:15:00 | 150.65 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-03-10 11:30:00 | 156.14 | 2025-03-10 15:15:00 | 152.33 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-03-10 12:15:00 | 155.96 | 2025-03-10 15:15:00 | 152.33 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-03-10 13:30:00 | 155.96 | 2025-03-10 15:15:00 | 152.33 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest1 | 2025-03-20 09:15:00 | 156.91 | 2025-03-24 14:15:00 | 158.73 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-04-24 09:15:00 | 179.28 | 2025-04-24 09:15:00 | 178.07 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-05 10:30:00 | 176.53 | 2025-05-05 12:15:00 | 180.64 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-05-07 11:00:00 | 180.49 | 2025-05-09 09:15:00 | 176.25 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-05-07 12:00:00 | 180.62 | 2025-05-09 09:15:00 | 176.25 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-05-20 15:15:00 | 194.00 | 2025-05-27 10:15:00 | 193.69 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-22 13:15:00 | 194.00 | 2025-05-27 10:15:00 | 193.69 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-22 13:45:00 | 194.21 | 2025-05-27 10:15:00 | 193.69 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-06-17 11:15:00 | 193.28 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-06-17 11:45:00 | 193.01 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-06-18 09:45:00 | 193.44 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-18 13:00:00 | 193.58 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-19 10:30:00 | 191.13 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-19 14:30:00 | 191.52 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-06-20 09:45:00 | 190.78 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-20 12:15:00 | 191.04 | 2025-06-24 10:15:00 | 194.09 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-07-14 11:15:00 | 213.27 | 2025-07-15 09:15:00 | 216.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-07-14 11:45:00 | 213.23 | 2025-07-15 09:15:00 | 216.90 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-07-22 12:15:00 | 211.89 | 2025-07-23 15:15:00 | 214.19 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-22 13:00:00 | 211.50 | 2025-07-23 15:15:00 | 214.19 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-07-23 10:45:00 | 211.64 | 2025-07-23 15:15:00 | 214.19 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-30 14:15:00 | 207.01 | 2025-07-31 09:15:00 | 208.06 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-08-04 09:15:00 | 216.34 | 2025-08-06 14:15:00 | 214.01 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-08-04 11:30:00 | 216.00 | 2025-08-06 14:15:00 | 214.01 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-04 13:00:00 | 215.65 | 2025-08-06 14:15:00 | 214.01 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-08-06 10:15:00 | 215.87 | 2025-08-06 14:15:00 | 214.01 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-19 11:45:00 | 212.72 | 2025-08-22 12:15:00 | 213.00 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-09-05 11:45:00 | 197.34 | 2025-09-05 15:15:00 | 200.03 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-09 09:15:00 | 199.74 | 2025-09-17 11:15:00 | 203.26 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-10-03 14:30:00 | 215.51 | 2025-10-08 13:15:00 | 216.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-10-20 10:15:00 | 223.70 | 2025-10-24 14:15:00 | 222.93 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-10-30 12:45:00 | 234.88 | 2025-10-31 09:15:00 | 231.71 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-11-07 10:15:00 | 252.40 | 2025-11-12 14:15:00 | 250.95 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-27 14:45:00 | 275.00 | 2025-11-28 11:15:00 | 273.30 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-11-27 15:15:00 | 274.86 | 2025-11-28 11:15:00 | 273.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-05 15:00:00 | 270.35 | 2025-12-12 12:15:00 | 271.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-12-19 13:45:00 | 278.75 | 2026-01-01 09:15:00 | 286.60 | STOP_HIT | 1.00 | 2.82% |
| SELL | retest2 | 2026-01-21 10:30:00 | 269.10 | 2026-01-21 14:15:00 | 278.35 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2026-01-29 13:30:00 | 287.60 | 2026-02-01 14:15:00 | 285.75 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-01 12:45:00 | 289.95 | 2026-02-01 14:15:00 | 285.75 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-09 11:15:00 | 289.60 | 2026-02-16 15:15:00 | 275.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 13:00:00 | 289.40 | 2026-02-16 15:15:00 | 275.26 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2026-02-11 10:00:00 | 289.50 | 2026-02-17 09:15:00 | 274.93 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-02-11 10:30:00 | 289.75 | 2026-02-17 09:15:00 | 275.02 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-02-12 09:15:00 | 282.50 | 2026-02-17 10:15:00 | 268.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 11:15:00 | 289.60 | 2026-02-17 13:15:00 | 276.70 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-09 13:00:00 | 289.40 | 2026-02-17 13:15:00 | 276.70 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2026-02-11 10:00:00 | 289.50 | 2026-02-17 13:15:00 | 276.70 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2026-02-11 10:30:00 | 289.75 | 2026-02-17 13:15:00 | 276.70 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2026-02-12 09:15:00 | 282.50 | 2026-02-17 13:15:00 | 276.70 | STOP_HIT | 0.50 | 2.05% |
| BUY | retest2 | 2026-02-25 09:15:00 | 290.30 | 2026-02-27 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-25 14:45:00 | 290.05 | 2026-02-27 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-02-26 09:15:00 | 290.60 | 2026-02-27 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-02-26 15:00:00 | 290.80 | 2026-02-27 10:15:00 | 285.25 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-03-11 11:30:00 | 249.90 | 2026-03-13 09:15:00 | 237.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 12:15:00 | 249.10 | 2026-03-13 09:15:00 | 236.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 249.90 | 2026-03-16 11:15:00 | 234.35 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2026-03-11 12:15:00 | 249.10 | 2026-03-16 11:15:00 | 234.35 | STOP_HIT | 0.50 | 5.92% |
| BUY | retest2 | 2026-03-20 09:15:00 | 245.85 | 2026-03-20 12:15:00 | 242.90 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-03-24 10:15:00 | 235.50 | 2026-03-25 09:15:00 | 243.35 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-03-24 15:00:00 | 235.30 | 2026-03-25 09:15:00 | 243.35 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2026-04-01 14:30:00 | 241.31 | 2026-04-06 11:15:00 | 244.72 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-04-06 10:30:00 | 241.00 | 2026-04-06 11:15:00 | 244.72 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-04-06 11:00:00 | 240.66 | 2026-04-06 11:15:00 | 244.72 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-04-10 09:15:00 | 255.84 | 2026-04-13 12:15:00 | 250.32 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-04-13 09:45:00 | 253.50 | 2026-04-13 12:15:00 | 250.32 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-16 11:00:00 | 250.19 | 2026-04-17 12:15:00 | 255.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-17 10:15:00 | 249.71 | 2026-04-17 12:15:00 | 255.40 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2026-04-29 11:00:00 | 277.52 | 2026-04-30 09:15:00 | 272.55 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest1 | 2026-04-29 11:45:00 | 280.16 | 2026-04-30 09:15:00 | 272.55 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2026-04-30 12:30:00 | 272.76 | 2026-04-30 13:15:00 | 271.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-04-30 13:15:00 | 273.00 | 2026-04-30 13:15:00 | 271.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-05-04 11:45:00 | 271.75 | 2026-05-08 15:15:00 | 258.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-06 10:45:00 | 272.25 | 2026-05-08 15:15:00 | 258.64 | PARTIAL | 0.50 | 5.00% |
