# ITI Ltd. (ITI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 300.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 234 |
| ALERT1 | 143 |
| ALERT2 | 134 |
| ALERT2_SKIP | 80 |
| ALERT3 | 318 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 144 |
| PARTIAL | 30 |
| TARGET_HIT | 15 |
| STOP_HIT | 135 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 180 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 82 / 98
- **Target hits / Stop hits / Partials:** 15 / 135 / 30
- **Avg / median % per leg:** 1.15% / -0.41%
- **Sum % (uncompounded):** 207.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 24 | 39.3% | 6 | 53 | 2 | 0.61% | 37.0% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 2 | 2 | 2 | 6.18% | 37.1% |
| BUY @ 3rd Alert (retest2) | 55 | 18 | 32.7% | 4 | 51 | 0 | -0.00% | -0.1% |
| SELL (all) | 119 | 58 | 48.7% | 9 | 82 | 28 | 1.43% | 170.7% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.39% | -8.8% |
| SELL @ 3rd Alert (retest2) | 117 | 58 | 49.6% | 9 | 80 | 28 | 1.53% | 179.4% |
| retest1 (combined) | 8 | 6 | 75.0% | 2 | 4 | 2 | 3.54% | 28.3% |
| retest2 (combined) | 172 | 76 | 44.2% | 13 | 131 | 28 | 1.04% | 179.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 10:15:00 | 106.40 | 105.81 | 105.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 12:15:00 | 106.60 | 105.96 | 105.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 09:15:00 | 105.75 | 106.17 | 105.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 105.75 | 106.17 | 105.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 105.75 | 106.17 | 105.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:00:00 | 105.75 | 106.17 | 105.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 105.20 | 105.97 | 105.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:45:00 | 104.90 | 105.97 | 105.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 11:15:00 | 105.35 | 105.85 | 105.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 13:15:00 | 105.05 | 105.59 | 105.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 15:15:00 | 105.00 | 104.50 | 104.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 15:15:00 | 105.00 | 104.50 | 104.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 105.00 | 104.50 | 104.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 106.65 | 104.50 | 104.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 106.85 | 104.97 | 105.07 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 106.60 | 105.30 | 105.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 110.80 | 106.80 | 106.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 09:15:00 | 108.30 | 109.07 | 107.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 108.30 | 109.07 | 107.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 108.30 | 109.07 | 107.90 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 106.70 | 107.51 | 107.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 12:15:00 | 106.35 | 106.76 | 107.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 09:15:00 | 106.70 | 106.62 | 106.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 106.70 | 106.62 | 106.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 106.70 | 106.62 | 106.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 105.00 | 106.71 | 106.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 09:45:00 | 105.60 | 105.37 | 105.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 10:00:00 | 106.00 | 105.69 | 105.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-05 10:15:00 | 108.05 | 106.16 | 105.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 10:15:00 | 108.05 | 106.16 | 105.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 12:15:00 | 108.70 | 107.04 | 106.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 110.90 | 111.56 | 110.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 11:45:00 | 110.50 | 111.56 | 110.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 110.65 | 111.36 | 110.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 110.50 | 111.36 | 110.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 112.10 | 111.51 | 110.74 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 108.15 | 110.26 | 110.40 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 13:15:00 | 109.50 | 108.45 | 108.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 09:15:00 | 112.40 | 109.35 | 108.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 15:15:00 | 110.80 | 110.85 | 109.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 110.90 | 110.86 | 110.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 110.90 | 110.86 | 110.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:45:00 | 110.65 | 110.86 | 110.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 110.20 | 110.63 | 110.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:30:00 | 110.05 | 110.63 | 110.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 110.50 | 110.61 | 110.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:15:00 | 109.90 | 110.61 | 110.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 109.20 | 110.33 | 110.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:45:00 | 109.10 | 110.33 | 110.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 109.35 | 110.13 | 110.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:45:00 | 109.35 | 110.13 | 110.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 108.80 | 109.86 | 109.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 14:15:00 | 108.65 | 109.45 | 109.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 108.25 | 108.22 | 108.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 10:15:00 | 108.25 | 108.22 | 108.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 108.00 | 108.17 | 108.68 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 14:15:00 | 108.80 | 108.67 | 108.66 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 108.35 | 108.61 | 108.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 10:15:00 | 108.05 | 108.46 | 108.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 12:15:00 | 108.00 | 107.92 | 108.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-30 13:00:00 | 108.00 | 107.92 | 108.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 107.35 | 107.62 | 107.92 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 13:15:00 | 108.40 | 108.09 | 108.07 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 09:15:00 | 107.00 | 108.01 | 108.05 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 110.80 | 108.26 | 108.06 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 108.45 | 108.68 | 108.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 107.60 | 108.38 | 108.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 12:15:00 | 108.45 | 108.34 | 108.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 12:15:00 | 108.45 | 108.34 | 108.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 108.45 | 108.34 | 108.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:45:00 | 108.30 | 108.34 | 108.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 13:15:00 | 108.25 | 108.32 | 108.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 13:45:00 | 108.35 | 108.32 | 108.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 108.55 | 108.37 | 108.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 15:00:00 | 108.55 | 108.37 | 108.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 108.35 | 108.37 | 108.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:15:00 | 108.90 | 108.37 | 108.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 108.40 | 108.37 | 108.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:00:00 | 108.00 | 108.27 | 108.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 15:15:00 | 107.80 | 108.17 | 108.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 12:15:00 | 108.05 | 108.24 | 108.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 14:15:00 | 107.95 | 108.17 | 108.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 108.10 | 108.16 | 108.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 11:15:00 | 107.80 | 108.06 | 108.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 12:15:00 | 109.25 | 108.28 | 108.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 12:15:00 | 109.25 | 108.28 | 108.25 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 107.15 | 108.06 | 108.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 14:15:00 | 106.80 | 107.80 | 108.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 107.70 | 107.43 | 107.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 14:15:00 | 107.70 | 107.43 | 107.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 107.70 | 107.43 | 107.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:30:00 | 107.85 | 107.43 | 107.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 107.85 | 107.52 | 107.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 108.85 | 107.52 | 107.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 108.95 | 107.80 | 107.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 111.60 | 108.99 | 108.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 09:15:00 | 110.00 | 110.19 | 109.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-19 09:30:00 | 109.95 | 110.19 | 109.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 109.75 | 110.11 | 109.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:00:00 | 109.75 | 110.11 | 109.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 109.80 | 110.04 | 109.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:30:00 | 109.55 | 110.04 | 109.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 109.65 | 109.97 | 109.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:00:00 | 109.65 | 109.97 | 109.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 109.70 | 109.91 | 109.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:30:00 | 109.40 | 109.91 | 109.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 109.70 | 109.87 | 109.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 15:00:00 | 109.70 | 109.87 | 109.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 15:15:00 | 109.25 | 109.75 | 109.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:15:00 | 109.65 | 109.75 | 109.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 109.15 | 109.63 | 109.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:45:00 | 109.20 | 109.63 | 109.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 109.00 | 109.50 | 109.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:45:00 | 108.80 | 109.50 | 109.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 11:15:00 | 109.00 | 109.40 | 109.42 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 14:15:00 | 109.85 | 109.35 | 109.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 09:15:00 | 110.80 | 109.72 | 109.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 14:15:00 | 109.70 | 109.89 | 109.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-24 15:00:00 | 109.70 | 109.89 | 109.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 109.45 | 109.80 | 109.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:45:00 | 109.00 | 109.66 | 109.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 10:15:00 | 109.10 | 109.55 | 109.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 12:15:00 | 108.70 | 109.31 | 109.45 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 111.55 | 109.46 | 109.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 11:15:00 | 112.50 | 110.33 | 109.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 13:15:00 | 111.90 | 112.33 | 111.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 13:15:00 | 111.90 | 112.33 | 111.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 13:15:00 | 111.90 | 112.33 | 111.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 13:30:00 | 111.65 | 112.33 | 111.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 112.25 | 112.50 | 112.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:30:00 | 112.30 | 112.54 | 112.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 113.00 | 112.84 | 112.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 112.90 | 112.84 | 112.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 112.25 | 112.72 | 112.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:45:00 | 111.95 | 112.72 | 112.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 111.70 | 112.52 | 112.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:00:00 | 111.70 | 112.52 | 112.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 111.30 | 112.27 | 112.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 110.65 | 111.95 | 112.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 110.60 | 110.46 | 111.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 15:00:00 | 110.60 | 110.46 | 111.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 110.65 | 110.49 | 110.99 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 10:15:00 | 112.30 | 111.14 | 111.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 11:15:00 | 119.15 | 112.74 | 111.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 119.25 | 120.55 | 118.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 119.25 | 120.55 | 118.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 119.25 | 120.55 | 118.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:30:00 | 118.65 | 120.55 | 118.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 10:15:00 | 118.15 | 120.07 | 118.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 10:30:00 | 118.25 | 120.07 | 118.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 119.15 | 119.89 | 118.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 14:45:00 | 119.40 | 119.51 | 118.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 13:15:00 | 117.50 | 119.21 | 118.82 | SL hit (close<static) qty=1.00 sl=117.85 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 13:15:00 | 118.40 | 118.66 | 118.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 117.95 | 118.51 | 118.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 117.10 | 115.10 | 116.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 117.10 | 115.10 | 116.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 117.10 | 115.10 | 116.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:45:00 | 116.95 | 115.10 | 116.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 115.60 | 115.20 | 116.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 11:30:00 | 115.05 | 115.28 | 116.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 15:00:00 | 115.10 | 115.58 | 116.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 09:45:00 | 115.10 | 115.21 | 115.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 14:00:00 | 115.05 | 115.06 | 115.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 113.65 | 114.57 | 115.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:15:00 | 112.75 | 113.94 | 114.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 09:45:00 | 113.10 | 113.54 | 114.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 14:30:00 | 112.95 | 113.37 | 113.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 115.90 | 113.82 | 114.02 | SL hit (close>static) qty=1.00 sl=115.35 alert=retest2 |

### Cycle 25 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 115.40 | 114.40 | 114.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 13:15:00 | 116.40 | 115.02 | 114.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 118.65 | 118.84 | 117.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 14:15:00 | 118.80 | 118.83 | 117.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 118.80 | 118.83 | 117.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 118.80 | 118.83 | 117.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 118.15 | 118.69 | 117.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 118.10 | 118.69 | 117.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 117.75 | 118.51 | 117.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:15:00 | 116.30 | 118.51 | 117.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 116.30 | 118.06 | 117.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 116.40 | 118.06 | 117.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 116.30 | 117.71 | 117.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:00:00 | 116.30 | 117.71 | 117.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 116.05 | 117.38 | 117.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 115.55 | 116.62 | 117.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 118.70 | 116.96 | 117.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 10:15:00 | 118.70 | 116.96 | 117.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 118.70 | 116.96 | 117.12 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 119.60 | 117.49 | 117.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 121.85 | 119.17 | 118.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 119.55 | 119.98 | 119.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 119.55 | 119.98 | 119.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 119.55 | 119.98 | 119.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 10:00:00 | 120.35 | 120.05 | 119.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 120.30 | 120.10 | 119.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 11:45:00 | 121.30 | 120.42 | 119.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 15:15:00 | 122.35 | 120.57 | 120.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 13:15:00 | 125.00 | 126.04 | 126.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 13:15:00 | 125.00 | 126.04 | 126.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 14:15:00 | 124.60 | 125.75 | 126.01 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 11:15:00 | 138.85 | 127.99 | 126.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 12:15:00 | 145.10 | 131.41 | 128.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 160.80 | 161.08 | 150.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 14:00:00 | 168.90 | 163.25 | 155.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-13 14:15:00 | 185.79 | 170.74 | 159.26 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 30 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 181.00 | 188.17 | 188.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 179.95 | 183.17 | 185.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 190.85 | 180.07 | 182.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 190.85 | 180.07 | 182.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 190.85 | 180.07 | 182.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:30:00 | 190.30 | 180.07 | 182.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 190.85 | 182.22 | 182.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:30:00 | 190.85 | 182.22 | 182.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 11:15:00 | 190.85 | 183.95 | 183.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 09:15:00 | 194.45 | 189.31 | 186.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 194.60 | 194.97 | 191.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 09:45:00 | 194.30 | 194.97 | 191.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 192.95 | 194.33 | 192.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:45:00 | 192.15 | 194.33 | 192.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 196.25 | 194.15 | 192.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 12:15:00 | 203.65 | 195.38 | 193.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 14:15:00 | 202.30 | 197.04 | 194.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:00:00 | 201.50 | 202.62 | 200.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 12:15:00 | 201.55 | 202.37 | 200.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 200.75 | 202.05 | 200.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 12:45:00 | 201.00 | 202.05 | 200.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 205.30 | 202.70 | 201.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-04 10:15:00 | 198.05 | 200.71 | 201.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 198.05 | 200.71 | 201.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 194.95 | 199.56 | 200.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 10:15:00 | 203.60 | 194.41 | 195.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 10:15:00 | 203.60 | 194.41 | 195.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 203.60 | 194.41 | 195.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 11:00:00 | 203.60 | 194.41 | 195.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 11:15:00 | 215.30 | 198.59 | 196.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-09 13:15:00 | 221.75 | 205.98 | 200.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 224.95 | 224.95 | 219.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 10:00:00 | 224.95 | 224.95 | 219.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 255.25 | 231.58 | 225.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:00:00 | 257.65 | 240.20 | 230.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-16 13:15:00 | 283.42 | 270.73 | 255.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 286.30 | 291.37 | 291.84 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 303.65 | 293.83 | 292.91 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 270.85 | 290.71 | 292.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 270.10 | 282.79 | 288.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 12:15:00 | 266.00 | 261.79 | 268.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 13:00:00 | 266.00 | 261.79 | 268.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 277.20 | 264.87 | 269.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:00:00 | 277.20 | 264.87 | 269.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 270.60 | 266.01 | 269.50 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 12:15:00 | 273.70 | 271.36 | 271.12 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 15:15:00 | 268.25 | 271.32 | 271.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 266.55 | 270.06 | 270.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 14:15:00 | 271.25 | 267.28 | 268.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 14:15:00 | 271.25 | 267.28 | 268.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 271.25 | 267.28 | 268.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:45:00 | 272.60 | 267.28 | 268.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 270.85 | 267.99 | 268.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:15:00 | 273.10 | 267.99 | 268.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 10:15:00 | 271.20 | 269.28 | 269.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 12:15:00 | 272.55 | 270.19 | 269.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 270.50 | 270.77 | 270.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 15:15:00 | 270.50 | 270.77 | 270.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 270.50 | 270.77 | 270.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 272.40 | 270.77 | 270.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 10:00:00 | 281.95 | 273.00 | 271.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 11:15:00 | 272.80 | 274.73 | 274.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 11:15:00 | 272.80 | 274.73 | 274.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 15:15:00 | 270.90 | 273.10 | 274.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 263.70 | 261.45 | 264.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 263.70 | 261.45 | 264.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 263.70 | 261.45 | 264.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:15:00 | 267.35 | 261.45 | 264.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 267.50 | 262.66 | 265.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:30:00 | 268.70 | 262.66 | 265.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 268.00 | 263.73 | 265.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:30:00 | 269.00 | 263.73 | 265.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 268.35 | 266.25 | 266.16 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 11:15:00 | 265.85 | 266.42 | 266.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 12:15:00 | 265.30 | 266.20 | 266.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 13:15:00 | 267.30 | 266.42 | 266.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 13:15:00 | 267.30 | 266.42 | 266.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 267.30 | 266.42 | 266.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:00:00 | 267.30 | 266.42 | 266.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-11-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 14:15:00 | 266.80 | 266.49 | 266.46 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 09:15:00 | 265.80 | 266.42 | 266.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 11:15:00 | 264.45 | 265.78 | 266.12 | Break + close below crossover candle low |

### Cycle 45 — BUY (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 09:15:00 | 281.75 | 267.09 | 266.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 11:15:00 | 284.50 | 272.74 | 269.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 11:15:00 | 276.70 | 277.23 | 273.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-21 12:00:00 | 276.70 | 277.23 | 273.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 275.00 | 276.20 | 274.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 273.35 | 276.20 | 274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 281.40 | 277.24 | 274.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:30:00 | 273.00 | 277.24 | 274.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 274.65 | 276.90 | 275.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:00:00 | 274.65 | 276.90 | 275.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 273.70 | 276.26 | 275.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:45:00 | 273.00 | 276.26 | 275.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 273.50 | 274.90 | 274.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 274.50 | 274.90 | 274.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 11:00:00 | 274.10 | 274.64 | 274.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 11:15:00 | 273.25 | 274.37 | 274.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 11:15:00 | 273.25 | 274.37 | 274.48 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 12:15:00 | 278.85 | 275.26 | 274.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 15:15:00 | 279.45 | 276.72 | 275.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 281.70 | 282.84 | 279.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-24 15:00:00 | 281.70 | 282.84 | 279.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 277.75 | 281.58 | 279.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:00:00 | 277.75 | 281.58 | 279.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 279.10 | 281.08 | 279.77 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 14:15:00 | 276.15 | 278.60 | 278.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 15:15:00 | 275.20 | 277.92 | 278.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 11:15:00 | 270.40 | 270.40 | 272.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-01 12:00:00 | 270.40 | 270.40 | 272.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 269.25 | 269.27 | 271.11 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 12:15:00 | 276.80 | 272.40 | 272.19 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 09:15:00 | 272.15 | 272.69 | 272.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 270.55 | 272.13 | 272.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 13:15:00 | 272.40 | 272.18 | 272.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 13:15:00 | 272.40 | 272.18 | 272.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 272.40 | 272.18 | 272.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 14:00:00 | 272.40 | 272.18 | 272.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 271.75 | 272.09 | 272.39 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 308.45 | 279.19 | 275.56 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 09:15:00 | 300.60 | 302.80 | 302.86 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 317.15 | 301.97 | 301.18 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 283.40 | 300.33 | 301.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 282.80 | 296.82 | 299.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 291.55 | 290.78 | 294.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 307.05 | 290.78 | 294.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 305.60 | 293.74 | 295.50 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 302.20 | 297.08 | 296.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 304.50 | 299.48 | 298.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 13:15:00 | 301.75 | 301.76 | 300.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 13:45:00 | 301.55 | 301.76 | 300.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 299.50 | 301.96 | 301.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 299.95 | 301.96 | 301.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 300.10 | 301.59 | 300.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 299.70 | 301.59 | 300.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 15:15:00 | 301.00 | 301.31 | 300.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 09:15:00 | 303.45 | 301.31 | 300.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 313.80 | 301.16 | 300.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 12:45:00 | 302.90 | 302.83 | 301.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 09:15:00 | 308.55 | 311.17 | 311.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 308.55 | 311.17 | 311.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 303.90 | 309.07 | 310.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 309.00 | 306.29 | 308.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 309.00 | 306.29 | 308.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 309.00 | 306.29 | 308.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 12:15:00 | 303.10 | 306.04 | 307.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 14:15:00 | 303.30 | 305.05 | 306.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 11:15:00 | 306.95 | 306.54 | 306.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 11:15:00 | 306.95 | 306.54 | 306.51 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 306.00 | 306.44 | 306.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 09:15:00 | 304.35 | 305.89 | 306.21 | Break + close below crossover candle low |

### Cycle 59 — BUY (started 2024-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 13:15:00 | 319.30 | 308.33 | 307.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 361.10 | 320.96 | 313.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 348.40 | 360.83 | 351.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 348.40 | 360.83 | 351.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 348.40 | 360.83 | 351.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 341.30 | 360.83 | 351.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 351.45 | 358.95 | 351.08 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 11:15:00 | 343.60 | 348.43 | 348.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-19 14:15:00 | 342.40 | 345.73 | 347.24 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 09:15:00 | 363.15 | 348.62 | 348.25 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 10:15:00 | 340.90 | 349.16 | 349.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 332.70 | 345.87 | 348.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 338.95 | 337.79 | 342.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 338.95 | 337.79 | 342.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 338.95 | 337.79 | 342.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:30:00 | 334.65 | 337.79 | 342.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 337.50 | 337.74 | 342.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 11:15:00 | 336.10 | 337.74 | 342.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 13:00:00 | 335.90 | 336.77 | 341.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 11:00:00 | 335.20 | 339.49 | 340.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 09:15:00 | 352.75 | 339.12 | 339.46 | SL hit (close>static) qty=1.00 sl=346.50 alert=retest2 |

### Cycle 63 — BUY (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 10:15:00 | 349.00 | 341.09 | 340.32 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 14:15:00 | 339.30 | 340.54 | 340.64 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 09:15:00 | 342.10 | 340.92 | 340.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 352.40 | 344.22 | 342.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 346.15 | 350.44 | 347.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 14:15:00 | 346.15 | 350.44 | 347.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 346.15 | 350.44 | 347.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 14:45:00 | 343.40 | 350.44 | 347.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 343.50 | 349.06 | 346.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 348.40 | 349.06 | 346.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 11:00:00 | 353.00 | 349.20 | 347.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 10:15:00 | 344.65 | 348.66 | 348.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 344.65 | 348.66 | 348.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 332.00 | 342.37 | 345.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 302.85 | 302.16 | 315.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:45:00 | 302.85 | 302.16 | 315.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 313.75 | 299.18 | 302.50 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 311.50 | 305.46 | 304.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 13:15:00 | 312.80 | 306.93 | 305.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 309.15 | 309.87 | 308.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 14:30:00 | 311.20 | 310.26 | 308.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 09:15:00 | 326.76 | 313.43 | 310.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-02-20 13:15:00 | 324.10 | 324.62 | 320.33 | SL hit (close<ema200) qty=0.50 sl=324.62 alert=retest1 |

### Cycle 68 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 314.10 | 319.38 | 319.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 308.40 | 315.78 | 317.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 313.10 | 312.25 | 314.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 09:15:00 | 316.35 | 312.25 | 314.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 314.90 | 312.78 | 314.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:15:00 | 313.50 | 312.78 | 314.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 12:00:00 | 313.60 | 313.26 | 314.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 310.75 | 313.85 | 314.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 10:15:00 | 314.90 | 313.86 | 313.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 314.90 | 313.86 | 313.78 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 311.50 | 313.71 | 313.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 306.30 | 312.22 | 313.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 295.20 | 294.01 | 297.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 295.25 | 294.26 | 297.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 295.25 | 294.26 | 297.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 296.80 | 294.26 | 297.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 291.70 | 293.89 | 296.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 10:45:00 | 290.75 | 292.42 | 294.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 12:15:00 | 289.85 | 292.14 | 294.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 15:00:00 | 289.65 | 291.37 | 293.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:30:00 | 288.80 | 289.42 | 291.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 282.20 | 283.53 | 286.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:45:00 | 282.50 | 283.53 | 286.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:15:00 | 276.21 | 280.38 | 283.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:15:00 | 275.36 | 280.38 | 283.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:15:00 | 275.17 | 280.38 | 283.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:15:00 | 274.36 | 280.38 | 283.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-12 09:15:00 | 261.68 | 269.65 | 275.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 71 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 278.35 | 260.51 | 259.27 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 11:15:00 | 251.40 | 259.08 | 259.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 248.10 | 251.69 | 254.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 253.00 | 251.51 | 253.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 253.00 | 251.51 | 253.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 253.00 | 251.51 | 253.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 253.00 | 251.51 | 253.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 250.75 | 251.35 | 253.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 15:15:00 | 247.70 | 250.94 | 252.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 254.60 | 251.16 | 252.54 | SL hit (close>static) qty=1.00 sl=253.35 alert=retest2 |

### Cycle 73 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 256.25 | 253.49 | 253.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 257.80 | 254.35 | 253.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 253.55 | 257.90 | 256.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 253.55 | 257.90 | 256.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 253.55 | 257.90 | 256.66 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 253.35 | 255.43 | 255.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 15:15:00 | 252.00 | 254.09 | 254.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 12:15:00 | 254.70 | 253.78 | 254.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 12:15:00 | 254.70 | 253.78 | 254.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 254.70 | 253.78 | 254.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:45:00 | 254.45 | 253.78 | 254.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 259.65 | 254.96 | 254.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:00:00 | 259.65 | 254.96 | 254.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 254.35 | 254.83 | 254.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 252.70 | 254.83 | 254.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 11:15:00 | 257.45 | 255.28 | 255.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 257.45 | 255.28 | 255.05 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 15:15:00 | 252.50 | 254.90 | 255.02 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 264.50 | 256.82 | 255.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 269.95 | 263.14 | 259.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 270.75 | 272.76 | 270.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 10:00:00 | 270.75 | 272.76 | 270.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 271.25 | 272.46 | 270.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 270.10 | 272.46 | 270.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 268.75 | 271.71 | 270.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 268.75 | 271.71 | 270.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 271.00 | 271.57 | 270.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 14:30:00 | 271.80 | 271.38 | 270.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 10:30:00 | 273.05 | 272.02 | 270.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 271.80 | 272.51 | 272.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 10:15:00 | 271.00 | 272.25 | 272.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 271.00 | 272.25 | 272.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 268.15 | 271.19 | 271.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 269.70 | 269.04 | 270.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 10:15:00 | 269.70 | 269.04 | 270.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 269.70 | 269.04 | 270.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 273.60 | 269.04 | 270.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 269.55 | 269.10 | 269.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 14:00:00 | 269.55 | 269.10 | 269.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 269.55 | 269.19 | 269.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 14:45:00 | 270.70 | 269.19 | 269.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 265.50 | 268.31 | 269.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:30:00 | 264.35 | 266.83 | 268.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 14:00:00 | 264.20 | 266.30 | 268.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 251.13 | 262.81 | 265.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 250.99 | 262.81 | 265.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 256.80 | 255.88 | 260.07 | SL hit (close>ema200) qty=0.50 sl=255.88 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 279.15 | 257.93 | 255.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 307.35 | 276.15 | 265.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 295.35 | 295.53 | 288.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:15:00 | 296.40 | 295.53 | 288.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 291.50 | 293.16 | 290.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 291.50 | 293.16 | 290.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 292.80 | 293.09 | 290.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 11:15:00 | 294.00 | 293.09 | 290.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:45:00 | 297.85 | 294.27 | 292.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 298.00 | 303.23 | 303.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 11:15:00 | 298.00 | 303.23 | 303.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 291.40 | 299.37 | 301.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 297.20 | 295.17 | 297.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 297.20 | 295.17 | 297.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 297.20 | 295.17 | 297.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 297.20 | 295.17 | 297.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 293.35 | 292.51 | 295.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 15:00:00 | 293.35 | 292.51 | 295.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 280.90 | 277.75 | 280.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:30:00 | 280.15 | 277.75 | 280.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 279.85 | 278.17 | 280.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:30:00 | 281.85 | 278.17 | 280.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 282.20 | 278.98 | 280.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 282.90 | 278.98 | 280.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 280.65 | 279.31 | 280.47 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 284.80 | 281.27 | 281.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 285.55 | 282.73 | 281.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 307.50 | 308.37 | 300.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 307.50 | 308.37 | 300.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 309.25 | 309.80 | 306.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 320.25 | 307.56 | 307.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 13:15:00 | 313.10 | 314.69 | 314.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 13:15:00 | 313.10 | 314.69 | 314.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 308.10 | 313.37 | 314.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 308.95 | 307.95 | 310.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:00:00 | 308.95 | 307.95 | 310.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 298.85 | 305.44 | 308.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 292.75 | 301.79 | 305.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 14:15:00 | 303.55 | 300.86 | 300.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 14:15:00 | 303.55 | 300.86 | 300.61 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 295.30 | 300.00 | 300.28 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 304.30 | 300.32 | 299.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 307.90 | 303.14 | 301.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 293.20 | 301.83 | 301.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 293.20 | 301.83 | 301.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 293.20 | 301.83 | 301.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 292.00 | 301.83 | 301.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 283.45 | 298.15 | 299.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 265.80 | 291.68 | 296.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 288.10 | 278.74 | 283.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 288.10 | 278.74 | 283.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 288.10 | 278.74 | 283.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 288.10 | 278.74 | 283.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 294.60 | 281.91 | 284.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 296.20 | 281.91 | 284.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 291.40 | 285.58 | 285.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 295.00 | 291.27 | 288.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 294.50 | 296.05 | 294.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 15:15:00 | 294.50 | 296.05 | 294.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 294.50 | 296.05 | 294.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 307.50 | 296.05 | 294.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 304.65 | 306.88 | 307.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 304.65 | 306.88 | 307.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 304.10 | 306.22 | 306.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 309.50 | 306.64 | 306.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 309.50 | 306.64 | 306.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 309.50 | 306.64 | 306.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 309.50 | 306.64 | 306.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 309.50 | 307.21 | 307.06 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 306.30 | 307.02 | 307.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 303.30 | 306.28 | 306.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 305.60 | 305.48 | 306.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 305.60 | 305.48 | 306.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 305.60 | 305.48 | 306.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:45:00 | 306.50 | 305.48 | 306.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 308.80 | 305.71 | 306.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 308.80 | 305.71 | 306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 308.85 | 306.33 | 306.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 308.85 | 306.33 | 306.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 318.70 | 308.81 | 307.47 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 306.55 | 309.36 | 309.36 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 325.75 | 311.32 | 309.92 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 15:15:00 | 307.00 | 309.28 | 309.58 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 310.40 | 309.43 | 309.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 311.00 | 309.74 | 309.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 309.20 | 309.63 | 309.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 15:15:00 | 309.20 | 309.63 | 309.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 309.20 | 309.63 | 309.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 309.15 | 309.45 | 309.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 308.55 | 309.27 | 309.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 307.70 | 308.95 | 309.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 14:15:00 | 309.75 | 308.56 | 308.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 14:15:00 | 309.75 | 308.56 | 308.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 309.75 | 308.56 | 308.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:45:00 | 310.60 | 308.56 | 308.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 309.60 | 308.77 | 308.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 310.25 | 308.77 | 308.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 314.00 | 309.90 | 309.43 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 309.50 | 311.47 | 311.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 303.00 | 309.67 | 310.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 307.30 | 306.59 | 308.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 307.30 | 306.59 | 308.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 307.30 | 306.59 | 308.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 313.50 | 306.59 | 308.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 307.15 | 306.70 | 308.14 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 314.05 | 309.06 | 308.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 12:15:00 | 323.90 | 312.43 | 310.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 318.35 | 320.36 | 317.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 318.35 | 320.36 | 317.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 318.10 | 319.52 | 317.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 318.10 | 319.52 | 317.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 317.00 | 319.02 | 317.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:45:00 | 317.25 | 319.02 | 317.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 316.25 | 318.47 | 317.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 315.60 | 318.47 | 317.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 313.05 | 316.03 | 316.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 309.70 | 314.76 | 315.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 308.00 | 299.47 | 303.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 308.00 | 299.47 | 303.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 308.00 | 299.47 | 303.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 309.10 | 299.47 | 303.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 302.50 | 300.08 | 303.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 302.05 | 300.73 | 303.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:15:00 | 302.10 | 302.00 | 303.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 286.95 | 299.64 | 301.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 287.00 | 299.64 | 301.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 300.25 | 298.10 | 300.17 | SL hit (close>ema200) qty=0.50 sl=298.10 alert=retest2 |

### Cycle 101 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 304.65 | 301.24 | 301.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 310.10 | 305.64 | 304.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 308.05 | 308.27 | 306.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 308.05 | 308.27 | 306.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 315.75 | 309.87 | 307.69 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 305.50 | 308.90 | 309.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 15:15:00 | 304.00 | 306.51 | 307.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 284.30 | 284.28 | 289.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 285.65 | 284.28 | 289.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 289.15 | 286.22 | 288.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 289.45 | 286.22 | 288.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 295.40 | 288.06 | 289.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 295.40 | 288.06 | 289.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 295.65 | 289.58 | 290.11 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 293.00 | 290.91 | 290.67 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 12:15:00 | 289.55 | 290.79 | 290.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 285.95 | 289.74 | 290.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 12:15:00 | 289.80 | 289.45 | 290.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 12:15:00 | 289.80 | 289.45 | 290.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 289.80 | 289.45 | 290.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:30:00 | 291.70 | 289.45 | 290.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 289.50 | 289.46 | 290.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 289.50 | 289.46 | 290.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 287.60 | 289.09 | 289.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:30:00 | 286.25 | 288.58 | 289.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 15:15:00 | 285.00 | 288.58 | 289.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 286.50 | 283.76 | 284.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 12:15:00 | 285.95 | 284.89 | 284.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 285.95 | 284.89 | 284.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 286.20 | 285.15 | 284.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 285.60 | 286.26 | 285.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 285.60 | 286.26 | 285.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 285.60 | 286.26 | 285.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 285.60 | 286.26 | 285.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 286.30 | 286.27 | 285.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 13:45:00 | 288.00 | 286.46 | 285.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 14:30:00 | 290.65 | 287.11 | 286.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 15:15:00 | 296.25 | 297.77 | 297.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 296.25 | 297.77 | 297.97 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 09:15:00 | 302.35 | 298.69 | 298.37 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 299.55 | 301.27 | 301.40 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 307.40 | 302.12 | 301.72 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 302.45 | 302.98 | 302.99 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 303.15 | 303.01 | 303.01 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 302.15 | 302.84 | 302.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 299.80 | 302.12 | 302.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 298.55 | 297.71 | 299.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 298.55 | 297.71 | 299.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 298.55 | 297.71 | 299.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:30:00 | 299.45 | 297.71 | 299.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 298.25 | 297.82 | 299.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:30:00 | 299.05 | 297.82 | 299.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 286.60 | 285.06 | 288.60 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 296.95 | 291.30 | 290.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 300.05 | 293.05 | 291.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 294.20 | 294.50 | 292.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 12:00:00 | 294.20 | 294.50 | 292.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 291.95 | 293.91 | 292.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 291.95 | 293.91 | 292.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 293.20 | 293.77 | 292.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 292.95 | 293.77 | 292.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 294.35 | 293.89 | 293.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 294.80 | 293.89 | 293.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 290.25 | 293.16 | 292.79 | SL hit (close<static) qty=1.00 sl=292.05 alert=retest2 |

### Cycle 114 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 290.50 | 292.21 | 292.39 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 293.85 | 292.22 | 292.21 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 290.05 | 291.95 | 292.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 289.40 | 290.62 | 291.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 279.60 | 279.49 | 281.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 15:00:00 | 279.60 | 279.49 | 281.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 281.15 | 279.99 | 281.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 11:30:00 | 280.60 | 281.33 | 281.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 12:15:00 | 280.30 | 281.33 | 281.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 12:15:00 | 266.57 | 272.53 | 275.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 13:15:00 | 266.29 | 271.31 | 274.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 262.95 | 260.15 | 265.07 | SL hit (close>ema200) qty=0.50 sl=260.15 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 246.40 | 245.48 | 245.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 248.40 | 246.20 | 245.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 245.85 | 246.60 | 246.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 14:15:00 | 245.85 | 246.60 | 246.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 245.85 | 246.60 | 246.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:30:00 | 245.45 | 246.60 | 246.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 246.60 | 246.60 | 246.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 244.95 | 246.60 | 246.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 248.85 | 247.05 | 246.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 244.85 | 247.05 | 246.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 247.20 | 247.08 | 246.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 247.20 | 247.08 | 246.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 247.55 | 247.17 | 246.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 247.70 | 247.17 | 246.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 247.75 | 247.43 | 246.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:30:00 | 247.10 | 247.43 | 246.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 246.80 | 247.48 | 247.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 246.80 | 247.48 | 247.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 246.40 | 247.26 | 246.96 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 13:15:00 | 246.40 | 246.75 | 246.77 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 247.00 | 246.80 | 246.79 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 245.90 | 246.62 | 246.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 244.35 | 246.16 | 246.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 246.00 | 245.77 | 246.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 13:15:00 | 246.00 | 245.77 | 246.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 246.00 | 245.77 | 246.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 246.00 | 245.77 | 246.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 245.50 | 245.48 | 245.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 246.95 | 245.48 | 245.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 244.45 | 245.28 | 245.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:30:00 | 242.80 | 244.57 | 245.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:15:00 | 241.45 | 244.27 | 245.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 241.50 | 240.43 | 241.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 242.85 | 240.97 | 241.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 230.66 | 236.12 | 238.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 229.38 | 236.12 | 238.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 229.42 | 236.12 | 238.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 230.71 | 236.12 | 238.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 218.52 | 225.97 | 231.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 234.40 | 220.60 | 219.13 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 221.55 | 224.27 | 224.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 15:15:00 | 220.30 | 222.29 | 223.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 224.59 | 222.75 | 223.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 224.59 | 222.75 | 223.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 224.59 | 222.75 | 223.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 224.59 | 222.75 | 223.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 224.75 | 223.15 | 223.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 223.30 | 223.16 | 223.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 12:30:00 | 223.46 | 223.17 | 223.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 226.75 | 224.27 | 223.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 226.75 | 224.27 | 223.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 229.41 | 226.06 | 224.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 11:15:00 | 316.39 | 324.09 | 306.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-12 12:00:00 | 316.39 | 324.09 | 306.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 308.71 | 321.01 | 306.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:00:00 | 308.71 | 321.01 | 306.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 297.66 | 316.34 | 306.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:00:00 | 297.66 | 316.34 | 306.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 294.61 | 312.00 | 305.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:30:00 | 296.25 | 312.00 | 305.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 296.01 | 302.69 | 302.00 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 290.79 | 300.31 | 300.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 09:15:00 | 286.43 | 293.86 | 297.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 13:15:00 | 290.03 | 288.78 | 293.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 14:00:00 | 290.03 | 288.78 | 293.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 294.13 | 289.47 | 292.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 283.68 | 289.47 | 292.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 282.21 | 288.02 | 291.82 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 293.36 | 290.90 | 290.84 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 282.20 | 289.63 | 290.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 15:15:00 | 278.50 | 283.95 | 286.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 279.95 | 276.80 | 279.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 279.95 | 276.80 | 279.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 279.95 | 276.80 | 279.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:30:00 | 281.24 | 276.80 | 279.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 277.40 | 276.92 | 278.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 280.00 | 276.92 | 278.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 281.00 | 277.73 | 279.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:30:00 | 281.55 | 277.73 | 279.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 278.69 | 277.92 | 279.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 15:15:00 | 277.00 | 278.38 | 278.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:00:00 | 276.91 | 277.86 | 278.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 295.51 | 280.17 | 279.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 295.51 | 280.17 | 279.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 12:15:00 | 298.50 | 287.68 | 283.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 288.00 | 291.06 | 286.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 288.00 | 291.06 | 286.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 287.30 | 290.30 | 286.59 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 09:15:00 | 285.60 | 286.19 | 286.20 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 286.75 | 286.30 | 286.25 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 11:15:00 | 285.40 | 286.12 | 286.17 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 287.00 | 286.21 | 286.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 288.40 | 286.92 | 286.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 287.05 | 287.97 | 287.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 287.05 | 287.97 | 287.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 287.05 | 287.97 | 287.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 286.20 | 287.97 | 287.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 286.65 | 287.70 | 287.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:15:00 | 286.00 | 287.70 | 287.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 287.40 | 287.64 | 287.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 286.20 | 287.64 | 287.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 286.80 | 287.48 | 287.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:00:00 | 286.80 | 287.48 | 287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 286.70 | 287.32 | 287.25 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 14:15:00 | 284.70 | 286.80 | 287.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 15:15:00 | 282.85 | 286.01 | 286.64 | Break + close below crossover candle low |

### Cycle 133 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 312.75 | 291.36 | 289.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 337.75 | 306.51 | 297.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 364.25 | 378.13 | 360.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 09:45:00 | 364.55 | 378.13 | 360.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 361.05 | 372.44 | 360.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:45:00 | 361.70 | 372.44 | 360.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 362.75 | 370.50 | 361.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 13:30:00 | 364.60 | 369.17 | 361.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 14:45:00 | 363.80 | 367.74 | 361.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 15:15:00 | 359.00 | 365.99 | 361.19 | SL hit (close<static) qty=1.00 sl=360.75 alert=retest2 |

### Cycle 134 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 356.05 | 361.59 | 361.71 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 380.80 | 363.24 | 361.94 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 364.85 | 366.65 | 366.76 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 370.10 | 367.34 | 367.06 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 360.55 | 365.94 | 366.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 356.15 | 362.99 | 365.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 14:15:00 | 342.15 | 339.28 | 343.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 14:15:00 | 342.15 | 339.28 | 343.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 342.15 | 339.28 | 343.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 342.15 | 339.28 | 343.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 340.60 | 339.55 | 343.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 336.45 | 338.95 | 342.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 356.20 | 335.52 | 336.23 | SL hit (close>static) qty=1.00 sl=343.25 alert=retest2 |

### Cycle 139 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 372.20 | 342.85 | 339.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 380.00 | 350.28 | 343.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 12:15:00 | 382.00 | 383.05 | 373.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 13:00:00 | 382.00 | 383.05 | 373.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 375.90 | 381.33 | 376.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 375.90 | 381.33 | 376.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 376.55 | 380.37 | 376.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 376.50 | 380.37 | 376.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 377.70 | 379.84 | 376.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 377.30 | 379.84 | 376.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 379.50 | 379.32 | 376.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 382.90 | 379.32 | 376.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-03 11:15:00 | 421.19 | 391.42 | 383.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 443.45 | 474.49 | 477.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 426.00 | 441.89 | 452.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 10:15:00 | 391.50 | 390.85 | 402.32 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 09:45:00 | 373.55 | 386.34 | 394.85 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:15:00 | 366.05 | 379.37 | 387.18 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 386.00 | 376.38 | 382.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-20 12:15:00 | 386.00 | 376.38 | 382.82 | SL hit (close>ema400) qty=1.00 sl=382.82 alert=retest1 |

### Cycle 141 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 318.75 | 313.34 | 312.80 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 311.80 | 313.35 | 313.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 306.30 | 311.89 | 312.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 281.40 | 280.49 | 287.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 281.40 | 280.49 | 287.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 287.60 | 281.91 | 287.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 287.60 | 281.91 | 287.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 285.25 | 282.58 | 287.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:45:00 | 288.90 | 282.58 | 287.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 288.00 | 283.66 | 287.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 288.00 | 283.66 | 287.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 289.50 | 284.83 | 287.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 284.80 | 284.83 | 287.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 290.15 | 285.89 | 288.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 274.60 | 285.30 | 286.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 260.87 | 270.38 | 276.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-18 09:15:00 | 247.14 | 257.78 | 266.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 143 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 269.20 | 259.86 | 258.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 09:15:00 | 282.65 | 269.44 | 264.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 276.40 | 278.63 | 272.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 276.40 | 278.63 | 272.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 276.40 | 278.63 | 272.79 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 261.00 | 271.23 | 272.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 246.05 | 259.07 | 264.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 251.08 | 249.85 | 256.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 251.08 | 249.85 | 256.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 251.08 | 249.85 | 256.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 246.05 | 249.41 | 255.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 10:15:00 | 253.50 | 251.55 | 251.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 253.50 | 251.55 | 251.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 258.50 | 254.44 | 253.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 259.00 | 262.15 | 258.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 259.00 | 262.15 | 258.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 261.35 | 261.99 | 258.75 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 247.90 | 256.31 | 257.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 15:15:00 | 243.97 | 247.15 | 250.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 253.34 | 248.39 | 250.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 253.34 | 248.39 | 250.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 253.34 | 248.39 | 250.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:30:00 | 253.36 | 248.39 | 250.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 253.36 | 249.38 | 250.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 253.36 | 249.38 | 250.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 253.36 | 250.18 | 250.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:30:00 | 253.36 | 250.18 | 250.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 14:15:00 | 253.36 | 251.73 | 251.53 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 250.20 | 251.28 | 251.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 14:15:00 | 248.00 | 250.42 | 250.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 249.70 | 248.64 | 249.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 13:15:00 | 249.70 | 248.64 | 249.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 249.70 | 248.64 | 249.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 249.70 | 248.64 | 249.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 251.12 | 249.14 | 249.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:30:00 | 252.70 | 249.14 | 249.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 251.50 | 249.61 | 249.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 252.00 | 249.61 | 249.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 257.00 | 251.09 | 250.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 14:15:00 | 263.50 | 256.46 | 253.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 256.00 | 257.26 | 254.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 256.00 | 257.26 | 254.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 255.55 | 256.91 | 254.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 261.69 | 257.96 | 256.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:00:00 | 260.00 | 257.96 | 256.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 13:45:00 | 260.03 | 259.65 | 257.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 12:00:00 | 262.00 | 264.22 | 262.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 260.00 | 263.37 | 262.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 260.00 | 263.37 | 262.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 261.05 | 262.91 | 262.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 255.10 | 260.72 | 261.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 255.10 | 260.72 | 261.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 253.03 | 257.53 | 259.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 253.50 | 251.70 | 254.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 253.50 | 251.70 | 254.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 253.50 | 251.70 | 254.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 249.00 | 251.24 | 253.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 14:30:00 | 248.75 | 250.94 | 252.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 249.95 | 250.94 | 252.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 249.40 | 251.00 | 252.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 251.00 | 251.00 | 252.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 249.10 | 251.00 | 251.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:45:00 | 249.00 | 251.18 | 251.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 13:15:00 | 252.90 | 251.65 | 251.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 252.90 | 251.65 | 251.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 253.99 | 252.12 | 251.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 249.00 | 251.76 | 251.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 249.00 | 251.76 | 251.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 249.00 | 251.76 | 251.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 248.16 | 251.76 | 251.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 249.00 | 251.21 | 251.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 234.04 | 245.59 | 248.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 238.00 | 237.26 | 241.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 238.00 | 237.26 | 241.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 238.00 | 237.26 | 241.78 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 251.95 | 242.33 | 242.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 255.00 | 250.33 | 246.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 15:15:00 | 251.55 | 251.90 | 249.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:15:00 | 255.20 | 251.90 | 249.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 268.20 | 257.43 | 253.77 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 09:15:00 | 267.96 | 257.43 | 253.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 262.72 | 264.85 | 260.29 | SL hit (close<ema200) qty=0.50 sl=264.85 alert=retest1 |

### Cycle 154 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 265.80 | 276.60 | 277.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 262.12 | 266.51 | 270.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 265.45 | 263.00 | 266.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 265.45 | 263.00 | 266.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 265.45 | 263.00 | 266.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 257.64 | 263.08 | 265.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 244.76 | 250.81 | 254.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 251.40 | 250.48 | 253.38 | SL hit (close>ema200) qty=0.50 sl=250.48 alert=retest2 |

### Cycle 155 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 253.75 | 248.52 | 247.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 261.90 | 251.19 | 249.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 280.30 | 281.47 | 277.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 280.30 | 281.47 | 277.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 277.50 | 280.59 | 277.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 275.60 | 280.59 | 277.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 277.50 | 279.97 | 277.91 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 271.95 | 276.31 | 276.68 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 280.85 | 277.06 | 276.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 303.70 | 285.16 | 281.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 346.60 | 354.04 | 338.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:30:00 | 347.00 | 354.04 | 338.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 333.55 | 345.55 | 341.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 329.30 | 345.55 | 341.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 329.30 | 338.38 | 338.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 327.40 | 333.43 | 335.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 331.45 | 331.07 | 332.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 10:30:00 | 332.05 | 331.07 | 332.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 330.45 | 331.06 | 332.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:30:00 | 332.25 | 331.06 | 332.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 327.00 | 329.02 | 330.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:45:00 | 326.40 | 328.29 | 330.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:00:00 | 326.40 | 327.91 | 330.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 11:15:00 | 339.00 | 329.41 | 329.47 | SL hit (close>static) qty=1.00 sl=333.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 336.80 | 330.89 | 330.14 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 328.95 | 330.06 | 330.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 326.60 | 329.25 | 329.72 | Break + close below crossover candle low |

### Cycle 161 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 334.10 | 329.68 | 329.64 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 329.00 | 329.61 | 329.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 323.45 | 328.20 | 329.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 319.00 | 318.35 | 321.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:30:00 | 318.50 | 318.35 | 321.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 327.30 | 320.44 | 322.02 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 325.60 | 323.23 | 323.01 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 318.55 | 322.20 | 322.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 313.55 | 316.64 | 319.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 310.95 | 310.13 | 313.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 310.95 | 310.13 | 313.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 312.00 | 310.75 | 313.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 312.90 | 310.75 | 313.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 318.95 | 312.29 | 313.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 318.95 | 312.29 | 313.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 318.10 | 313.45 | 313.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 318.90 | 313.45 | 313.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 321.45 | 315.05 | 314.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 324.65 | 320.50 | 318.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 320.00 | 321.88 | 320.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 320.00 | 321.88 | 320.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 320.00 | 321.88 | 320.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 320.00 | 321.88 | 320.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 320.95 | 321.69 | 320.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 322.45 | 321.19 | 320.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:30:00 | 322.05 | 321.28 | 320.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 322.25 | 321.33 | 320.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:30:00 | 322.00 | 321.62 | 321.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 319.90 | 321.50 | 321.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 319.90 | 321.50 | 321.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 322.00 | 321.60 | 321.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 326.05 | 321.60 | 321.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 326.55 | 327.19 | 327.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 326.55 | 327.19 | 327.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 324.10 | 326.37 | 326.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 319.55 | 319.12 | 320.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 319.55 | 319.12 | 320.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 319.55 | 319.12 | 320.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 321.15 | 319.12 | 320.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 319.20 | 319.09 | 320.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 319.90 | 319.09 | 320.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 316.45 | 318.60 | 319.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 315.40 | 318.60 | 319.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 315.75 | 317.15 | 318.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 315.75 | 316.94 | 318.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 321.15 | 318.01 | 317.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 321.15 | 318.01 | 317.85 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 317.65 | 318.33 | 318.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 317.20 | 318.11 | 318.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 317.50 | 317.16 | 317.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 317.50 | 317.16 | 317.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 317.50 | 317.16 | 317.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:30:00 | 316.70 | 317.12 | 317.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 12:30:00 | 316.95 | 317.04 | 317.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 315.50 | 316.98 | 317.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 315.60 | 315.95 | 316.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 315.15 | 315.79 | 316.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 313.95 | 315.79 | 316.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 313.25 | 314.24 | 315.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 313.55 | 313.84 | 314.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 12:15:00 | 301.10 | 305.99 | 309.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:15:00 | 300.86 | 302.46 | 306.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:15:00 | 299.72 | 302.46 | 306.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:15:00 | 299.82 | 302.46 | 306.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:15:00 | 298.25 | 302.46 | 306.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:15:00 | 297.59 | 302.46 | 306.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:15:00 | 297.87 | 302.46 | 306.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 296.40 | 295.65 | 299.66 | SL hit (close>ema200) qty=0.50 sl=295.65 alert=retest2 |

### Cycle 169 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 302.80 | 292.49 | 291.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 09:15:00 | 321.30 | 302.80 | 297.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 310.50 | 313.84 | 307.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:30:00 | 310.25 | 313.84 | 307.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 308.90 | 312.29 | 308.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 308.90 | 312.29 | 308.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 305.65 | 310.39 | 308.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 305.65 | 310.39 | 308.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 301.55 | 308.63 | 307.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 296.20 | 308.63 | 307.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 294.65 | 305.83 | 306.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 291.95 | 300.46 | 301.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 296.30 | 294.41 | 297.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 296.30 | 294.41 | 297.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 296.30 | 294.41 | 297.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 296.30 | 294.41 | 297.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 293.40 | 294.21 | 296.67 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 296.70 | 295.50 | 295.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 299.75 | 296.35 | 295.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 296.35 | 296.91 | 296.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 15:15:00 | 296.35 | 296.91 | 296.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 296.35 | 296.91 | 296.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 294.60 | 296.91 | 296.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 295.55 | 296.64 | 296.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 294.25 | 296.64 | 296.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 294.65 | 296.24 | 296.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 294.10 | 296.24 | 296.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 294.45 | 295.88 | 296.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 293.90 | 295.48 | 295.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 296.50 | 295.69 | 295.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 13:15:00 | 296.50 | 295.69 | 295.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 296.50 | 295.69 | 295.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:45:00 | 296.30 | 295.69 | 295.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 295.65 | 295.68 | 295.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 295.65 | 295.68 | 295.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 295.85 | 295.71 | 295.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 295.15 | 295.71 | 295.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 294.20 | 295.41 | 295.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 293.90 | 294.94 | 295.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 293.65 | 294.55 | 295.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 305.85 | 288.37 | 286.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 305.85 | 288.37 | 286.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 316.00 | 303.13 | 296.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 308.75 | 311.01 | 304.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 11:15:00 | 306.45 | 309.26 | 307.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 306.45 | 309.26 | 307.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 306.45 | 309.26 | 307.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 307.80 | 308.97 | 307.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 308.00 | 308.97 | 307.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 305.65 | 307.28 | 307.19 | SL hit (close<static) qty=1.00 sl=306.05 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 308.00 | 308.65 | 308.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 307.30 | 308.18 | 308.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 308.80 | 307.84 | 308.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 308.80 | 307.84 | 308.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 308.80 | 307.84 | 308.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 309.00 | 307.84 | 308.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 308.00 | 307.87 | 308.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 308.05 | 307.87 | 308.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 308.80 | 308.06 | 308.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 308.80 | 308.06 | 308.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 308.95 | 308.24 | 308.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 309.35 | 308.24 | 308.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 309.35 | 308.46 | 308.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 323.20 | 311.75 | 309.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 320.35 | 320.48 | 317.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 320.35 | 320.48 | 317.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 319.30 | 320.55 | 318.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 320.25 | 320.31 | 318.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:00:00 | 320.00 | 320.03 | 318.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 316.20 | 318.63 | 318.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 316.20 | 318.63 | 318.67 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 320.00 | 318.69 | 318.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 324.75 | 320.75 | 319.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 320.30 | 322.21 | 321.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 320.30 | 322.21 | 321.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 320.30 | 322.21 | 321.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 319.55 | 322.21 | 321.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 319.80 | 321.73 | 321.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 319.80 | 321.73 | 321.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 321.70 | 322.32 | 321.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 321.70 | 322.32 | 321.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 322.90 | 322.44 | 321.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 320.70 | 322.44 | 321.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 321.35 | 322.22 | 321.68 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 318.10 | 320.76 | 321.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 315.60 | 318.63 | 319.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 304.45 | 304.23 | 307.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 304.45 | 304.23 | 307.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 303.85 | 302.12 | 303.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 305.35 | 302.12 | 303.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 303.50 | 302.39 | 303.95 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 308.50 | 305.17 | 304.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 308.80 | 305.89 | 305.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 340.95 | 349.35 | 338.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:00:00 | 340.95 | 349.35 | 338.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 343.80 | 348.24 | 339.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 347.70 | 341.84 | 339.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 337.90 | 341.32 | 340.03 | SL hit (close<static) qty=1.00 sl=339.05 alert=retest2 |

### Cycle 180 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 333.35 | 338.54 | 339.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 332.25 | 335.67 | 337.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 347.80 | 330.36 | 331.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 347.80 | 330.36 | 331.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 347.80 | 330.36 | 331.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 348.70 | 330.36 | 331.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 344.95 | 333.28 | 332.47 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 329.95 | 334.40 | 334.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 329.00 | 331.14 | 332.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 331.50 | 330.68 | 331.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 331.50 | 330.68 | 331.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 331.50 | 330.68 | 331.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 329.50 | 330.68 | 331.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 329.60 | 330.47 | 331.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:30:00 | 328.50 | 330.24 | 330.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 331.95 | 330.37 | 330.67 | SL hit (close>static) qty=1.00 sl=331.80 alert=retest2 |

### Cycle 183 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 333.00 | 331.24 | 331.04 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 330.05 | 331.04 | 331.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 14:15:00 | 329.40 | 330.60 | 330.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 326.20 | 326.05 | 327.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:45:00 | 326.65 | 326.05 | 327.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 327.05 | 326.27 | 327.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:15:00 | 326.80 | 326.27 | 327.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 326.80 | 326.38 | 327.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 326.25 | 326.38 | 327.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 324.95 | 326.09 | 327.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 323.50 | 325.69 | 326.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 321.60 | 323.98 | 325.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 323.70 | 317.33 | 317.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 323.70 | 317.33 | 317.14 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 316.80 | 320.39 | 320.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 315.50 | 318.35 | 319.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 325.25 | 305.94 | 306.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 325.25 | 305.94 | 306.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 325.25 | 305.94 | 306.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 327.25 | 305.94 | 306.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 326.90 | 310.13 | 308.61 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 309.60 | 312.91 | 312.96 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 315.20 | 312.01 | 311.94 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 310.80 | 312.05 | 312.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 309.30 | 311.30 | 311.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 306.00 | 304.98 | 306.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 15:15:00 | 306.00 | 304.98 | 306.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 306.00 | 304.98 | 306.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 303.95 | 304.98 | 306.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 303.25 | 304.63 | 306.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:45:00 | 302.40 | 304.21 | 305.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 302.80 | 304.21 | 305.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:45:00 | 302.65 | 303.87 | 305.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 302.60 | 303.66 | 305.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 299.75 | 302.68 | 304.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 312.50 | 303.91 | 303.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 312.50 | 303.91 | 303.84 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 298.90 | 304.00 | 304.29 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 306.00 | 304.23 | 304.14 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 303.25 | 303.92 | 304.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 302.30 | 303.60 | 303.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 302.30 | 302.19 | 302.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 11:45:00 | 302.50 | 302.19 | 302.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 302.50 | 302.18 | 302.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 302.50 | 302.18 | 302.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 302.50 | 302.24 | 302.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 303.95 | 302.24 | 302.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 303.10 | 302.41 | 302.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 302.65 | 302.50 | 302.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 305.10 | 303.07 | 302.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 305.10 | 303.07 | 302.95 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 302.15 | 303.33 | 303.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 301.35 | 302.38 | 302.88 | Break + close below crossover candle low |

### Cycle 197 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 326.80 | 302.21 | 300.81 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 307.40 | 312.07 | 312.51 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 312.35 | 311.27 | 311.26 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 14:15:00 | 310.60 | 311.19 | 311.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 15:15:00 | 309.95 | 310.94 | 311.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 311.95 | 311.14 | 311.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 311.95 | 311.14 | 311.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 311.95 | 311.14 | 311.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 311.95 | 311.14 | 311.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 311.90 | 311.29 | 311.25 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 310.70 | 311.14 | 311.19 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 312.70 | 311.26 | 311.20 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 309.65 | 311.62 | 311.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 309.30 | 311.16 | 311.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 310.60 | 309.49 | 310.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 14:15:00 | 310.60 | 309.49 | 310.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 310.60 | 309.49 | 310.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 310.60 | 309.49 | 310.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 309.05 | 309.40 | 310.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 309.20 | 309.40 | 310.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 308.75 | 309.27 | 309.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 307.90 | 308.67 | 309.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 292.50 | 298.83 | 302.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 297.00 | 295.66 | 298.84 | SL hit (close>ema200) qty=0.50 sl=295.66 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 314.20 | 301.70 | 301.17 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 13:15:00 | 302.00 | 302.63 | 302.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 15:15:00 | 300.85 | 302.05 | 302.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 302.65 | 302.17 | 302.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 302.65 | 302.17 | 302.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 302.65 | 302.17 | 302.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 297.45 | 301.88 | 302.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 282.58 | 290.52 | 294.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 289.40 | 287.94 | 291.34 | SL hit (close>ema200) qty=0.50 sl=287.94 alert=retest2 |

### Cycle 207 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 292.65 | 285.25 | 284.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 294.00 | 290.53 | 289.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 288.50 | 290.66 | 289.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 288.50 | 290.66 | 289.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 288.50 | 290.66 | 289.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 288.50 | 290.66 | 289.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 285.70 | 289.67 | 289.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 287.75 | 289.67 | 289.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 285.25 | 288.79 | 288.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 282.55 | 287.54 | 288.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 283.50 | 282.14 | 284.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 283.50 | 282.14 | 284.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 285.00 | 282.71 | 284.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 288.95 | 282.71 | 284.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 287.45 | 283.66 | 284.88 | EMA400 retest candle locked (from downside) |

### Cycle 209 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 288.50 | 286.04 | 285.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 289.15 | 286.66 | 286.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 291.10 | 292.55 | 290.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 15:15:00 | 291.10 | 292.55 | 290.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 291.10 | 292.55 | 290.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 290.20 | 292.55 | 290.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 288.55 | 291.75 | 290.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 288.65 | 291.75 | 290.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 289.45 | 291.29 | 290.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 288.20 | 291.29 | 290.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 289.15 | 290.20 | 290.24 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 292.35 | 290.44 | 290.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 297.55 | 294.00 | 292.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 295.65 | 295.78 | 294.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 294.10 | 295.78 | 294.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 292.80 | 295.18 | 294.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 292.80 | 295.18 | 294.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 292.15 | 294.58 | 293.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 292.15 | 294.58 | 293.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 292.45 | 293.56 | 293.56 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 293.65 | 293.58 | 293.57 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 293.40 | 293.54 | 293.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 290.70 | 292.97 | 293.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 294.40 | 292.24 | 292.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 13:15:00 | 294.40 | 292.24 | 292.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 294.40 | 292.24 | 292.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:00:00 | 294.40 | 292.24 | 292.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 294.55 | 292.70 | 292.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 292.00 | 292.70 | 292.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 14:15:00 | 277.40 | 279.59 | 281.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 275.00 | 274.83 | 277.10 | SL hit (close>ema200) qty=0.50 sl=274.83 alert=retest2 |

### Cycle 215 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 262.60 | 259.91 | 259.77 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 257.65 | 259.55 | 259.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 248.70 | 257.38 | 258.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 252.35 | 251.83 | 254.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 252.35 | 251.83 | 254.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 252.35 | 251.83 | 254.44 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 261.00 | 255.76 | 255.27 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 252.10 | 255.34 | 255.48 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 260.50 | 256.00 | 255.69 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 247.45 | 255.57 | 255.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 245.45 | 253.54 | 254.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 270.70 | 251.51 | 252.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 270.70 | 251.51 | 252.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 270.70 | 251.51 | 252.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:00:00 | 270.70 | 251.51 | 252.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 10:15:00 | 273.00 | 255.81 | 254.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 15:15:00 | 287.60 | 269.05 | 261.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 14:15:00 | 275.30 | 278.66 | 270.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 15:00:00 | 275.30 | 278.66 | 270.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 280.10 | 278.95 | 271.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 267.60 | 278.95 | 271.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 266.75 | 276.51 | 271.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 267.30 | 276.51 | 271.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 269.20 | 275.04 | 271.16 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 14:15:00 | 265.50 | 268.88 | 269.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 15:15:00 | 263.25 | 267.75 | 268.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 09:15:00 | 270.65 | 268.33 | 268.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 270.65 | 268.33 | 268.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 270.65 | 268.33 | 268.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 277.85 | 268.33 | 268.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — BUY (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 10:15:00 | 272.30 | 269.13 | 269.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-19 11:15:00 | 278.50 | 271.00 | 269.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 09:15:00 | 269.05 | 272.08 | 271.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 269.05 | 272.08 | 271.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 269.05 | 272.08 | 271.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:15:00 | 268.70 | 272.08 | 271.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 270.00 | 271.66 | 270.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 270.85 | 271.66 | 270.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 269.15 | 270.53 | 270.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 269.15 | 270.53 | 270.55 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 270.80 | 270.59 | 270.57 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 269.00 | 270.27 | 270.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 259.85 | 268.19 | 269.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 258.15 | 256.82 | 260.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 257.60 | 256.82 | 260.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 261.75 | 257.60 | 259.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 260.80 | 257.60 | 259.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 261.45 | 258.37 | 259.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 262.75 | 258.37 | 259.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 259.90 | 259.86 | 260.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 256.70 | 259.86 | 260.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 243.86 | 252.94 | 256.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 252.84 | 246.00 | 250.11 | SL hit (close>ema200) qty=0.50 sl=246.00 alert=retest2 |

### Cycle 227 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 254.76 | 251.96 | 251.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 255.60 | 252.69 | 252.22 | Break + close above crossover candle high |

### Cycle 228 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 247.30 | 251.61 | 251.77 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 255.59 | 251.72 | 251.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 257.60 | 253.73 | 252.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 256.80 | 256.83 | 255.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 262.50 | 256.83 | 255.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 288.75 | 269.69 | 263.59 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 230 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 296.90 | 300.67 | 301.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 295.37 | 299.61 | 300.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 300.42 | 298.98 | 299.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 300.42 | 298.98 | 299.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 300.42 | 298.98 | 299.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 301.20 | 298.98 | 299.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 299.96 | 299.17 | 299.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 298.81 | 299.17 | 299.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 305.00 | 300.54 | 300.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 305.00 | 300.54 | 300.20 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 295.20 | 301.24 | 301.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 294.81 | 299.96 | 300.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 298.75 | 298.60 | 299.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 298.75 | 298.60 | 299.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 300.95 | 299.07 | 300.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 302.00 | 299.07 | 300.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 300.00 | 299.26 | 300.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 298.75 | 299.31 | 299.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 298.75 | 299.23 | 299.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 302.50 | 300.06 | 299.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 302.50 | 300.06 | 299.99 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 300.50 | 301.51 | 301.59 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-30 09:15:00 | 105.00 | 2023-06-05 10:15:00 | 108.05 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2023-05-31 09:45:00 | 105.60 | 2023-06-05 10:15:00 | 108.05 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2023-06-05 10:00:00 | 106.00 | 2023-06-05 10:15:00 | 108.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2023-07-11 12:00:00 | 108.00 | 2023-07-13 12:15:00 | 109.25 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2023-07-11 15:15:00 | 107.80 | 2023-07-13 12:15:00 | 109.25 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-07-12 12:15:00 | 108.05 | 2023-07-13 12:15:00 | 109.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-07-12 14:15:00 | 107.95 | 2023-07-13 12:15:00 | 109.25 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-07-13 11:15:00 | 107.80 | 2023-07-13 12:15:00 | 109.25 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-08-09 14:45:00 | 119.40 | 2023-08-10 13:15:00 | 117.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-08-10 15:15:00 | 119.65 | 2023-08-11 10:15:00 | 117.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-08-16 11:30:00 | 115.05 | 2023-08-22 09:15:00 | 115.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-08-16 15:00:00 | 115.10 | 2023-08-22 09:15:00 | 115.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-08-17 09:45:00 | 115.10 | 2023-08-22 09:15:00 | 115.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-08-17 14:00:00 | 115.05 | 2023-08-22 11:15:00 | 115.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2023-08-18 15:15:00 | 112.75 | 2023-08-22 11:15:00 | 115.40 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-08-21 09:45:00 | 113.10 | 2023-08-22 11:15:00 | 115.40 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-08-21 14:30:00 | 112.95 | 2023-08-22 11:15:00 | 115.40 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2023-09-01 11:45:00 | 121.30 | 2023-09-08 13:15:00 | 125.00 | STOP_HIT | 1.00 | 3.05% |
| BUY | retest2 | 2023-09-01 15:15:00 | 122.35 | 2023-09-08 13:15:00 | 125.00 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest1 | 2023-09-13 14:00:00 | 168.90 | 2023-09-13 14:15:00 | 185.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-27 12:15:00 | 203.65 | 2023-10-04 10:15:00 | 198.05 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2023-09-27 14:15:00 | 202.30 | 2023-10-04 10:15:00 | 198.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2023-09-29 11:00:00 | 201.50 | 2023-10-04 10:15:00 | 198.05 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-09-29 12:15:00 | 201.55 | 2023-10-04 10:15:00 | 198.05 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2023-10-13 12:00:00 | 257.65 | 2023-10-16 13:15:00 | 283.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-06 09:15:00 | 272.40 | 2023-11-08 11:15:00 | 272.80 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2023-11-06 10:00:00 | 281.95 | 2023-11-08 11:15:00 | 272.80 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2023-11-23 09:15:00 | 274.50 | 2023-11-23 11:15:00 | 273.25 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-11-23 11:00:00 | 274.10 | 2023-11-23 11:15:00 | 273.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-12-28 09:15:00 | 303.45 | 2024-01-08 09:15:00 | 308.55 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2023-12-29 09:15:00 | 313.80 | 2024-01-08 09:15:00 | 308.55 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-12-29 12:45:00 | 302.90 | 2024-01-08 09:15:00 | 308.55 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2024-01-10 12:15:00 | 303.10 | 2024-01-12 11:15:00 | 306.95 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-01-10 14:15:00 | 303.30 | 2024-01-12 11:15:00 | 306.95 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-01-24 11:15:00 | 336.10 | 2024-01-30 09:15:00 | 352.75 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2024-01-24 13:00:00 | 335.90 | 2024-01-30 09:15:00 | 352.75 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2024-01-29 11:00:00 | 335.20 | 2024-01-30 09:15:00 | 352.75 | STOP_HIT | 1.00 | -5.24% |
| BUY | retest2 | 2024-02-06 09:15:00 | 348.40 | 2024-02-08 10:15:00 | 344.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-02-06 11:00:00 | 353.00 | 2024-02-08 10:15:00 | 344.65 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2024-02-16 14:30:00 | 311.20 | 2024-02-19 09:15:00 | 326.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-16 14:30:00 | 311.20 | 2024-02-20 13:15:00 | 324.10 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2024-02-23 10:15:00 | 313.50 | 2024-02-27 10:15:00 | 314.90 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-02-23 12:00:00 | 313.60 | 2024-02-27 10:15:00 | 314.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-02-26 09:15:00 | 310.75 | 2024-02-27 10:15:00 | 314.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-03-05 10:45:00 | 290.75 | 2024-03-11 09:15:00 | 276.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 12:15:00 | 289.85 | 2024-03-11 09:15:00 | 275.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 15:00:00 | 289.65 | 2024-03-11 09:15:00 | 275.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:30:00 | 288.80 | 2024-03-11 09:15:00 | 274.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 10:45:00 | 290.75 | 2024-03-12 09:15:00 | 261.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 12:15:00 | 289.85 | 2024-03-12 09:15:00 | 260.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 15:00:00 | 289.65 | 2024-03-12 09:15:00 | 260.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-06 09:30:00 | 288.80 | 2024-03-12 09:15:00 | 259.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-20 15:15:00 | 247.70 | 2024-03-21 09:15:00 | 254.60 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-03-27 15:15:00 | 252.70 | 2024-03-28 11:15:00 | 257.45 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-04-04 14:30:00 | 271.80 | 2024-04-09 10:15:00 | 271.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-04-05 10:30:00 | 273.05 | 2024-04-09 10:15:00 | 271.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-04-09 09:15:00 | 271.80 | 2024-04-09 10:15:00 | 271.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-04-12 12:30:00 | 264.35 | 2024-04-15 09:15:00 | 251.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 14:00:00 | 264.20 | 2024-04-15 09:15:00 | 250.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 12:30:00 | 264.35 | 2024-04-16 09:15:00 | 256.80 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2024-04-12 14:00:00 | 264.20 | 2024-04-16 09:15:00 | 256.80 | STOP_HIT | 0.50 | 2.80% |
| BUY | retest2 | 2024-04-26 11:15:00 | 294.00 | 2024-05-03 11:15:00 | 298.00 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2024-04-29 09:45:00 | 297.85 | 2024-05-03 11:15:00 | 298.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-05-22 09:15:00 | 320.25 | 2024-05-24 13:15:00 | 313.10 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-05-29 09:15:00 | 292.75 | 2024-05-30 14:15:00 | 303.55 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2024-06-11 09:15:00 | 307.50 | 2024-06-19 12:15:00 | 304.65 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-07-22 12:15:00 | 302.05 | 2024-07-23 12:15:00 | 286.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 15:15:00 | 302.10 | 2024-07-23 12:15:00 | 287.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 302.05 | 2024-07-24 09:15:00 | 300.25 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2024-07-22 15:15:00 | 302.10 | 2024-07-24 09:15:00 | 300.25 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2024-07-24 11:00:00 | 302.05 | 2024-07-24 13:15:00 | 304.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-07-24 12:00:00 | 302.10 | 2024-07-24 13:15:00 | 304.65 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-08-13 14:30:00 | 286.25 | 2024-08-19 12:15:00 | 285.95 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-08-13 15:15:00 | 285.00 | 2024-08-19 12:15:00 | 285.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-08-19 10:15:00 | 286.50 | 2024-08-19 12:15:00 | 285.95 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-08-20 13:45:00 | 288.00 | 2024-08-26 15:15:00 | 296.25 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2024-08-20 14:30:00 | 290.65 | 2024-08-26 15:15:00 | 296.25 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2024-09-12 09:15:00 | 294.80 | 2024-09-12 09:15:00 | 290.25 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-25 11:30:00 | 280.60 | 2024-09-27 12:15:00 | 266.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 12:15:00 | 280.30 | 2024-09-27 13:15:00 | 266.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 11:30:00 | 280.60 | 2024-10-01 09:15:00 | 262.95 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2024-09-25 12:15:00 | 280.30 | 2024-10-01 09:15:00 | 262.95 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2024-10-17 09:30:00 | 242.80 | 2024-10-22 09:15:00 | 230.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 11:15:00 | 241.45 | 2024-10-22 09:15:00 | 229.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 241.50 | 2024-10-22 09:15:00 | 229.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:30:00 | 242.85 | 2024-10-22 09:15:00 | 230.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 242.80 | 2024-10-23 09:15:00 | 218.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-17 11:15:00 | 241.45 | 2024-10-23 09:15:00 | 217.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 241.50 | 2024-10-23 09:15:00 | 217.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 10:30:00 | 242.85 | 2024-10-23 09:15:00 | 218.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-05 11:30:00 | 223.30 | 2024-11-06 09:15:00 | 226.75 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-11-05 12:30:00 | 223.46 | 2024-11-06 09:15:00 | 226.75 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-11-26 15:15:00 | 277.00 | 2024-11-28 09:15:00 | 295.51 | STOP_HIT | 1.00 | -6.68% |
| SELL | retest2 | 2024-11-27 10:00:00 | 276.91 | 2024-11-28 09:15:00 | 295.51 | STOP_HIT | 1.00 | -6.72% |
| BUY | retest2 | 2024-12-11 13:30:00 | 364.60 | 2024-12-11 15:15:00 | 359.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-12-11 14:45:00 | 363.80 | 2024-12-11 15:15:00 | 359.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-12-12 09:15:00 | 375.10 | 2024-12-13 09:15:00 | 354.00 | STOP_HIT | 1.00 | -5.63% |
| BUY | retest2 | 2024-12-12 14:15:00 | 363.70 | 2024-12-13 09:15:00 | 354.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-12-26 09:30:00 | 336.45 | 2024-12-30 09:15:00 | 356.20 | STOP_HIT | 1.00 | -5.87% |
| BUY | retest2 | 2025-01-02 15:15:00 | 382.90 | 2025-01-03 11:15:00 | 421.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-01-17 09:45:00 | 373.55 | 2025-01-20 12:15:00 | 386.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest1 | 2025-01-20 09:15:00 | 366.05 | 2025-01-20 12:15:00 | 386.00 | STOP_HIT | 1.00 | -5.45% |
| SELL | retest2 | 2025-01-21 13:15:00 | 370.00 | 2025-01-22 09:15:00 | 351.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 13:15:00 | 370.00 | 2025-01-23 09:15:00 | 368.00 | STOP_HIT | 0.50 | 0.54% |
| SELL | retest2 | 2025-02-14 09:15:00 | 274.60 | 2025-02-17 09:15:00 | 260.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 274.60 | 2025-02-18 09:15:00 | 247.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 11:15:00 | 246.05 | 2025-03-06 10:15:00 | 253.50 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-03-21 09:30:00 | 261.69 | 2025-03-26 09:15:00 | 255.10 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-03-21 10:00:00 | 260.00 | 2025-03-26 09:15:00 | 255.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-03-21 13:45:00 | 260.03 | 2025-03-26 09:15:00 | 255.10 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-03-25 12:00:00 | 262.00 | 2025-03-26 09:15:00 | 255.10 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-03-28 13:00:00 | 249.00 | 2025-04-03 13:15:00 | 252.90 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-03-28 14:30:00 | 248.75 | 2025-04-03 13:15:00 | 252.90 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-03-28 15:00:00 | 249.95 | 2025-04-03 13:15:00 | 252.90 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-04-01 11:15:00 | 249.40 | 2025-04-03 13:15:00 | 252.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-04-02 09:15:00 | 249.10 | 2025-04-03 13:15:00 | 252.90 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-04-03 09:45:00 | 249.00 | 2025-04-03 13:15:00 | 252.90 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest1 | 2025-04-15 09:15:00 | 255.20 | 2025-04-16 09:15:00 | 267.96 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-15 09:15:00 | 255.20 | 2025-04-17 09:15:00 | 262.72 | STOP_HIT | 0.50 | 2.95% |
| BUY | retest2 | 2025-04-21 11:00:00 | 271.17 | 2025-04-25 09:15:00 | 265.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-04-21 13:45:00 | 270.95 | 2025-04-25 09:15:00 | 265.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-04-30 09:15:00 | 257.64 | 2025-05-07 09:15:00 | 244.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 257.64 | 2025-05-07 11:15:00 | 251.40 | STOP_HIT | 0.50 | 2.42% |
| SELL | retest2 | 2025-06-06 10:45:00 | 326.40 | 2025-06-09 11:15:00 | 339.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-06-06 12:00:00 | 326.40 | 2025-06-09 11:15:00 | 339.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2025-06-27 09:15:00 | 322.45 | 2025-07-04 12:15:00 | 326.55 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-06-27 10:30:00 | 322.05 | 2025-07-04 12:15:00 | 326.55 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-06-27 11:30:00 | 322.25 | 2025-07-04 12:15:00 | 326.55 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-06-27 12:30:00 | 322.00 | 2025-07-04 12:15:00 | 326.55 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-06-30 09:15:00 | 326.05 | 2025-07-04 12:15:00 | 326.55 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-07-11 10:15:00 | 315.40 | 2025-07-15 09:15:00 | 321.15 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-07-11 12:30:00 | 315.75 | 2025-07-15 09:15:00 | 321.15 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-07-11 14:15:00 | 315.75 | 2025-07-15 09:15:00 | 321.15 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-07-18 10:30:00 | 316.70 | 2025-07-25 12:15:00 | 301.10 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-07-18 12:30:00 | 316.95 | 2025-07-28 11:15:00 | 300.86 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-07-21 11:15:00 | 315.50 | 2025-07-28 11:15:00 | 299.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 315.60 | 2025-07-28 11:15:00 | 299.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:15:00 | 313.95 | 2025-07-28 11:15:00 | 298.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:30:00 | 313.25 | 2025-07-28 11:15:00 | 297.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 313.55 | 2025-07-28 11:15:00 | 297.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:30:00 | 316.70 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 6.41% |
| SELL | retest2 | 2025-07-18 12:30:00 | 316.95 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 6.48% |
| SELL | retest2 | 2025-07-21 11:15:00 | 315.50 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 6.05% |
| SELL | retest2 | 2025-07-22 10:15:00 | 315.60 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 6.08% |
| SELL | retest2 | 2025-07-22 11:15:00 | 313.95 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 5.59% |
| SELL | retest2 | 2025-07-23 09:30:00 | 313.25 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2025-07-24 10:30:00 | 313.55 | 2025-07-29 12:15:00 | 296.40 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2025-08-25 11:30:00 | 293.90 | 2025-09-02 09:15:00 | 305.85 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-08-25 14:15:00 | 293.65 | 2025-09-02 09:15:00 | 305.85 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2025-09-05 13:15:00 | 308.00 | 2025-09-08 11:15:00 | 305.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-09-08 13:00:00 | 308.85 | 2025-09-10 10:15:00 | 308.20 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-09-08 13:30:00 | 314.75 | 2025-09-11 09:15:00 | 308.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-09-09 15:00:00 | 309.70 | 2025-09-11 09:15:00 | 308.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-09-10 09:15:00 | 309.90 | 2025-09-11 09:15:00 | 308.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-17 13:15:00 | 320.25 | 2025-09-18 12:15:00 | 316.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-09-17 15:00:00 | 320.00 | 2025-09-18 12:15:00 | 316.20 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-10 09:15:00 | 347.70 | 2025-10-10 13:15:00 | 337.90 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-10-28 14:30:00 | 328.50 | 2025-10-29 09:15:00 | 331.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-11-04 11:15:00 | 323.50 | 2025-11-12 09:15:00 | 323.70 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-06 09:15:00 | 321.60 | 2025-11-12 09:15:00 | 323.70 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-04 11:45:00 | 302.40 | 2025-12-08 09:15:00 | 312.50 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-12-04 12:15:00 | 302.80 | 2025-12-08 09:15:00 | 312.50 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-12-04 12:45:00 | 302.65 | 2025-12-08 09:15:00 | 312.50 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-12-04 13:30:00 | 302.60 | 2025-12-08 09:15:00 | 312.50 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-12-12 10:45:00 | 302.65 | 2025-12-12 12:15:00 | 305.10 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-01-08 10:45:00 | 307.90 | 2026-01-12 09:15:00 | 292.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 307.90 | 2026-01-12 15:15:00 | 297.00 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-01-19 09:15:00 | 297.45 | 2026-01-21 10:15:00 | 282.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 297.45 | 2026-01-21 15:15:00 | 289.40 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2026-02-12 15:15:00 | 292.00 | 2026-02-23 14:15:00 | 277.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 15:15:00 | 292.00 | 2026-02-25 09:15:00 | 275.00 | STOP_HIT | 0.50 | 5.82% |
| BUY | retest2 | 2026-03-20 11:15:00 | 270.85 | 2026-03-20 13:15:00 | 269.15 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-03-27 09:15:00 | 256.70 | 2026-03-30 09:15:00 | 243.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 256.70 | 2026-04-01 09:15:00 | 252.84 | STOP_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2026-04-08 09:15:00 | 262.50 | 2026-04-09 09:15:00 | 288.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 287.70 | 2026-04-22 09:15:00 | 316.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:45:00 | 288.90 | 2026-04-22 09:15:00 | 317.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 298.81 | 2026-04-28 09:15:00 | 305.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-05-04 12:00:00 | 298.75 | 2026-05-05 09:15:00 | 302.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-04 12:30:00 | 298.75 | 2026-05-05 09:15:00 | 302.50 | STOP_HIT | 1.00 | -1.26% |
