# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 139.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 25 |
| ALERT3 | 136 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 62 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 64 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 40
- **Target hits / Stop hits / Partials:** 1 / 64 / 8
- **Avg / median % per leg:** 0.36% / -0.55%
- **Sum % (uncompounded):** 26.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 13 | 32.5% | 0 | 40 | 0 | -0.62% | -24.9% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.57% | 1.6% |
| BUY @ 3rd Alert (retest2) | 39 | 12 | 30.8% | 0 | 39 | 0 | -0.68% | -26.5% |
| SELL (all) | 33 | 20 | 60.6% | 1 | 24 | 8 | 1.56% | 51.5% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 1 | 1 | 2 | 6.49% | 25.9% |
| SELL @ 3rd Alert (retest2) | 29 | 16 | 55.2% | 0 | 23 | 6 | 0.88% | 25.6% |
| retest1 (combined) | 5 | 5 | 100.0% | 1 | 2 | 2 | 5.50% | 27.5% |
| retest2 (combined) | 68 | 28 | 41.2% | 0 | 62 | 6 | -0.01% | -0.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 114.25 | 110.79 | 110.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 114.52 | 111.53 | 110.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 112.49 | 112.57 | 111.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 114.40 | 112.57 | 111.74 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 116.20 | 117.07 | 116.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 116.20 | 117.07 | 116.30 | SL hit (close<ema400) qty=1.00 sl=116.30 alert=retest1 |

### Cycle 2 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 115.68 | 116.69 | 116.76 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 117.17 | 116.75 | 116.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 117.53 | 117.00 | 116.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 116.85 | 117.03 | 116.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 11:15:00 | 116.85 | 117.03 | 116.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 116.85 | 117.03 | 116.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:30:00 | 116.70 | 117.03 | 116.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 116.88 | 117.00 | 116.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 116.60 | 117.00 | 116.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 116.70 | 117.07 | 116.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 116.70 | 117.07 | 116.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 116.75 | 117.01 | 116.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 116.61 | 117.01 | 116.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 116.95 | 117.00 | 116.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 116.86 | 117.00 | 116.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 116.34 | 116.87 | 116.91 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 118.88 | 117.23 | 117.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 119.90 | 118.08 | 117.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 118.65 | 118.72 | 118.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:00:00 | 118.65 | 118.72 | 118.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 118.22 | 118.76 | 118.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 118.22 | 118.76 | 118.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 118.32 | 118.67 | 118.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 117.87 | 118.67 | 118.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 119.38 | 118.81 | 118.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 120.55 | 119.16 | 118.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 125.36 | 126.68 | 126.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 125.36 | 126.68 | 126.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 12:15:00 | 125.26 | 126.24 | 126.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 121.32 | 120.91 | 121.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 121.32 | 120.91 | 121.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 121.80 | 121.30 | 121.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 120.70 | 121.16 | 121.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 114.66 | 117.38 | 118.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 114.15 | 113.59 | 114.88 | SL hit (close>ema200) qty=0.50 sl=113.59 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 116.21 | 115.42 | 115.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 116.99 | 116.39 | 116.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 116.72 | 117.41 | 116.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 116.72 | 117.41 | 116.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 116.72 | 117.41 | 116.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 116.72 | 117.41 | 116.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 116.95 | 117.32 | 116.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 119.19 | 117.32 | 116.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 118.06 | 118.86 | 118.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 118.06 | 118.86 | 118.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 117.23 | 118.27 | 118.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 118.12 | 118.00 | 118.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 118.12 | 118.00 | 118.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 118.12 | 118.00 | 118.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 117.49 | 118.08 | 118.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 117.65 | 117.89 | 118.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 115.86 | 115.27 | 115.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 115.86 | 115.27 | 115.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 116.62 | 115.96 | 115.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 116.33 | 116.96 | 116.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 116.33 | 116.96 | 116.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 116.33 | 116.96 | 116.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 116.43 | 116.96 | 116.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 116.15 | 116.80 | 116.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 116.02 | 116.80 | 116.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 116.18 | 116.67 | 116.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 116.23 | 116.67 | 116.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 116.09 | 116.56 | 116.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 116.09 | 116.56 | 116.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 116.08 | 116.46 | 116.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 116.40 | 116.31 | 116.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 115.55 | 116.16 | 116.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 115.55 | 116.16 | 116.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 115.33 | 116.00 | 116.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 115.30 | 115.20 | 115.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 115.30 | 115.20 | 115.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 115.30 | 115.20 | 115.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 115.35 | 115.20 | 115.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 113.00 | 113.11 | 113.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:15:00 | 112.70 | 113.09 | 113.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 114.18 | 113.14 | 113.46 | SL hit (close>static) qty=1.00 sl=113.70 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 115.10 | 113.76 | 113.70 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 113.17 | 113.70 | 113.73 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 115.00 | 113.78 | 113.74 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 112.96 | 113.59 | 113.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 112.84 | 113.44 | 113.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 112.08 | 111.92 | 112.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 112.08 | 111.92 | 112.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 115.39 | 112.72 | 112.74 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 115.79 | 113.33 | 113.02 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 111.95 | 113.31 | 113.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 110.45 | 111.96 | 112.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 110.93 | 110.37 | 111.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 110.93 | 110.37 | 111.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 110.93 | 110.37 | 111.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 110.71 | 110.37 | 111.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 110.95 | 110.49 | 111.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 110.95 | 110.49 | 111.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 111.58 | 110.71 | 111.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 111.58 | 110.71 | 111.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 111.83 | 110.93 | 111.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 111.69 | 110.93 | 111.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 111.38 | 111.19 | 111.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 111.43 | 111.19 | 111.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 111.80 | 111.31 | 111.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 111.80 | 111.31 | 111.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 111.07 | 111.26 | 111.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:15:00 | 110.74 | 111.26 | 111.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 110.37 | 110.88 | 111.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 112.09 | 111.17 | 111.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 112.09 | 111.17 | 111.17 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 110.84 | 111.23 | 111.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 109.88 | 110.96 | 111.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 111.81 | 111.08 | 111.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 111.81 | 111.08 | 111.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 111.81 | 111.08 | 111.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 111.81 | 111.08 | 111.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 111.84 | 111.23 | 111.20 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 110.81 | 111.13 | 111.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 110.39 | 110.98 | 111.10 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 112.23 | 111.23 | 111.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 113.04 | 111.99 | 111.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 113.24 | 113.47 | 112.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 113.24 | 113.47 | 112.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 113.24 | 113.47 | 112.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 113.24 | 113.47 | 112.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 113.25 | 113.45 | 113.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 113.44 | 113.45 | 113.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 113.38 | 113.44 | 113.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 114.35 | 113.27 | 113.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 115.40 | 115.89 | 115.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 115.40 | 115.89 | 115.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 115.01 | 115.51 | 115.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 111.47 | 111.22 | 112.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 111.45 | 111.22 | 112.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 111.18 | 110.86 | 111.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 111.91 | 110.86 | 111.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 111.70 | 111.15 | 111.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 111.70 | 111.15 | 111.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 111.86 | 111.29 | 111.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 111.86 | 111.29 | 111.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 111.96 | 111.42 | 111.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 111.60 | 111.42 | 111.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 113.29 | 111.80 | 111.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 113.73 | 113.18 | 112.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 113.30 | 113.35 | 112.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:30:00 | 113.20 | 113.35 | 112.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 113.56 | 113.39 | 113.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 113.00 | 113.39 | 113.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 113.08 | 113.33 | 113.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 113.08 | 113.33 | 113.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 112.51 | 113.16 | 112.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 112.51 | 113.16 | 112.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 112.15 | 112.96 | 112.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 112.15 | 112.96 | 112.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 112.17 | 112.80 | 112.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 111.58 | 112.37 | 112.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 112.77 | 112.38 | 112.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 112.77 | 112.38 | 112.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 112.77 | 112.38 | 112.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 112.77 | 112.38 | 112.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 112.53 | 112.41 | 112.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 112.48 | 112.41 | 112.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 112.95 | 112.52 | 112.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 112.81 | 112.52 | 112.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 113.36 | 112.69 | 112.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 113.73 | 113.00 | 112.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 113.15 | 113.21 | 112.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 113.15 | 113.21 | 112.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 112.83 | 113.14 | 112.99 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 112.71 | 112.88 | 112.91 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 114.75 | 113.23 | 113.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 116.55 | 113.89 | 113.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 117.50 | 118.02 | 116.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:00:00 | 117.50 | 118.02 | 116.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 116.59 | 117.74 | 116.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 116.69 | 117.74 | 116.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 116.73 | 117.53 | 116.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 116.56 | 117.53 | 116.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 117.37 | 117.49 | 116.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 117.21 | 117.49 | 116.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 117.43 | 117.53 | 117.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 117.51 | 117.53 | 117.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 117.10 | 117.44 | 117.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 117.10 | 117.44 | 117.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 118.00 | 117.55 | 117.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 119.06 | 117.51 | 117.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:45:00 | 119.09 | 117.68 | 117.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 13:30:00 | 119.18 | 118.21 | 117.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 118.95 | 118.97 | 118.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 118.65 | 118.91 | 118.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 118.93 | 118.93 | 118.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 120.11 | 120.99 | 121.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 120.11 | 120.99 | 121.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 119.62 | 120.60 | 120.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 118.98 | 118.15 | 119.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 118.98 | 118.15 | 119.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 118.91 | 118.30 | 119.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 118.50 | 118.34 | 119.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 118.71 | 118.76 | 119.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 122.01 | 119.45 | 119.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 122.01 | 119.45 | 119.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 123.05 | 121.53 | 120.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 123.88 | 124.03 | 122.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 10:45:00 | 124.05 | 124.03 | 122.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 125.01 | 126.04 | 125.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 124.76 | 126.04 | 125.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 123.94 | 125.62 | 125.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 123.94 | 125.62 | 125.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 124.75 | 125.14 | 125.17 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 125.80 | 125.26 | 125.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 127.31 | 125.94 | 125.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 126.56 | 126.61 | 126.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 126.56 | 126.61 | 126.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 126.05 | 126.51 | 126.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 125.93 | 126.51 | 126.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 126.45 | 126.50 | 126.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 125.81 | 126.50 | 126.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 126.02 | 126.40 | 126.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 125.94 | 126.40 | 126.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 126.08 | 126.34 | 126.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:30:00 | 126.08 | 126.34 | 126.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 126.45 | 126.36 | 126.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 127.19 | 126.36 | 126.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 125.63 | 126.34 | 126.23 | SL hit (close<static) qty=1.00 sl=126.01 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 124.61 | 125.99 | 126.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 124.07 | 125.61 | 125.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 125.43 | 124.90 | 125.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 125.43 | 124.90 | 125.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 125.43 | 124.90 | 125.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 125.43 | 124.90 | 125.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 126.15 | 125.15 | 125.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 126.15 | 125.15 | 125.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 126.42 | 125.40 | 125.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 126.42 | 125.40 | 125.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 126.55 | 125.63 | 125.61 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 125.21 | 125.63 | 125.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 124.75 | 125.45 | 125.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 126.01 | 125.57 | 125.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 126.01 | 125.57 | 125.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 126.01 | 125.57 | 125.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 126.01 | 125.57 | 125.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 125.50 | 125.55 | 125.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:30:00 | 124.55 | 125.02 | 125.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 129.49 | 125.49 | 125.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 129.49 | 125.49 | 125.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 129.90 | 128.37 | 127.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 134.00 | 134.35 | 132.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:00:00 | 134.00 | 134.35 | 132.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 140.29 | 140.56 | 139.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:15:00 | 139.71 | 140.56 | 139.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 138.86 | 140.22 | 139.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 138.86 | 140.22 | 139.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 138.91 | 139.96 | 139.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 138.70 | 139.96 | 139.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 140.45 | 140.76 | 140.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 140.31 | 140.76 | 140.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 139.89 | 140.59 | 140.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 139.89 | 140.59 | 140.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 140.00 | 140.47 | 140.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 142.08 | 140.47 | 140.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 140.12 | 141.33 | 141.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 140.12 | 141.33 | 141.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 139.36 | 140.50 | 140.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 141.46 | 140.69 | 140.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 141.46 | 140.69 | 140.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 141.46 | 140.69 | 140.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 141.01 | 140.69 | 140.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 141.85 | 140.93 | 141.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 141.81 | 140.93 | 141.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 143.08 | 141.36 | 141.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 143.92 | 141.87 | 141.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 143.35 | 144.60 | 143.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 143.35 | 144.60 | 143.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 143.35 | 144.60 | 143.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 143.35 | 144.60 | 143.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 143.65 | 144.41 | 143.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 142.40 | 144.41 | 143.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 144.84 | 144.50 | 143.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:00:00 | 145.44 | 144.69 | 143.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 145.35 | 145.49 | 145.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 145.47 | 147.33 | 147.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 145.47 | 147.33 | 147.50 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 147.21 | 146.85 | 146.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 150.85 | 147.65 | 147.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 147.53 | 148.45 | 147.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 147.53 | 148.45 | 147.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 147.53 | 148.45 | 147.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 147.26 | 148.45 | 147.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 147.54 | 148.27 | 147.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 147.28 | 148.27 | 147.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 146.23 | 147.86 | 147.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 146.23 | 147.86 | 147.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 147.25 | 147.74 | 147.74 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 148.00 | 147.53 | 147.47 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 145.87 | 147.19 | 147.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 142.55 | 146.01 | 146.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 142.07 | 141.81 | 143.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:45:00 | 142.01 | 141.81 | 143.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 142.64 | 141.76 | 142.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 142.64 | 141.76 | 142.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 143.12 | 142.03 | 142.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:15:00 | 143.47 | 142.03 | 142.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 142.28 | 142.08 | 142.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 141.60 | 142.58 | 142.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 141.05 | 140.88 | 140.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 141.05 | 140.88 | 140.88 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 139.66 | 140.63 | 140.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 138.47 | 139.84 | 140.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 140.55 | 139.67 | 140.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 140.55 | 139.67 | 140.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 140.55 | 139.67 | 140.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 140.01 | 139.67 | 140.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 141.05 | 139.95 | 140.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 11:30:00 | 140.36 | 140.23 | 140.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 141.20 | 140.43 | 140.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 141.20 | 140.43 | 140.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 141.94 | 141.38 | 141.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 140.83 | 141.30 | 141.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 140.83 | 141.30 | 141.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 140.83 | 141.30 | 141.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 140.45 | 141.30 | 141.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 140.39 | 141.12 | 141.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 140.39 | 141.12 | 141.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 139.82 | 140.86 | 140.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 139.59 | 140.60 | 140.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 141.68 | 140.31 | 140.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 141.68 | 140.31 | 140.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 141.68 | 140.31 | 140.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 142.27 | 140.31 | 140.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 141.93 | 140.63 | 140.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 141.93 | 140.63 | 140.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 142.18 | 140.94 | 140.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 143.11 | 142.07 | 141.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 142.03 | 142.26 | 141.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 14:00:00 | 142.03 | 142.26 | 141.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 142.34 | 142.28 | 141.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 141.87 | 142.28 | 141.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 142.00 | 142.22 | 141.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 143.27 | 142.22 | 141.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 142.51 | 142.20 | 141.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 142.51 | 142.25 | 141.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 142.62 | 142.25 | 141.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 142.58 | 142.65 | 142.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 142.58 | 142.65 | 142.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 142.71 | 142.65 | 142.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:45:00 | 142.58 | 142.65 | 142.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 142.65 | 142.65 | 142.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 143.03 | 142.65 | 142.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 142.85 | 142.68 | 142.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 12:15:00 | 141.93 | 142.45 | 142.40 | SL hit (close<static) qty=1.00 sl=142.26 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 141.84 | 142.33 | 142.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 140.97 | 142.05 | 142.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 140.10 | 140.06 | 140.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:45:00 | 140.05 | 140.06 | 140.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 140.18 | 139.59 | 139.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 138.40 | 139.59 | 139.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 140.53 | 139.91 | 140.02 | SL hit (close>static) qty=1.00 sl=140.36 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 141.97 | 140.44 | 140.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 142.15 | 140.78 | 140.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 148.54 | 149.22 | 147.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 148.54 | 149.22 | 147.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 150.80 | 150.96 | 149.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 150.43 | 150.96 | 149.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 150.32 | 150.81 | 149.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 151.21 | 150.79 | 149.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 151.05 | 150.96 | 150.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 147.33 | 150.18 | 150.09 | SL hit (close<static) qty=1.00 sl=149.71 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 148.12 | 149.77 | 149.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 145.75 | 147.55 | 148.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 145.59 | 145.42 | 146.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 145.91 | 145.42 | 146.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 146.49 | 145.64 | 146.59 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 148.72 | 147.11 | 146.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 151.70 | 148.08 | 147.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 160.85 | 161.17 | 158.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 11:00:00 | 160.85 | 161.17 | 158.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 158.24 | 159.94 | 158.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 158.13 | 159.94 | 158.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 158.49 | 159.65 | 158.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:45:00 | 157.25 | 159.65 | 158.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 156.65 | 159.03 | 158.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 156.65 | 159.03 | 158.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 157.37 | 158.70 | 158.59 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 15:15:00 | 157.45 | 158.45 | 158.49 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 165.10 | 159.78 | 159.09 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 159.64 | 161.62 | 161.63 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 161.90 | 161.68 | 161.65 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 161.09 | 161.56 | 161.60 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 162.18 | 161.69 | 161.65 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 12:15:00 | 160.60 | 161.47 | 161.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 160.24 | 161.22 | 161.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 162.98 | 161.57 | 161.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 162.98 | 161.57 | 161.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 163.45 | 161.95 | 161.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 164.33 | 162.54 | 162.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 164.84 | 166.03 | 164.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 164.84 | 166.03 | 164.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 163.60 | 165.54 | 164.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 163.60 | 165.54 | 164.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 164.91 | 165.42 | 164.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:45:00 | 165.76 | 165.33 | 164.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 165.84 | 165.43 | 164.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 159.78 | 163.83 | 164.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 159.78 | 163.83 | 164.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 154.52 | 161.38 | 163.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 152.00 | 151.68 | 155.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 156.65 | 151.68 | 155.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 156.65 | 152.67 | 155.23 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 157.70 | 156.33 | 156.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 158.56 | 156.78 | 156.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 12:15:00 | 161.70 | 162.24 | 161.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 13:00:00 | 161.70 | 162.24 | 161.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 163.93 | 166.60 | 165.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 163.45 | 166.60 | 165.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 165.87 | 166.46 | 165.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 166.35 | 166.32 | 165.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 163.74 | 165.68 | 165.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 163.74 | 165.68 | 165.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 161.79 | 164.66 | 165.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 163.91 | 162.94 | 163.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 163.91 | 162.94 | 163.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 164.93 | 163.34 | 163.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:00:00 | 164.93 | 163.34 | 163.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 165.71 | 163.81 | 164.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 165.71 | 163.81 | 164.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 165.71 | 164.46 | 164.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 166.91 | 164.95 | 164.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 171.68 | 171.81 | 170.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 171.68 | 171.81 | 170.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 169.08 | 171.16 | 170.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 169.08 | 171.16 | 170.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 168.89 | 170.71 | 170.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 170.08 | 170.71 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 170.64 | 170.69 | 170.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 171.80 | 170.69 | 170.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 171.50 | 171.00 | 170.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:45:00 | 171.59 | 171.23 | 170.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 174.23 | 171.18 | 170.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 174.15 | 174.79 | 173.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 174.15 | 174.79 | 173.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 174.91 | 174.82 | 174.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 175.27 | 174.82 | 174.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 175.50 | 174.81 | 174.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 176.71 | 174.89 | 174.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 175.36 | 175.11 | 174.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 176.16 | 176.48 | 175.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 176.16 | 176.48 | 175.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 176.02 | 176.39 | 175.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 173.23 | 176.39 | 175.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 172.84 | 175.68 | 175.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 172.84 | 175.68 | 175.56 | SL hit (close<static) qty=1.00 sl=173.72 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 173.11 | 175.17 | 175.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 163.94 | 171.43 | 173.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 165.57 | 165.28 | 168.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:30:00 | 164.46 | 164.98 | 167.77 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 15:00:00 | 163.43 | 164.11 | 166.64 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 156.24 | 159.39 | 162.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 155.26 | 159.39 | 162.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-03-09 10:15:00 | 148.01 | 157.50 | 161.57 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 65 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 152.89 | 151.53 | 151.34 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 148.15 | 151.34 | 151.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 145.85 | 148.99 | 150.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 149.59 | 148.28 | 149.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 151.22 | 148.28 | 149.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 151.66 | 148.96 | 149.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 151.44 | 148.96 | 149.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 150.53 | 149.27 | 149.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 149.85 | 149.27 | 149.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 150.99 | 150.04 | 150.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 150.99 | 150.04 | 150.04 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 144.43 | 149.02 | 149.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 143.81 | 147.18 | 148.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 145.75 | 145.31 | 146.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 145.11 | 145.37 | 146.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 147.68 | 145.91 | 146.84 | SL hit (close>static) qty=1.00 sl=147.60 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 150.15 | 147.59 | 147.34 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 143.73 | 147.37 | 147.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 141.73 | 144.82 | 146.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 140.51 | 139.97 | 142.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 140.31 | 139.97 | 142.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 140.34 | 140.50 | 142.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 143.40 | 141.08 | 142.42 | SL hit (close>static) qty=1.00 sl=142.81 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 142.75 | 140.95 | 140.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 143.02 | 141.37 | 140.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 138.08 | 141.00 | 140.88 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 137.54 | 140.31 | 140.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 11:15:00 | 137.24 | 139.70 | 140.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 145.86 | 139.80 | 139.91 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 147.25 | 141.29 | 140.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 147.99 | 145.45 | 144.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 142.17 | 145.97 | 145.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 144.62 | 145.03 | 144.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 144.50 | 144.84 | 144.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 144.50 | 144.84 | 144.86 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 145.35 | 144.96 | 144.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 148.02 | 145.57 | 145.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 148.41 | 148.54 | 147.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 148.41 | 148.54 | 147.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 149.20 | 148.80 | 147.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:30:00 | 149.89 | 148.57 | 148.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 149.77 | 148.79 | 148.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:00:00 | 150.72 | 151.85 | 151.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 148.38 | 150.77 | 150.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 148.38 | 150.77 | 150.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 146.76 | 149.97 | 150.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 148.22 | 147.93 | 148.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 148.22 | 147.93 | 148.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 145.55 | 145.06 | 146.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 144.41 | 145.50 | 146.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 137.19 | 138.98 | 140.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 141.06 | 139.19 | 140.00 | SL hit (close>ema200) qty=0.50 sl=139.19 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 142.35 | 140.37 | 140.25 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 139.95 | 140.32 | 140.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 139.17 | 140.09 | 140.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 15:15:00 | 139.85 | 139.76 | 140.01 | EMA200 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:15:00 | 114.40 | 2025-05-20 13:15:00 | 116.20 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-05-21 10:15:00 | 117.96 | 2025-05-22 13:15:00 | 115.68 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-05-21 13:00:00 | 118.00 | 2025-05-22 13:15:00 | 115.68 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-21 13:45:00 | 117.91 | 2025-05-22 13:15:00 | 115.68 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-05-30 13:00:00 | 120.55 | 2025-06-11 10:15:00 | 125.36 | STOP_HIT | 1.00 | 3.99% |
| SELL | retest2 | 2025-06-17 11:45:00 | 120.70 | 2025-06-19 12:15:00 | 114.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:45:00 | 120.70 | 2025-06-23 13:15:00 | 114.15 | STOP_HIT | 0.50 | 5.43% |
| BUY | retest2 | 2025-06-30 09:15:00 | 119.19 | 2025-07-03 12:15:00 | 118.06 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-07-07 15:15:00 | 117.49 | 2025-07-15 10:15:00 | 115.86 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-07-08 09:30:00 | 117.65 | 2025-07-15 10:15:00 | 115.86 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2025-07-18 09:15:00 | 116.40 | 2025-07-18 09:15:00 | 115.55 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-24 11:15:00 | 112.70 | 2025-07-24 13:15:00 | 114.18 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-05 13:15:00 | 110.74 | 2025-08-06 12:15:00 | 112.09 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-06 10:00:00 | 110.37 | 2025-08-06 12:15:00 | 112.09 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-08-18 09:15:00 | 114.35 | 2025-08-22 15:15:00 | 115.40 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2025-09-17 10:15:00 | 119.06 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-09-17 10:45:00 | 119.09 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-09-17 13:30:00 | 119.18 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-09-18 12:45:00 | 118.95 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-09-18 14:45:00 | 118.93 | 2025-09-25 11:15:00 | 120.11 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-09-29 11:30:00 | 118.50 | 2025-09-30 09:15:00 | 122.01 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-09-29 15:00:00 | 118.71 | 2025-09-30 09:15:00 | 122.01 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-10-13 14:15:00 | 127.19 | 2025-10-14 09:15:00 | 125.63 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-17 12:30:00 | 124.55 | 2025-10-20 10:15:00 | 129.49 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2025-11-03 09:15:00 | 142.08 | 2025-11-06 11:15:00 | 140.12 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-11 13:00:00 | 145.44 | 2025-11-21 10:15:00 | 145.47 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-11-14 09:15:00 | 145.35 | 2025-11-21 10:15:00 | 145.47 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-12-08 09:15:00 | 141.60 | 2025-12-10 10:15:00 | 141.05 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-12-11 11:30:00 | 140.36 | 2025-12-11 12:15:00 | 141.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-19 09:15:00 | 143.27 | 2025-12-23 12:15:00 | 141.93 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-19 10:30:00 | 142.51 | 2025-12-23 12:15:00 | 141.93 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-19 12:30:00 | 142.51 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-19 13:15:00 | 142.62 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-12-23 09:15:00 | 143.03 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-23 09:45:00 | 142.85 | 2025-12-23 13:15:00 | 141.84 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-30 09:15:00 | 138.40 | 2025-12-30 11:15:00 | 140.53 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-07 11:15:00 | 151.21 | 2026-01-08 10:15:00 | 147.33 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-07 14:00:00 | 151.05 | 2026-01-08 10:15:00 | 147.33 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-01-30 09:45:00 | 165.76 | 2026-02-01 09:15:00 | 159.78 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2026-01-30 10:30:00 | 165.84 | 2026-02-01 09:15:00 | 159.78 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2026-02-11 13:00:00 | 166.35 | 2026-02-12 11:15:00 | 163.74 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-02-20 11:15:00 | 171.80 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2026-02-20 13:15:00 | 171.50 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2026-02-20 14:45:00 | 171.59 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2026-02-23 09:15:00 | 174.23 | 2026-03-02 09:15:00 | 172.84 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-25 12:15:00 | 175.27 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-02-25 15:00:00 | 175.50 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-26 09:15:00 | 176.71 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-26 10:30:00 | 175.36 | 2026-03-02 10:15:00 | 173.11 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest1 | 2026-03-05 11:30:00 | 164.46 | 2026-03-09 09:15:00 | 156.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 15:00:00 | 163.43 | 2026-03-09 09:15:00 | 155.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 11:30:00 | 164.46 | 2026-03-09 10:15:00 | 148.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-03-05 15:00:00 | 163.43 | 2026-03-10 10:15:00 | 153.71 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2026-03-11 10:15:00 | 155.14 | 2026-03-16 10:15:00 | 147.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 155.00 | 2026-03-16 10:15:00 | 147.58 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2026-03-11 13:30:00 | 155.35 | 2026-03-16 10:15:00 | 147.37 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2026-03-11 14:30:00 | 155.13 | 2026-03-16 12:15:00 | 147.25 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-03-11 10:15:00 | 155.14 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-11 11:30:00 | 155.00 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2026-03-11 13:30:00 | 155.35 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2026-03-11 14:30:00 | 155.13 | 2026-03-16 14:15:00 | 150.73 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-13 09:15:00 | 153.54 | 2026-03-18 11:15:00 | 152.89 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 149.85 | 2026-03-20 13:15:00 | 150.99 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-24 10:30:00 | 145.11 | 2026-03-24 12:15:00 | 147.68 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-04-01 10:15:00 | 140.31 | 2026-04-01 12:15:00 | 143.40 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-04-01 11:45:00 | 140.34 | 2026-04-01 12:15:00 | 143.40 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-02 09:15:00 | 136.65 | 2026-04-06 13:15:00 | 142.75 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-04-13 12:15:00 | 144.62 | 2026-04-13 13:15:00 | 144.50 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-20 10:30:00 | 149.89 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-21 09:15:00 | 149.77 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-04-23 13:00:00 | 150.72 | 2026-04-24 09:15:00 | 148.38 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-29 14:15:00 | 144.41 | 2026-05-05 11:15:00 | 137.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 14:15:00 | 144.41 | 2026-05-06 09:15:00 | 141.06 | STOP_HIT | 0.50 | 2.32% |
