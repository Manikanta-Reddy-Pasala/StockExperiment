# Aditya Birla Fashion and Retail Ltd. (ABFRL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 66.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 55 |
| ALERT2 | 55 |
| ALERT2_SKIP | 26 |
| ALERT3 | 152 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 63 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 45
- **Target hits / Stop hits / Partials:** 6 / 57 / 8
- **Avg / median % per leg:** 0.94% / -0.95%
- **Sum % (uncompounded):** 66.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 6 | 20.0% | 6 | 24 | 0 | 0.82% | 24.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 6 | 20.0% | 6 | 24 | 0 | 0.82% | 24.6% |
| SELL (all) | 41 | 20 | 48.8% | 0 | 33 | 8 | 1.03% | 42.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 20 | 48.8% | 0 | 33 | 8 | 1.03% | 42.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 26 | 36.6% | 6 | 57 | 8 | 0.94% | 66.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 266.05 | 260.37 | 260.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 268.60 | 263.81 | 261.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 268.50 | 268.51 | 265.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 268.50 | 268.51 | 265.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 290.85 | 280.98 | 277.95 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 273.60 | 280.53 | 280.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 271.00 | 277.55 | 279.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 89.10 | 89.10 | 109.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:00:00 | 89.10 | 89.10 | 109.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 87.50 | 87.18 | 87.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 87.72 | 87.18 | 87.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 77.87 | 77.15 | 77.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 77.87 | 77.15 | 77.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 77.69 | 77.26 | 77.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 77.42 | 77.29 | 77.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 73.55 | 75.71 | 76.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 74.41 | 74.34 | 75.31 | SL hit (close>ema200) qty=0.50 sl=74.34 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 74.65 | 73.95 | 73.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 74.80 | 74.12 | 73.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 74.45 | 74.53 | 74.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 11:00:00 | 74.45 | 74.53 | 74.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 74.84 | 74.62 | 74.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:45:00 | 74.44 | 74.62 | 74.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 74.36 | 74.56 | 74.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 74.36 | 74.56 | 74.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 74.07 | 74.47 | 74.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 74.07 | 74.47 | 74.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 73.95 | 74.36 | 74.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 74.39 | 74.36 | 74.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 73.81 | 74.33 | 74.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 73.81 | 74.33 | 74.37 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 74.83 | 74.34 | 74.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 75.38 | 74.78 | 74.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 74.61 | 74.82 | 74.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 74.61 | 74.82 | 74.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 74.61 | 74.82 | 74.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 74.61 | 74.82 | 74.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 74.54 | 74.76 | 74.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 74.43 | 74.76 | 74.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 75.02 | 74.81 | 74.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 75.01 | 74.81 | 74.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 73.80 | 74.64 | 74.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 73.80 | 74.64 | 74.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 74.45 | 74.61 | 74.61 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 75.08 | 74.71 | 74.66 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 74.45 | 74.61 | 74.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 74.19 | 74.51 | 74.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 74.66 | 74.54 | 74.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 74.66 | 74.54 | 74.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 74.66 | 74.54 | 74.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 74.66 | 74.54 | 74.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 13:15:00 | 76.31 | 74.89 | 74.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 78.08 | 75.53 | 75.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 76.64 | 76.81 | 76.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:30:00 | 76.68 | 76.81 | 76.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 77.67 | 76.93 | 76.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:15:00 | 78.38 | 76.93 | 76.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:45:00 | 77.84 | 77.54 | 76.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 78.07 | 77.59 | 76.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 77.97 | 77.75 | 76.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 77.61 | 77.92 | 77.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 77.61 | 77.92 | 77.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 78.86 | 78.10 | 77.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 77.75 | 78.10 | 77.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 78.41 | 78.74 | 78.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 78.41 | 78.74 | 78.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 78.96 | 78.79 | 78.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 77.70 | 78.25 | 78.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 77.70 | 78.25 | 78.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 76.59 | 77.77 | 78.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 77.04 | 76.99 | 77.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 77.04 | 76.99 | 77.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 77.23 | 77.04 | 77.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 77.35 | 77.04 | 77.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 76.70 | 76.97 | 77.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 76.56 | 76.89 | 77.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 77.39 | 77.02 | 77.13 | SL hit (close>static) qty=1.00 sl=77.31 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 75.65 | 75.44 | 75.43 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 74.94 | 75.34 | 75.39 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 75.70 | 75.39 | 75.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 75.92 | 75.50 | 75.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 75.37 | 75.52 | 75.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 75.37 | 75.52 | 75.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 75.37 | 75.52 | 75.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 75.37 | 75.52 | 75.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 75.17 | 75.45 | 75.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 75.17 | 75.45 | 75.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 75.16 | 75.39 | 75.40 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 75.80 | 75.47 | 75.43 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 74.97 | 75.34 | 75.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 74.67 | 75.20 | 75.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 74.50 | 74.36 | 74.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 74.37 | 74.36 | 74.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 74.62 | 73.93 | 74.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 74.62 | 73.93 | 74.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 74.09 | 73.96 | 74.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:15:00 | 74.63 | 73.96 | 74.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 74.55 | 74.08 | 74.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 74.54 | 74.08 | 74.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 75.66 | 74.39 | 74.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 75.66 | 74.39 | 74.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 75.27 | 74.57 | 74.50 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 73.68 | 74.54 | 74.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 12:15:00 | 73.58 | 74.13 | 74.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 13:15:00 | 74.16 | 73.59 | 73.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 74.16 | 73.59 | 73.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 74.16 | 73.59 | 73.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 74.16 | 73.59 | 73.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 73.45 | 73.56 | 73.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:30:00 | 74.06 | 73.56 | 73.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 73.54 | 73.47 | 73.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:15:00 | 74.00 | 73.47 | 73.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 74.05 | 73.59 | 73.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:30:00 | 74.07 | 73.59 | 73.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 74.69 | 73.81 | 73.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 74.69 | 73.81 | 73.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 75.02 | 74.05 | 73.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 75.55 | 74.35 | 74.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 75.29 | 75.42 | 74.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:00:00 | 75.29 | 75.42 | 74.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 75.32 | 75.40 | 74.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 75.32 | 75.40 | 74.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 75.93 | 75.51 | 75.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:45:00 | 77.01 | 75.91 | 75.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 76.24 | 75.86 | 75.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 75.02 | 75.76 | 75.64 | SL hit (close<static) qty=1.00 sl=75.04 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 75.23 | 75.55 | 75.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 74.77 | 75.35 | 75.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 74.71 | 74.71 | 74.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 14:45:00 | 74.77 | 74.71 | 74.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 74.70 | 74.66 | 74.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:15:00 | 74.36 | 74.57 | 74.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:45:00 | 74.25 | 74.47 | 74.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 76.10 | 74.90 | 74.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 76.10 | 74.90 | 74.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 77.61 | 75.65 | 75.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 76.90 | 77.06 | 76.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 76.90 | 77.06 | 76.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 76.90 | 77.06 | 76.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 76.60 | 77.06 | 76.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 76.62 | 76.95 | 76.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 76.57 | 76.95 | 76.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 76.70 | 76.90 | 76.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:15:00 | 76.46 | 76.90 | 76.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 76.66 | 76.85 | 76.64 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 75.90 | 76.49 | 76.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 75.62 | 76.16 | 76.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 15:15:00 | 76.10 | 75.88 | 76.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 76.81 | 75.88 | 76.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 23 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 79.94 | 76.70 | 76.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 11:15:00 | 81.18 | 77.97 | 77.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 82.36 | 82.40 | 81.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 81.06 | 82.40 | 81.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 81.36 | 82.19 | 81.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 81.69 | 82.19 | 81.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 82.08 | 82.17 | 81.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 81.19 | 82.17 | 81.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 79.71 | 81.68 | 81.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 79.71 | 81.68 | 81.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 79.85 | 81.31 | 81.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:15:00 | 79.48 | 81.31 | 81.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 78.86 | 80.82 | 80.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 78.01 | 80.26 | 80.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 77.97 | 77.95 | 78.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 77.97 | 77.95 | 78.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 81.12 | 78.59 | 79.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 81.12 | 78.59 | 79.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 81.24 | 79.12 | 79.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 81.44 | 79.12 | 79.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 83.37 | 79.97 | 79.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 83.58 | 80.69 | 79.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 83.00 | 83.11 | 81.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 82.12 | 83.11 | 81.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 82.37 | 82.96 | 82.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 84.36 | 81.82 | 81.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:30:00 | 83.28 | 83.17 | 82.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 83.38 | 83.24 | 82.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 83.29 | 83.25 | 82.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-08 12:15:00 | 92.80 | 87.16 | 85.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 87.57 | 88.56 | 88.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 86.79 | 88.20 | 88.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 88.28 | 87.95 | 88.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 88.28 | 87.95 | 88.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 88.28 | 87.95 | 88.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 88.26 | 87.95 | 88.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 88.45 | 88.05 | 88.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 87.94 | 88.05 | 88.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 87.82 | 88.00 | 88.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 87.80 | 87.99 | 88.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 87.99 | 87.74 | 87.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 87.44 | 87.68 | 87.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 87.28 | 87.68 | 87.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 88.85 | 87.41 | 87.62 | SL hit (close>static) qty=1.00 sl=88.73 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 89.80 | 87.89 | 87.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 90.71 | 88.45 | 88.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 15:15:00 | 89.74 | 89.88 | 89.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:15:00 | 89.96 | 89.88 | 89.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 89.58 | 89.82 | 89.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 89.58 | 89.82 | 89.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 91.82 | 90.22 | 89.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 93.64 | 91.34 | 90.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 93.20 | 91.82 | 90.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 15:00:00 | 93.52 | 93.01 | 91.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 91.22 | 91.66 | 91.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 91.22 | 91.66 | 91.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 89.91 | 91.07 | 91.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 85.27 | 84.82 | 86.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 85.17 | 84.82 | 86.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 85.38 | 84.80 | 85.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 85.03 | 84.80 | 85.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 85.48 | 84.93 | 85.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 85.48 | 84.93 | 85.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 85.00 | 84.95 | 85.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 84.99 | 85.09 | 85.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 86.26 | 85.33 | 85.43 | SL hit (close>static) qty=1.00 sl=85.55 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 87.00 | 85.66 | 85.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 87.57 | 86.04 | 85.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 87.50 | 87.60 | 86.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 87.75 | 87.60 | 86.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 86.91 | 87.49 | 86.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 86.97 | 87.49 | 86.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 87.00 | 87.39 | 86.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 86.91 | 87.39 | 86.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 86.90 | 87.30 | 86.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 86.97 | 87.30 | 86.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 86.60 | 87.16 | 86.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 86.60 | 87.16 | 86.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 86.23 | 86.97 | 86.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 86.30 | 86.97 | 86.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 86.06 | 86.79 | 86.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 85.80 | 86.79 | 86.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 86.02 | 86.64 | 86.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 85.25 | 85.88 | 86.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 86.22 | 85.94 | 86.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 86.22 | 85.94 | 86.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 85.75 | 85.91 | 86.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 85.37 | 85.91 | 86.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 81.10 | 81.90 | 82.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 81.32 | 81.09 | 82.04 | SL hit (close>ema200) qty=0.50 sl=81.09 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 83.35 | 82.52 | 82.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 83.75 | 83.16 | 82.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 82.82 | 83.09 | 82.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 82.82 | 83.09 | 82.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 82.82 | 83.09 | 82.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 83.06 | 83.09 | 82.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 82.75 | 83.02 | 82.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 82.76 | 83.02 | 82.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 82.69 | 82.96 | 82.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 82.52 | 82.96 | 82.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 82.10 | 82.79 | 82.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 82.10 | 82.79 | 82.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 81.95 | 82.62 | 82.69 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 82.98 | 82.70 | 82.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 83.68 | 83.00 | 82.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 83.58 | 83.67 | 83.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:45:00 | 83.43 | 83.67 | 83.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 83.32 | 83.55 | 83.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 83.19 | 83.55 | 83.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 83.00 | 83.44 | 83.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 83.11 | 83.44 | 83.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 82.85 | 83.32 | 83.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 82.85 | 83.32 | 83.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 81.91 | 83.04 | 83.12 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 84.33 | 82.84 | 82.65 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 82.85 | 83.25 | 83.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 82.60 | 83.12 | 83.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 84.15 | 82.52 | 82.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 84.15 | 82.52 | 82.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 84.15 | 82.52 | 82.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 84.15 | 82.52 | 82.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 83.23 | 82.66 | 82.75 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 83.65 | 82.86 | 82.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 84.36 | 83.16 | 82.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 84.03 | 84.36 | 83.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 14:15:00 | 84.03 | 84.36 | 83.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 84.03 | 84.36 | 83.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 84.03 | 84.36 | 83.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 84.05 | 84.30 | 83.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 80.60 | 84.30 | 83.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 79.06 | 83.25 | 83.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 78.00 | 80.94 | 82.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 79.25 | 78.44 | 79.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 79.25 | 78.44 | 79.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 79.25 | 78.44 | 79.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 79.73 | 78.44 | 79.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 79.76 | 78.71 | 79.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 79.76 | 78.71 | 79.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 79.64 | 78.89 | 79.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 79.22 | 78.89 | 79.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 79.39 | 79.28 | 79.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 80.20 | 79.46 | 79.81 | SL hit (close>static) qty=1.00 sl=79.98 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 81.17 | 79.80 | 79.62 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 79.50 | 79.79 | 79.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 78.88 | 79.61 | 79.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 78.54 | 78.51 | 78.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:45:00 | 78.46 | 78.51 | 78.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 78.04 | 78.03 | 78.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:30:00 | 77.55 | 78.00 | 78.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 77.70 | 77.96 | 78.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 78.77 | 78.08 | 78.34 | SL hit (close>static) qty=1.00 sl=78.50 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 76.94 | 76.47 | 76.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 77.15 | 76.82 | 76.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 77.85 | 77.88 | 77.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 77.76 | 77.88 | 77.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 77.49 | 77.80 | 77.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 77.49 | 77.80 | 77.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 76.87 | 77.61 | 77.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 76.87 | 77.61 | 77.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 76.89 | 77.47 | 77.38 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 76.51 | 77.20 | 77.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 76.39 | 76.91 | 77.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 76.85 | 76.77 | 76.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:30:00 | 76.59 | 76.77 | 76.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 76.75 | 76.78 | 76.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:15:00 | 76.94 | 76.78 | 76.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 77.07 | 76.84 | 76.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 77.07 | 76.84 | 76.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 76.50 | 76.77 | 76.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:00:00 | 76.32 | 76.68 | 76.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 76.41 | 76.55 | 76.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 77.30 | 76.82 | 76.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 77.30 | 76.82 | 76.79 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 76.06 | 76.71 | 76.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 75.78 | 76.52 | 76.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 75.62 | 75.54 | 76.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:30:00 | 75.28 | 75.54 | 76.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 76.00 | 75.63 | 75.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 76.00 | 75.63 | 75.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 76.00 | 75.71 | 75.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 76.66 | 75.71 | 75.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 77.20 | 76.00 | 76.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 77.54 | 76.00 | 76.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 76.26 | 76.06 | 76.07 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 76.51 | 76.15 | 76.11 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 75.72 | 76.05 | 76.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 75.59 | 75.96 | 76.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 75.57 | 75.36 | 75.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 14:15:00 | 75.57 | 75.36 | 75.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 75.57 | 75.36 | 75.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 75.57 | 75.36 | 75.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 75.45 | 75.38 | 75.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 76.23 | 75.38 | 75.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 76.64 | 75.63 | 75.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 76.64 | 75.63 | 75.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 76.60 | 75.83 | 75.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 77.41 | 76.27 | 75.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 76.34 | 76.83 | 76.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 76.34 | 76.83 | 76.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 76.34 | 76.83 | 76.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 76.34 | 76.83 | 76.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 76.06 | 76.67 | 76.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 76.06 | 76.67 | 76.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 75.38 | 76.22 | 76.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 15:15:00 | 75.32 | 76.04 | 76.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 76.52 | 74.96 | 75.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 76.52 | 74.96 | 75.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 76.52 | 74.96 | 75.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 76.52 | 74.96 | 75.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 77.42 | 75.46 | 75.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 77.87 | 75.46 | 75.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 77.67 | 75.90 | 75.72 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 75.43 | 75.80 | 75.83 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 76.09 | 75.79 | 75.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 76.32 | 75.92 | 75.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 77.75 | 78.16 | 77.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 77.75 | 78.16 | 77.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 78.91 | 78.31 | 77.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 78.26 | 78.31 | 77.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 77.87 | 78.28 | 77.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 77.87 | 78.28 | 77.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 77.71 | 78.17 | 77.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 77.65 | 78.17 | 77.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 77.75 | 78.08 | 77.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:15:00 | 77.36 | 78.08 | 77.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 77.30 | 77.93 | 77.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 77.30 | 77.93 | 77.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 77.01 | 77.60 | 77.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 76.77 | 77.44 | 77.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 77.28 | 76.95 | 77.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 77.28 | 76.95 | 77.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 77.28 | 76.95 | 77.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 77.28 | 76.95 | 77.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 77.11 | 76.98 | 77.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 77.14 | 76.98 | 77.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 76.80 | 76.95 | 77.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 76.49 | 76.95 | 77.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 76.26 | 76.78 | 77.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 76.61 | 76.33 | 76.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 77.22 | 76.51 | 76.64 | SL hit (close>static) qty=1.00 sl=77.20 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 77.20 | 76.80 | 76.76 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 76.45 | 76.73 | 76.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 75.97 | 76.56 | 76.65 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 77.72 | 76.67 | 76.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 78.23 | 77.45 | 77.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 77.31 | 77.48 | 77.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 77.31 | 77.48 | 77.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 77.24 | 77.43 | 77.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 77.05 | 77.43 | 77.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 77.20 | 77.38 | 77.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 77.50 | 77.38 | 77.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:45:00 | 77.32 | 77.20 | 77.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 76.17 | 76.97 | 77.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 76.17 | 76.97 | 77.03 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 77.50 | 77.06 | 77.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 77.86 | 77.29 | 77.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 77.26 | 77.33 | 77.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 11:15:00 | 77.26 | 77.33 | 77.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 77.26 | 77.33 | 77.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 77.26 | 77.33 | 77.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 77.25 | 77.32 | 77.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 76.81 | 77.32 | 77.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 76.85 | 77.22 | 77.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:15:00 | 76.80 | 77.22 | 77.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 77.21 | 77.22 | 77.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 77.06 | 77.22 | 77.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 77.48 | 77.27 | 77.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 77.15 | 77.27 | 77.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 78.27 | 77.47 | 77.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 77.65 | 77.47 | 77.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 77.11 | 77.40 | 77.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 77.11 | 77.40 | 77.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 76.22 | 77.16 | 77.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 75.90 | 76.91 | 77.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 76.34 | 76.15 | 76.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 76.34 | 76.15 | 76.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 76.34 | 76.15 | 76.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 76.65 | 76.15 | 76.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 75.55 | 76.03 | 76.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 75.43 | 75.88 | 76.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 71.66 | 72.16 | 72.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 71.43 | 70.74 | 71.65 | SL hit (close>ema200) qty=0.50 sl=70.74 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 64.72 | 63.31 | 63.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 65.26 | 63.70 | 63.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 63.73 | 63.97 | 63.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 63.73 | 63.97 | 63.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 63.73 | 63.97 | 63.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 63.74 | 63.97 | 63.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 63.64 | 63.90 | 63.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:30:00 | 64.02 | 63.89 | 63.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 64.29 | 64.03 | 63.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 65.19 | 64.55 | 64.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 63.94 | 65.15 | 65.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 63.94 | 65.15 | 65.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 63.19 | 64.76 | 65.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 64.12 | 64.08 | 64.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 64.12 | 64.08 | 64.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 66.91 | 64.63 | 64.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 66.71 | 64.63 | 64.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 66.45 | 65.00 | 64.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 66.45 | 65.00 | 64.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 67.77 | 65.94 | 65.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 66.80 | 69.39 | 68.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 66.80 | 69.39 | 68.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 66.80 | 69.39 | 68.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 66.80 | 69.39 | 68.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 67.25 | 68.96 | 68.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 66.80 | 68.96 | 68.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 69.89 | 68.90 | 68.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:30:00 | 70.90 | 69.28 | 68.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 71.99 | 70.15 | 69.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 70.07 | 71.41 | 71.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 70.07 | 71.41 | 71.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 69.98 | 70.40 | 70.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 74.20 | 70.80 | 70.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 74.20 | 70.80 | 70.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 74.20 | 70.80 | 70.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:45:00 | 74.15 | 70.80 | 70.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 72.44 | 71.13 | 70.98 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 71.65 | 72.08 | 72.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 71.25 | 71.77 | 71.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 70.74 | 70.14 | 70.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 70.74 | 70.14 | 70.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 70.74 | 70.14 | 70.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 69.10 | 69.79 | 70.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 69.14 | 69.79 | 70.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 69.26 | 69.67 | 70.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 68.82 | 69.15 | 69.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 69.66 | 69.05 | 69.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 69.85 | 69.05 | 69.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 69.12 | 69.06 | 69.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 68.72 | 68.92 | 69.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.64 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.68 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.80 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.38 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.28 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 66.20 | 66.09 | 66.91 | SL hit (close>ema200) qty=0.50 sl=66.09 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 15:15:00 | 62.15 | 62.00 | 62.00 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 61.07 | 61.82 | 61.92 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 62.80 | 62.12 | 62.03 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 60.10 | 61.73 | 61.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 59.87 | 60.91 | 61.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 58.63 | 58.54 | 59.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 58.63 | 58.54 | 59.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 59.39 | 58.64 | 59.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 59.61 | 58.64 | 59.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 58.86 | 58.68 | 59.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 58.27 | 59.19 | 59.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 62.34 | 58.64 | 58.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 62.34 | 58.64 | 58.61 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 56.34 | 58.59 | 58.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 55.92 | 58.06 | 58.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 57.79 | 57.29 | 57.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 57.79 | 57.29 | 57.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 57.79 | 57.29 | 57.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 57.77 | 57.29 | 57.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 57.87 | 57.41 | 57.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:00:00 | 57.87 | 57.41 | 57.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 57.85 | 57.50 | 57.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 57.85 | 57.50 | 57.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 58.95 | 57.79 | 57.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 58.97 | 57.79 | 57.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 58.81 | 57.99 | 58.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:15:00 | 58.42 | 57.99 | 58.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 58.20 | 58.07 | 58.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 58.20 | 58.07 | 58.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 60.22 | 58.50 | 58.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 59.04 | 59.19 | 58.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 59.04 | 59.19 | 58.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 58.78 | 59.11 | 58.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 58.21 | 59.11 | 58.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 57.58 | 58.80 | 58.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 57.58 | 58.80 | 58.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 57.05 | 58.45 | 58.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 56.20 | 57.36 | 57.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 57.49 | 55.71 | 56.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 57.49 | 55.71 | 56.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 57.49 | 55.71 | 56.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 57.49 | 55.71 | 56.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 57.28 | 56.02 | 56.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 57.55 | 56.02 | 56.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 58.03 | 57.18 | 57.07 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 55.66 | 56.93 | 56.98 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 57.71 | 56.94 | 56.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 59.07 | 57.70 | 57.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 57.91 | 58.26 | 57.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 57.91 | 58.26 | 57.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 58.16 | 58.24 | 57.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 59.20 | 57.99 | 57.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 58.35 | 58.82 | 58.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 65.12 | 61.42 | 60.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 64.00 | 64.59 | 64.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 63.63 | 64.30 | 64.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 64.00 | 62.63 | 63.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 64.00 | 62.63 | 63.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 64.00 | 62.63 | 63.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 64.00 | 62.63 | 63.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 64.31 | 62.97 | 63.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 64.36 | 62.97 | 63.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 63.99 | 63.56 | 63.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 64.05 | 63.74 | 63.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 63.70 | 63.73 | 63.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 63.70 | 63.73 | 63.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 63.70 | 63.73 | 63.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 63.67 | 63.73 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 63.55 | 63.69 | 63.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 63.60 | 63.69 | 63.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 63.70 | 63.70 | 63.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 63.52 | 63.70 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 63.58 | 63.67 | 63.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 63.58 | 63.67 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 63.64 | 63.67 | 63.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 64.37 | 63.67 | 63.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 63.75 | 63.67 | 63.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 64.19 | 63.78 | 63.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 63.75 | 64.25 | 64.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 63.05 | 64.01 | 64.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 63.05 | 64.01 | 64.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 62.86 | 63.78 | 63.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 66.00 | 64.07 | 64.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 14:15:00 | 66.00 | 64.07 | 64.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 66.00 | 64.07 | 64.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 66.00 | 64.07 | 64.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 65.93 | 64.44 | 64.24 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 63.90 | 64.10 | 64.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 63.81 | 64.00 | 64.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 11:15:00 | 64.00 | 63.98 | 64.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:30:00 | 63.99 | 63.98 | 64.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 64.38 | 64.06 | 64.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 65.36 | 64.06 | 64.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 66.15 | 64.48 | 64.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 67.04 | 65.23 | 64.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 66.77 | 67.16 | 66.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 66.85 | 67.16 | 66.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 66.65 | 67.01 | 66.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 66.65 | 67.01 | 66.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 66.39 | 66.88 | 66.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 66.39 | 66.88 | 66.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 66.37 | 66.78 | 66.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 66.15 | 66.78 | 66.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 66.15 | 66.66 | 66.34 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-11 12:00:00 | 77.42 | 2025-06-13 09:15:00 | 73.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:00:00 | 77.42 | 2025-06-13 15:15:00 | 74.41 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest2 | 2025-06-25 09:15:00 | 74.39 | 2025-06-26 11:15:00 | 73.81 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-04 10:15:00 | 78.38 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-04 12:45:00 | 77.84 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-07-04 13:45:00 | 78.07 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-07-04 14:30:00 | 77.97 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-14 15:00:00 | 76.56 | 2025-07-15 14:15:00 | 77.39 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-16 14:00:00 | 76.47 | 2025-07-22 15:15:00 | 75.65 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-08-06 13:45:00 | 77.01 | 2025-08-08 09:15:00 | 75.02 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-08-07 15:15:00 | 76.24 | 2025-08-08 09:15:00 | 75.02 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-12 14:15:00 | 74.36 | 2025-08-14 09:15:00 | 76.10 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-08-12 14:45:00 | 74.25 | 2025-08-14 09:15:00 | 76.10 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-09-04 09:15:00 | 84.36 | 2025-09-08 12:15:00 | 92.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-04 14:30:00 | 83.28 | 2025-09-08 12:15:00 | 91.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 09:30:00 | 83.38 | 2025-09-08 12:15:00 | 91.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 13:00:00 | 83.29 | 2025-09-08 12:15:00 | 91.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-11 09:30:00 | 89.59 | 2025-09-12 11:15:00 | 87.57 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-09-11 11:45:00 | 89.00 | 2025-09-12 11:15:00 | 87.57 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-09-11 15:00:00 | 89.28 | 2025-09-12 11:15:00 | 87.57 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-09-15 09:15:00 | 87.94 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-15 10:00:00 | 87.82 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-15 12:00:00 | 87.80 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-16 09:30:00 | 87.99 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-16 11:15:00 | 87.28 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-22 09:45:00 | 93.64 | 2025-09-24 11:15:00 | 91.22 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-09-22 10:45:00 | 93.20 | 2025-09-24 11:15:00 | 91.22 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-09-22 15:00:00 | 93.52 | 2025-09-24 11:15:00 | 91.22 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-10-01 13:45:00 | 84.99 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-08 11:15:00 | 85.37 | 2025-10-14 11:15:00 | 81.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 11:15:00 | 85.37 | 2025-10-15 09:15:00 | 81.32 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2025-11-10 09:15:00 | 79.22 | 2025-11-10 13:15:00 | 80.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-10 13:00:00 | 79.39 | 2025-11-10 13:15:00 | 80.20 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-11-10 13:30:00 | 79.36 | 2025-11-10 14:15:00 | 80.04 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-11 09:15:00 | 79.17 | 2025-11-12 10:15:00 | 81.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-11-18 13:30:00 | 77.55 | 2025-11-19 09:15:00 | 78.77 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-11-18 15:15:00 | 77.70 | 2025-11-19 09:15:00 | 78.77 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-21 09:15:00 | 77.41 | 2025-11-26 13:15:00 | 76.94 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-12-04 13:00:00 | 76.32 | 2025-12-05 14:15:00 | 77.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-05 09:15:00 | 76.41 | 2025-12-05 14:15:00 | 77.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-30 09:15:00 | 76.49 | 2025-12-31 11:15:00 | 77.22 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-30 09:45:00 | 76.26 | 2025-12-31 11:15:00 | 77.22 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-12-31 11:00:00 | 76.61 | 2025-12-31 11:15:00 | 77.22 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-01-05 12:15:00 | 77.50 | 2026-01-06 09:15:00 | 76.17 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-05 14:45:00 | 77.32 | 2026-01-06 09:15:00 | 76.17 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-09 11:30:00 | 75.43 | 2026-01-16 14:15:00 | 71.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 75.43 | 2026-01-19 14:15:00 | 71.43 | STOP_HIT | 0.50 | 5.30% |
| BUY | retest2 | 2026-01-29 11:30:00 | 64.02 | 2026-02-02 09:15:00 | 63.94 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-01-29 13:30:00 | 64.29 | 2026-02-02 09:15:00 | 63.94 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-01-30 09:30:00 | 65.19 | 2026-02-02 09:15:00 | 63.94 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-02-03 10:15:00 | 66.71 | 2026-02-03 10:15:00 | 66.45 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-02-09 10:30:00 | 70.90 | 2026-02-11 12:15:00 | 70.07 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-02-09 11:45:00 | 71.99 | 2026-02-11 12:15:00 | 70.07 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-02-24 09:30:00 | 69.10 | 2026-03-02 09:15:00 | 65.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:15:00 | 69.14 | 2026-03-02 09:15:00 | 65.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:45:00 | 69.26 | 2026-03-02 09:15:00 | 65.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 68.82 | 2026-03-02 09:15:00 | 65.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 68.72 | 2026-03-02 09:15:00 | 65.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:30:00 | 69.10 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-02-24 10:15:00 | 69.14 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2026-02-24 10:45:00 | 69.26 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2026-02-25 09:30:00 | 68.82 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-02-26 11:30:00 | 68.72 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2026-03-19 09:15:00 | 58.27 | 2026-03-20 11:15:00 | 62.34 | STOP_HIT | 1.00 | -6.98% |
| SELL | retest2 | 2026-03-24 14:15:00 | 58.42 | 2026-03-24 15:15:00 | 58.20 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2026-04-08 09:15:00 | 59.20 | 2026-04-15 09:15:00 | 65.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 14:45:00 | 58.35 | 2026-04-15 09:15:00 | 64.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 64.37 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-29 10:15:00 | 63.75 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-04-29 10:45:00 | 64.19 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-05-04 11:15:00 | 63.75 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -1.10% |
