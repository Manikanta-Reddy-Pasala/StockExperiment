# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 225.02
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 22 |
| ALERT3 | 114 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 54 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 40
- **Target hits / Stop hits / Partials:** 2 / 56 / 11
- **Avg / median % per leg:** 0.76% / -0.74%
- **Sum % (uncompounded):** 52.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 4 | 14.3% | 2 | 26 | 0 | -0.78% | -21.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.36% | -5.4% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 2 | 22 | 0 | -0.68% | -16.3% |
| SELL (all) | 41 | 25 | 61.0% | 0 | 30 | 11 | 1.81% | 74.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 25 | 61.0% | 0 | 30 | 11 | 1.81% | 74.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.36% | -5.4% |
| retest2 (combined) | 65 | 29 | 44.6% | 2 | 52 | 11 | 0.89% | 57.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 108.52 | 105.76 | 105.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 109.47 | 107.34 | 106.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 107.89 | 108.03 | 107.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 107.89 | 108.03 | 107.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 108.13 | 107.97 | 107.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 109.01 | 107.94 | 107.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 111.61 | 108.16 | 107.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 110.04 | 111.36 | 111.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 110.04 | 111.36 | 111.41 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 112.44 | 111.18 | 111.05 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 110.60 | 111.25 | 111.32 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 112.07 | 111.44 | 111.36 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 110.55 | 111.27 | 111.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 109.73 | 110.96 | 111.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 111.01 | 110.88 | 111.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 111.01 | 110.88 | 111.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 111.01 | 110.88 | 111.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 111.01 | 110.88 | 111.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 110.45 | 110.79 | 111.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 110.24 | 110.54 | 110.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 110.14 | 110.34 | 110.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:30:00 | 110.25 | 109.98 | 110.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 108.55 | 109.98 | 110.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 110.88 | 109.97 | 110.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 110.88 | 109.97 | 110.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 110.72 | 110.12 | 110.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 110.72 | 110.12 | 110.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 111.06 | 110.42 | 110.47 | SL hit (close>static) qty=1.00 sl=111.02 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 111.00 | 110.53 | 110.52 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 109.49 | 110.40 | 110.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 109.16 | 110.00 | 110.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 109.00 | 108.91 | 109.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 109.00 | 108.91 | 109.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 109.00 | 108.91 | 109.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 109.00 | 108.91 | 109.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 110.28 | 109.21 | 109.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 110.15 | 109.21 | 109.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 110.13 | 109.39 | 109.50 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 110.20 | 109.67 | 109.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 110.76 | 109.85 | 109.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 117.57 | 117.94 | 116.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 117.51 | 117.94 | 116.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 115.55 | 117.31 | 116.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 115.17 | 117.31 | 116.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 115.15 | 116.88 | 116.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 115.15 | 116.88 | 116.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 113.14 | 115.52 | 115.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 111.85 | 113.74 | 114.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 113.42 | 113.39 | 114.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 113.42 | 113.39 | 114.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 112.68 | 113.02 | 113.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 113.42 | 113.02 | 113.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 109.98 | 110.61 | 111.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 109.35 | 110.44 | 111.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 109.52 | 110.44 | 111.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 108.89 | 108.53 | 108.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 108.89 | 108.53 | 108.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 109.39 | 108.72 | 108.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 117.22 | 117.25 | 115.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 117.28 | 117.25 | 115.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 118.19 | 118.53 | 117.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 118.06 | 118.53 | 117.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 118.34 | 118.49 | 117.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 117.85 | 118.49 | 117.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 118.20 | 118.37 | 117.96 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 117.56 | 117.79 | 117.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 116.88 | 117.45 | 117.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 118.18 | 116.99 | 117.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 118.18 | 116.99 | 117.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 118.18 | 116.99 | 117.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 118.18 | 116.99 | 117.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 118.32 | 117.25 | 117.29 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 119.50 | 117.70 | 117.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 15:15:00 | 119.74 | 118.11 | 117.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 121.27 | 121.41 | 120.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:45:00 | 121.17 | 121.41 | 120.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 120.78 | 121.16 | 120.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 120.78 | 121.16 | 120.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 121.17 | 121.16 | 120.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 121.88 | 121.06 | 120.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 121.89 | 121.21 | 120.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 123.60 | 121.21 | 120.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 121.98 | 121.64 | 121.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 121.37 | 121.58 | 121.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 121.17 | 121.58 | 121.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 121.69 | 121.61 | 121.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 122.88 | 121.67 | 121.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 122.40 | 121.82 | 121.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:45:00 | 122.40 | 121.87 | 121.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 121.08 | 121.50 | 121.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 121.08 | 121.50 | 121.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 120.53 | 121.22 | 121.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 119.30 | 119.21 | 119.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:15:00 | 119.57 | 119.21 | 119.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 119.66 | 119.30 | 119.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 120.34 | 119.30 | 119.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 118.81 | 119.23 | 119.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 118.70 | 119.13 | 119.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:00:00 | 118.70 | 118.96 | 119.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 118.42 | 118.92 | 119.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 15:00:00 | 118.42 | 118.82 | 119.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 118.25 | 118.50 | 118.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 117.85 | 118.37 | 118.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.77 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.77 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.50 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 112.50 | 114.08 | 115.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 114.16 | 114.10 | 115.24 | SL hit (close>ema200) qty=0.50 sl=114.10 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 117.65 | 115.34 | 115.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 118.60 | 116.00 | 115.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 116.54 | 117.41 | 116.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 116.54 | 117.41 | 116.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 116.54 | 117.41 | 116.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 117.76 | 117.26 | 116.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:00:00 | 118.00 | 117.41 | 117.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 113.18 | 116.24 | 116.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 113.18 | 116.24 | 116.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 111.13 | 113.65 | 114.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 114.40 | 113.72 | 114.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 11:15:00 | 114.40 | 113.72 | 114.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 114.40 | 113.72 | 114.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 114.40 | 113.72 | 114.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 113.34 | 113.65 | 114.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 112.95 | 113.65 | 114.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 115.97 | 114.13 | 114.45 | SL hit (close>static) qty=1.00 sl=114.43 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 116.19 | 114.87 | 114.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 118.52 | 116.10 | 115.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 118.05 | 118.19 | 117.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 118.10 | 118.19 | 117.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 117.68 | 118.09 | 117.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 117.41 | 118.09 | 117.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 117.47 | 117.97 | 117.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 117.47 | 117.97 | 117.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 116.90 | 117.75 | 117.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 116.79 | 117.75 | 117.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 116.72 | 117.36 | 117.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 116.86 | 117.36 | 117.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 116.64 | 117.22 | 117.27 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 119.05 | 117.34 | 117.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 119.76 | 118.31 | 117.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 120.34 | 120.45 | 119.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 120.34 | 120.45 | 119.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 119.72 | 120.22 | 119.85 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 119.35 | 119.69 | 119.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 119.20 | 119.53 | 119.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 119.65 | 119.56 | 119.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 119.65 | 119.56 | 119.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 119.65 | 119.56 | 119.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 119.64 | 119.56 | 119.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 118.97 | 119.44 | 119.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 118.68 | 119.44 | 119.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 118.91 | 119.33 | 119.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:45:00 | 118.78 | 119.00 | 119.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 120.26 | 119.20 | 119.27 | SL hit (close>static) qty=1.00 sl=119.69 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 120.64 | 119.49 | 119.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 120.90 | 119.77 | 119.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 119.50 | 120.01 | 119.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 10:15:00 | 119.50 | 120.01 | 119.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 119.50 | 120.01 | 119.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 119.57 | 120.01 | 119.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 119.48 | 119.90 | 119.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 119.64 | 119.90 | 119.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 119.48 | 119.82 | 119.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:30:00 | 119.52 | 119.82 | 119.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 119.07 | 119.67 | 119.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 119.07 | 119.67 | 119.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 118.53 | 119.44 | 119.53 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 09:15:00 | 121.09 | 119.67 | 119.61 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 118.81 | 119.54 | 119.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 117.30 | 118.99 | 119.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 118.86 | 118.80 | 119.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 118.86 | 118.80 | 119.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 119.08 | 118.86 | 119.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 119.08 | 118.86 | 119.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 119.45 | 118.98 | 119.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:15:00 | 119.52 | 118.98 | 119.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 120.04 | 119.19 | 119.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 120.04 | 119.19 | 119.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 120.00 | 119.35 | 119.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 121.30 | 119.74 | 119.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 121.01 | 121.13 | 120.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:30:00 | 121.14 | 121.13 | 120.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 120.51 | 121.01 | 120.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 120.37 | 121.01 | 120.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 121.73 | 121.15 | 120.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 122.47 | 121.20 | 120.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 122.00 | 121.48 | 121.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 122.70 | 121.61 | 121.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:45:00 | 122.54 | 121.70 | 121.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 121.90 | 121.81 | 121.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 121.37 | 121.81 | 121.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 121.37 | 121.72 | 121.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 121.29 | 121.72 | 121.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 121.06 | 121.59 | 121.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:15:00 | 121.00 | 121.59 | 121.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 120.76 | 121.42 | 121.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:15:00 | 120.69 | 121.42 | 121.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 120.76 | 121.29 | 121.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 120.76 | 121.29 | 121.33 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 121.98 | 121.43 | 121.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 122.28 | 121.60 | 121.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 127.62 | 128.92 | 127.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 13:15:00 | 127.62 | 128.92 | 127.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 127.62 | 128.92 | 127.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 127.62 | 128.92 | 127.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 126.75 | 128.49 | 127.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 126.75 | 128.49 | 127.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 127.00 | 128.19 | 127.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 127.88 | 128.19 | 127.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 127.80 | 128.37 | 127.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 127.90 | 128.37 | 127.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 125.64 | 127.83 | 127.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 125.05 | 127.83 | 127.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 124.73 | 127.21 | 127.32 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 128.75 | 127.26 | 127.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 129.77 | 128.36 | 127.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 129.56 | 129.76 | 128.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:45:00 | 129.40 | 129.76 | 128.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 129.27 | 129.66 | 128.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 129.27 | 129.66 | 128.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 129.24 | 129.57 | 128.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 128.35 | 129.57 | 128.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 128.30 | 129.32 | 128.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 127.82 | 129.32 | 128.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 127.97 | 129.05 | 128.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 127.87 | 129.05 | 128.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 128.00 | 128.67 | 128.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 127.63 | 128.36 | 128.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 15:15:00 | 126.70 | 126.40 | 126.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:15:00 | 135.25 | 126.40 | 126.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 31 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 134.97 | 128.12 | 127.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 142.64 | 133.32 | 130.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 163.40 | 163.89 | 153.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:00:00 | 163.40 | 163.89 | 153.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 153.70 | 161.53 | 156.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 153.70 | 161.53 | 156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 153.90 | 160.00 | 156.65 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 144.65 | 153.63 | 154.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 144.30 | 146.34 | 148.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 146.40 | 146.18 | 147.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 15:00:00 | 146.40 | 146.18 | 147.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 147.60 | 146.46 | 147.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 144.90 | 146.46 | 147.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 144.85 | 146.14 | 147.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:45:00 | 144.20 | 145.57 | 146.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:00:00 | 144.10 | 145.08 | 146.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 147.98 | 146.73 | 146.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 147.98 | 146.73 | 146.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 151.35 | 147.65 | 147.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 147.16 | 149.00 | 148.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 147.16 | 149.00 | 148.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 147.16 | 149.00 | 148.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 154.09 | 151.09 | 149.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:45:00 | 153.07 | 152.46 | 150.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 148.92 | 150.24 | 150.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 148.92 | 150.24 | 150.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 147.78 | 149.50 | 150.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 149.20 | 148.97 | 149.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 149.20 | 148.97 | 149.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 149.20 | 148.97 | 149.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 149.20 | 148.97 | 149.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 149.00 | 148.98 | 149.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 148.59 | 148.90 | 149.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 14:15:00 | 149.50 | 149.03 | 149.27 | SL hit (close>static) qty=1.00 sl=149.48 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 150.92 | 149.56 | 149.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 152.93 | 150.24 | 149.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 153.80 | 154.29 | 152.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:15:00 | 153.79 | 154.29 | 152.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 152.02 | 153.84 | 152.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 151.89 | 153.84 | 152.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 151.85 | 153.44 | 152.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 153.25 | 153.44 | 152.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-20 09:15:00 | 168.58 | 164.57 | 161.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 164.22 | 166.54 | 166.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 163.30 | 165.12 | 165.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 163.55 | 163.14 | 164.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 163.55 | 163.14 | 164.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 163.55 | 163.14 | 164.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 164.24 | 163.14 | 164.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 166.96 | 163.90 | 164.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 166.96 | 163.90 | 164.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 164.03 | 163.93 | 164.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:45:00 | 163.80 | 163.96 | 164.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 162.50 | 163.59 | 164.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 163.64 | 162.83 | 163.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 155.61 | 157.71 | 159.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 154.38 | 157.71 | 159.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 155.46 | 157.71 | 159.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 157.84 | 157.74 | 158.92 | SL hit (close>ema200) qty=0.50 sl=157.74 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 152.24 | 151.86 | 151.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 153.19 | 152.17 | 151.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 152.25 | 152.52 | 152.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 152.25 | 152.52 | 152.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 152.25 | 152.52 | 152.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 152.25 | 152.52 | 152.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 151.79 | 152.38 | 152.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 151.79 | 152.38 | 152.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 152.10 | 152.32 | 152.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 154.70 | 152.32 | 152.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 150.94 | 152.98 | 153.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 150.94 | 152.98 | 153.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 150.79 | 152.04 | 152.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 151.71 | 151.19 | 151.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 151.71 | 151.19 | 151.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 151.71 | 151.19 | 151.93 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 152.47 | 152.04 | 152.00 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 151.58 | 151.97 | 151.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 151.15 | 151.80 | 151.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 147.29 | 147.07 | 148.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 147.08 | 147.07 | 148.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 147.59 | 147.17 | 147.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 147.70 | 147.17 | 147.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 147.97 | 147.33 | 147.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 148.12 | 147.33 | 147.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 147.80 | 147.43 | 147.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 147.47 | 147.52 | 147.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 148.44 | 147.77 | 148.00 | SL hit (close>static) qty=1.00 sl=148.19 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 149.80 | 148.33 | 148.22 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 147.65 | 148.31 | 148.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 147.52 | 148.15 | 148.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 147.96 | 147.94 | 148.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:45:00 | 147.89 | 147.94 | 148.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 146.91 | 147.61 | 147.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:30:00 | 146.25 | 147.07 | 147.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 146.26 | 146.77 | 147.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 146.17 | 146.56 | 147.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 138.94 | 141.05 | 142.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 138.95 | 141.05 | 142.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 138.86 | 141.05 | 142.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 140.16 | 139.65 | 141.28 | SL hit (close>ema200) qty=0.50 sl=139.65 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 142.27 | 141.42 | 141.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 144.62 | 142.53 | 141.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 144.01 | 144.33 | 143.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 13:00:00 | 144.01 | 144.33 | 143.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 143.22 | 144.43 | 143.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 143.22 | 144.43 | 143.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 143.18 | 144.18 | 143.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 142.45 | 144.18 | 143.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 142.60 | 143.67 | 143.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 142.24 | 143.38 | 143.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 141.77 | 141.74 | 142.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 141.77 | 141.74 | 142.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 142.89 | 141.92 | 142.15 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 143.36 | 142.46 | 142.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 143.75 | 143.05 | 142.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 143.28 | 143.56 | 143.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 143.28 | 143.56 | 143.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 143.28 | 143.56 | 143.18 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 142.48 | 143.06 | 143.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 142.16 | 142.88 | 143.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 141.09 | 140.99 | 141.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 141.09 | 140.99 | 141.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 141.09 | 140.99 | 141.75 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 144.29 | 142.06 | 142.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 145.03 | 142.65 | 142.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 143.29 | 143.52 | 143.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 15:00:00 | 143.29 | 143.52 | 143.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 142.89 | 143.39 | 143.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 146.85 | 145.03 | 143.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 145.75 | 146.79 | 146.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 145.75 | 146.79 | 146.81 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 148.30 | 146.92 | 146.85 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 146.11 | 146.95 | 146.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 145.21 | 146.28 | 146.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 141.61 | 141.55 | 142.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 141.59 | 141.55 | 142.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 142.84 | 141.98 | 142.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 142.85 | 141.98 | 142.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 142.55 | 142.09 | 142.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 141.94 | 142.55 | 142.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 143.20 | 142.78 | 142.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 143.20 | 142.78 | 142.76 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 142.30 | 142.73 | 142.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 141.40 | 142.46 | 142.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 138.79 | 138.31 | 139.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 139.19 | 138.31 | 139.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 136.87 | 138.02 | 139.31 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 141.45 | 139.76 | 139.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 141.72 | 140.15 | 139.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 139.92 | 140.13 | 139.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 11:15:00 | 139.92 | 140.13 | 139.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 139.92 | 140.13 | 139.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 140.00 | 140.13 | 139.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 136.09 | 139.32 | 139.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 133.38 | 138.13 | 138.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 136.60 | 136.48 | 137.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 15:15:00 | 135.90 | 134.78 | 135.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 135.90 | 134.78 | 135.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 134.15 | 134.78 | 135.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 136.20 | 134.79 | 135.21 | SL hit (close>static) qty=1.00 sl=135.90 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 136.10 | 135.07 | 134.93 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 134.00 | 134.76 | 134.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 131.28 | 133.45 | 134.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 133.06 | 133.02 | 133.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 133.06 | 133.02 | 133.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 134.48 | 133.31 | 133.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 134.48 | 133.31 | 133.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 134.50 | 133.55 | 133.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 141.22 | 133.55 | 133.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 142.20 | 135.28 | 134.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 146.00 | 138.86 | 136.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 152.00 | 152.57 | 149.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 153.18 | 152.69 | 149.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 15:00:00 | 152.95 | 152.71 | 150.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:15:00 | 153.45 | 152.62 | 150.96 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:45:00 | 153.14 | 152.50 | 151.19 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 151.10 | 151.95 | 151.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 151.10 | 151.95 | 151.40 | SL hit (close<ema400) qty=1.00 sl=151.40 alert=retest1 |

### Cycle 58 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 148.98 | 150.73 | 150.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 148.02 | 149.87 | 150.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 149.76 | 149.30 | 149.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 149.76 | 149.30 | 149.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 149.76 | 149.30 | 149.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 149.30 | 149.30 | 149.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 149.50 | 149.34 | 149.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 149.78 | 149.34 | 149.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 149.89 | 149.45 | 149.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 150.32 | 149.45 | 149.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 150.95 | 149.75 | 149.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 150.95 | 149.75 | 149.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 150.94 | 149.99 | 150.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 149.25 | 149.99 | 150.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:15:00 | 141.79 | 146.24 | 148.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 12:15:00 | 143.45 | 142.51 | 144.56 | SL hit (close>ema200) qty=0.50 sl=142.51 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 144.23 | 142.78 | 142.77 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 142.06 | 143.27 | 143.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 140.39 | 142.69 | 143.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 141.44 | 141.31 | 141.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 15:00:00 | 141.44 | 141.31 | 141.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 141.33 | 141.13 | 141.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 141.31 | 141.13 | 141.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 139.14 | 137.62 | 138.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 139.14 | 137.62 | 138.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 137.73 | 137.64 | 138.48 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 140.07 | 138.82 | 138.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 141.40 | 139.58 | 139.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 12:15:00 | 139.56 | 139.71 | 139.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 13:00:00 | 139.56 | 139.71 | 139.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 139.69 | 139.70 | 139.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 139.48 | 139.70 | 139.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 138.98 | 139.56 | 139.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 138.98 | 139.56 | 139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 139.30 | 139.51 | 139.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 136.99 | 139.51 | 139.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 138.47 | 139.30 | 139.22 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 138.79 | 139.15 | 139.17 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 139.50 | 139.17 | 139.16 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 138.90 | 139.12 | 139.14 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 139.23 | 139.16 | 139.15 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 12:15:00 | 138.90 | 139.10 | 139.13 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 140.50 | 139.36 | 139.24 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 138.94 | 139.21 | 139.22 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 09:15:00 | 143.27 | 140.02 | 139.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 147.22 | 142.08 | 140.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 14:15:00 | 146.84 | 147.59 | 145.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 15:00:00 | 146.84 | 147.59 | 145.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 147.60 | 147.59 | 145.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 150.27 | 148.25 | 146.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 150.50 | 152.37 | 152.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 150.50 | 152.37 | 152.51 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 154.85 | 152.72 | 152.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 156.40 | 153.45 | 152.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 13:15:00 | 152.82 | 153.49 | 153.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 13:15:00 | 152.82 | 153.49 | 153.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 152.82 | 153.49 | 153.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 152.82 | 153.49 | 153.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 151.55 | 153.10 | 152.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 151.55 | 153.10 | 152.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 151.75 | 152.83 | 152.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 145.80 | 151.43 | 152.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 148.26 | 148.12 | 149.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 10:45:00 | 148.06 | 148.12 | 149.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 151.07 | 148.71 | 149.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 151.12 | 148.71 | 149.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 151.68 | 149.31 | 149.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 152.80 | 149.31 | 149.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 150.20 | 149.84 | 150.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 154.27 | 149.84 | 150.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 154.79 | 150.83 | 150.46 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 151.19 | 152.57 | 152.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 150.63 | 152.18 | 152.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 154.35 | 152.32 | 152.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 154.35 | 152.32 | 152.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 154.35 | 152.32 | 152.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 155.30 | 152.32 | 152.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 155.23 | 152.90 | 152.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 157.69 | 153.86 | 153.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 152.30 | 154.84 | 154.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 152.30 | 154.84 | 154.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 152.30 | 154.84 | 154.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 151.38 | 154.84 | 154.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 151.06 | 154.09 | 153.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 152.54 | 153.91 | 153.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 167.79 | 164.20 | 162.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 217.90 | 218.95 | 219.07 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 221.38 | 219.27 | 219.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 222.69 | 219.96 | 219.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 226.18 | 228.78 | 226.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 226.18 | 228.78 | 226.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 226.18 | 228.78 | 226.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:15:00 | 232.84 | 228.79 | 227.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:00:00 | 232.75 | 229.58 | 228.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 226.25 | 228.58 | 228.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 226.25 | 228.58 | 228.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 225.26 | 227.58 | 228.12 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 14:15:00 | 109.01 | 2025-05-21 11:15:00 | 110.04 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-05-16 09:15:00 | 111.61 | 2025-05-21 11:15:00 | 110.04 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-30 09:45:00 | 110.24 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-05-30 14:00:00 | 110.14 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-05-30 14:30:00 | 110.25 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-30 15:00:00 | 108.55 | 2025-06-02 12:15:00 | 111.06 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-06-19 10:30:00 | 109.35 | 2025-06-24 12:15:00 | 108.89 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-06-19 11:15:00 | 109.52 | 2025-06-24 12:15:00 | 108.89 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-07-14 09:15:00 | 121.88 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-14 09:45:00 | 121.89 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-14 10:15:00 | 123.60 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-07-14 13:15:00 | 121.98 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-15 09:15:00 | 122.88 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-15 09:45:00 | 122.40 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-15 10:45:00 | 122.40 | 2025-07-16 14:15:00 | 121.08 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-22 11:00:00 | 118.70 | 2025-07-28 13:15:00 | 112.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 13:00:00 | 118.70 | 2025-07-28 13:15:00 | 112.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:15:00 | 118.42 | 2025-07-28 13:15:00 | 112.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 15:00:00 | 118.42 | 2025-07-28 13:15:00 | 112.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 118.70 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-07-22 13:00:00 | 118.70 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-07-22 14:15:00 | 118.42 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-07-22 15:00:00 | 118.42 | 2025-07-28 14:15:00 | 114.16 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-07-23 10:45:00 | 117.85 | 2025-07-29 13:15:00 | 117.65 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-08-01 09:15:00 | 117.76 | 2025-08-01 14:15:00 | 113.18 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-08-01 10:00:00 | 118.00 | 2025-08-01 14:15:00 | 113.18 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2025-08-07 13:15:00 | 112.95 | 2025-08-07 14:15:00 | 115.97 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-08-22 11:15:00 | 118.68 | 2025-08-25 12:15:00 | 120.26 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-22 12:00:00 | 118.91 | 2025-08-25 12:15:00 | 120.26 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-25 09:45:00 | 118.78 | 2025-08-25 12:15:00 | 120.26 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-09-03 09:15:00 | 122.47 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-09-03 13:45:00 | 122.00 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-04 09:15:00 | 122.70 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-04 10:45:00 | 122.54 | 2025-09-05 11:15:00 | 120.76 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-30 11:45:00 | 144.20 | 2025-10-01 12:15:00 | 147.98 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-30 14:00:00 | 144.10 | 2025-10-01 12:15:00 | 147.98 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-10-07 09:30:00 | 154.09 | 2025-10-08 14:15:00 | 148.92 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-10-07 13:45:00 | 153.07 | 2025-10-08 14:15:00 | 148.92 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-10-10 12:00:00 | 148.59 | 2025-10-10 14:15:00 | 149.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-10-13 09:15:00 | 148.53 | 2025-10-13 09:15:00 | 150.92 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-10-15 11:15:00 | 153.25 | 2025-10-20 09:15:00 | 168.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-29 13:45:00 | 163.80 | 2025-11-04 09:15:00 | 155.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 14:30:00 | 162.50 | 2025-11-04 09:15:00 | 154.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:15:00 | 163.64 | 2025-11-04 09:15:00 | 155.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 13:45:00 | 163.80 | 2025-11-04 10:15:00 | 157.84 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2025-10-29 14:30:00 | 162.50 | 2025-11-04 10:15:00 | 157.84 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-10-30 13:15:00 | 163.64 | 2025-11-04 10:15:00 | 157.84 | STOP_HIT | 0.50 | 3.54% |
| BUY | retest2 | 2025-11-14 09:15:00 | 154.70 | 2025-11-18 09:15:00 | 150.94 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-11-26 12:30:00 | 147.47 | 2025-11-26 14:15:00 | 148.44 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-12-02 13:30:00 | 146.25 | 2025-12-08 13:15:00 | 138.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 10:30:00 | 146.26 | 2025-12-08 13:15:00 | 138.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 12:45:00 | 146.17 | 2025-12-08 13:15:00 | 138.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:30:00 | 146.25 | 2025-12-09 11:15:00 | 140.16 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-12-03 10:30:00 | 146.26 | 2025-12-09 11:15:00 | 140.16 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-12-03 12:45:00 | 146.17 | 2025-12-09 11:15:00 | 140.16 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest2 | 2026-01-01 09:30:00 | 146.85 | 2026-01-05 13:15:00 | 145.75 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-14 10:00:00 | 141.94 | 2026-01-16 09:15:00 | 143.20 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-01-29 09:15:00 | 134.15 | 2026-01-29 13:15:00 | 136.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-29 14:15:00 | 133.91 | 2026-02-01 10:15:00 | 136.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest1 | 2026-02-06 11:00:00 | 153.18 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2026-02-06 15:00:00 | 152.95 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-02-09 09:15:00 | 153.45 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2026-02-09 10:45:00 | 153.14 | 2026-02-09 15:15:00 | 151.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-02-12 09:15:00 | 149.25 | 2026-02-13 11:15:00 | 141.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 149.25 | 2026-02-16 12:15:00 | 143.45 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest2 | 2026-03-16 09:30:00 | 150.27 | 2026-03-19 14:15:00 | 150.50 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2026-04-02 11:45:00 | 152.54 | 2026-04-08 09:15:00 | 167.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-07 10:15:00 | 232.84 | 2026-05-08 12:15:00 | 226.25 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-05-07 11:00:00 | 232.75 | 2026-05-08 12:15:00 | 226.25 | STOP_HIT | 1.00 | -2.79% |
