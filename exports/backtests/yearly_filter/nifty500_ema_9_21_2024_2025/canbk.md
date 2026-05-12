# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 134.13
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 130 |
| ALERT1 | 95 |
| ALERT2 | 92 |
| ALERT2_SKIP | 36 |
| ALERT3 | 267 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 142 |
| PARTIAL | 18 |
| TARGET_HIT | 5 |
| STOP_HIT | 140 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 163 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 95
- **Target hits / Stop hits / Partials:** 5 / 140 / 18
- **Avg / median % per leg:** 0.74% / -0.32%
- **Sum % (uncompounded):** 120.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 27 | 44.3% | 5 | 56 | 0 | 0.76% | 46.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| BUY @ 3rd Alert (retest2) | 59 | 27 | 45.8% | 5 | 54 | 0 | 0.82% | 48.2% |
| SELL (all) | 102 | 41 | 40.2% | 0 | 84 | 18 | 0.73% | 74.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.22% | -1.2% |
| SELL @ 3rd Alert (retest2) | 101 | 41 | 40.6% | 0 | 83 | 18 | 0.75% | 75.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.10% | -3.3% |
| retest2 (combined) | 160 | 68 | 42.5% | 5 | 137 | 18 | 0.77% | 123.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 112.81 | 111.32 | 111.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 15:15:00 | 113.80 | 112.54 | 111.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 115.70 | 117.04 | 115.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 09:30:00 | 115.70 | 117.04 | 115.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 113.75 | 116.39 | 115.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 113.75 | 116.39 | 115.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 113.75 | 115.86 | 114.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 113.75 | 115.86 | 114.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 113.65 | 114.40 | 114.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 114.60 | 114.46 | 114.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 11:15:00 | 114.05 | 114.38 | 114.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 12:15:00 | 113.40 | 114.18 | 114.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 114.25 | 114.04 | 114.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 114.25 | 114.04 | 114.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 114.25 | 114.04 | 114.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:30:00 | 114.50 | 114.04 | 114.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 114.40 | 114.11 | 114.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 12:00:00 | 114.40 | 114.11 | 114.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 114.50 | 114.19 | 114.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 114.05 | 114.19 | 114.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 114.70 | 114.29 | 114.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 12:15:00 | 115.65 | 114.78 | 114.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 12:15:00 | 115.80 | 115.82 | 115.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 13:00:00 | 115.80 | 115.82 | 115.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 117.20 | 117.73 | 117.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 117.20 | 117.73 | 117.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 117.30 | 117.64 | 117.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 118.10 | 117.64 | 117.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 116.15 | 117.35 | 117.08 | SL hit (close<static) qty=1.00 sl=117.10 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 116.20 | 117.05 | 117.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 115.95 | 116.69 | 116.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 116.45 | 116.41 | 116.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 116.45 | 116.41 | 116.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 116.45 | 116.41 | 116.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:30:00 | 117.55 | 116.41 | 116.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 116.10 | 116.34 | 116.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:00:00 | 115.80 | 116.29 | 116.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:30:00 | 115.65 | 116.18 | 116.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:15:00 | 115.90 | 116.15 | 116.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:15:00 | 115.85 | 116.18 | 116.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 115.10 | 115.97 | 116.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:45:00 | 114.85 | 115.59 | 116.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 116.85 | 115.90 | 116.09 | SL hit (close>static) qty=1.00 sl=116.35 alert=retest2 |

### Cycle 7 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 117.55 | 116.23 | 116.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 118.80 | 116.74 | 116.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 120.90 | 124.59 | 121.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 120.90 | 124.59 | 121.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 120.90 | 124.59 | 121.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 119.05 | 124.59 | 121.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 115.45 | 122.76 | 121.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 115.45 | 122.76 | 121.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 102.60 | 118.73 | 119.56 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 117.95 | 116.31 | 116.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 122.04 | 119.14 | 118.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 121.22 | 121.67 | 120.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 121.22 | 121.67 | 120.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 122.43 | 122.85 | 122.12 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 120.84 | 121.78 | 121.85 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 121.90 | 121.63 | 121.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 122.77 | 121.90 | 121.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 12:15:00 | 121.51 | 121.84 | 121.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 12:15:00 | 121.51 | 121.84 | 121.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 121.51 | 121.84 | 121.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 121.51 | 121.84 | 121.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 122.06 | 121.88 | 121.78 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 121.24 | 121.65 | 121.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 09:15:00 | 120.81 | 121.28 | 121.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 121.03 | 119.86 | 120.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 121.03 | 119.86 | 120.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 121.03 | 119.86 | 120.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 121.03 | 119.86 | 120.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 119.75 | 119.84 | 120.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:45:00 | 118.43 | 119.49 | 120.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:00:00 | 118.46 | 119.14 | 119.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:30:00 | 118.85 | 118.39 | 118.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:00:00 | 118.90 | 118.52 | 118.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 118.93 | 118.61 | 118.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:15:00 | 117.92 | 118.61 | 118.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 121.52 | 118.90 | 118.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 121.52 | 118.90 | 118.75 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 13:15:00 | 118.26 | 119.16 | 119.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 117.16 | 118.33 | 118.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 117.04 | 116.98 | 117.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 117.60 | 117.10 | 117.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 117.60 | 117.10 | 117.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 117.50 | 117.10 | 117.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 117.04 | 117.09 | 117.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 116.74 | 117.14 | 117.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 11:15:00 | 117.66 | 117.32 | 117.38 | SL hit (close>static) qty=1.00 sl=117.61 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 117.73 | 117.47 | 117.44 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 116.93 | 117.42 | 117.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 115.85 | 116.99 | 117.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 116.67 | 116.04 | 116.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 10:15:00 | 116.67 | 116.04 | 116.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 116.67 | 116.04 | 116.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 116.50 | 116.04 | 116.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 116.95 | 116.22 | 116.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:30:00 | 117.05 | 116.22 | 116.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 116.70 | 116.41 | 116.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:15:00 | 115.85 | 116.41 | 116.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 13:15:00 | 117.01 | 114.57 | 114.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 117.01 | 114.57 | 114.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 14:15:00 | 117.40 | 115.13 | 114.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 116.02 | 116.03 | 115.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 14:00:00 | 116.02 | 116.03 | 115.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 117.39 | 116.31 | 115.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 115.64 | 116.31 | 115.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 115.87 | 116.30 | 115.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:30:00 | 115.95 | 116.30 | 115.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 115.68 | 116.18 | 115.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 116.15 | 116.18 | 115.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 113.99 | 115.74 | 115.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 113.40 | 115.27 | 115.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 114.99 | 114.11 | 114.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 114.99 | 114.11 | 114.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 114.99 | 114.11 | 114.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 114.68 | 114.11 | 114.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 115.07 | 114.30 | 114.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 115.38 | 114.30 | 114.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 115.00 | 114.44 | 114.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:15:00 | 114.83 | 114.44 | 114.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 114.75 | 114.61 | 114.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 114.70 | 114.63 | 114.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 114.34 | 114.64 | 114.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 114.51 | 114.61 | 114.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 110.74 | 114.61 | 114.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 109.09 | 114.14 | 114.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 109.01 | 114.14 | 114.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 15:15:00 | 112.73 | 112.66 | 113.29 | SL hit (close>ema200) qty=0.50 sl=112.66 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 113.79 | 112.68 | 112.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 116.79 | 113.69 | 113.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 115.37 | 115.58 | 114.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:00:00 | 115.37 | 115.58 | 114.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 115.34 | 115.64 | 115.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 115.34 | 115.64 | 115.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 114.47 | 115.40 | 115.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 114.47 | 115.40 | 115.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 114.74 | 115.27 | 115.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 114.94 | 115.09 | 115.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:15:00 | 115.06 | 115.09 | 115.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 114.63 | 115.00 | 115.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 114.63 | 115.00 | 115.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 113.51 | 114.70 | 114.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 107.32 | 106.71 | 108.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 106.26 | 106.65 | 108.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 107.56 | 106.20 | 107.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 107.56 | 106.20 | 107.09 | SL hit (close>ema400) qty=1.00 sl=107.09 alert=retest1 |

### Cycle 21 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 108.22 | 107.49 | 107.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 109.35 | 108.13 | 107.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 109.23 | 109.47 | 108.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 109.23 | 109.47 | 108.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 109.23 | 109.47 | 108.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 108.58 | 109.47 | 108.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 109.11 | 109.49 | 109.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 109.35 | 109.49 | 109.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 109.06 | 109.40 | 109.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 108.76 | 109.40 | 109.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 108.68 | 109.26 | 109.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 108.68 | 109.26 | 109.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 108.01 | 109.01 | 108.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 108.00 | 109.01 | 108.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 107.50 | 108.71 | 108.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 106.71 | 108.31 | 108.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 106.97 | 106.42 | 107.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 106.97 | 106.42 | 107.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 106.97 | 106.42 | 107.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 106.70 | 106.42 | 107.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 110.20 | 107.85 | 107.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 110.20 | 107.85 | 107.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 111.38 | 109.93 | 108.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 10:15:00 | 110.79 | 110.92 | 110.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 11:00:00 | 110.79 | 110.92 | 110.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 111.69 | 112.19 | 111.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 111.69 | 112.19 | 111.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 112.03 | 112.16 | 111.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 112.03 | 112.16 | 111.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 111.80 | 112.09 | 111.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:30:00 | 111.76 | 112.09 | 111.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 111.55 | 111.98 | 111.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 111.55 | 111.98 | 111.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 111.43 | 111.87 | 111.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 111.43 | 111.87 | 111.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 111.45 | 111.79 | 111.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 111.16 | 111.49 | 111.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 110.32 | 110.00 | 110.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 110.32 | 110.00 | 110.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 110.30 | 110.06 | 110.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 111.42 | 110.06 | 110.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 112.03 | 110.45 | 110.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 112.03 | 110.45 | 110.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 111.67 | 110.70 | 110.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 110.60 | 110.80 | 110.82 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 13:15:00 | 112.97 | 111.24 | 111.02 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 108.93 | 111.04 | 111.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 108.00 | 108.81 | 109.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 103.90 | 103.11 | 104.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 103.90 | 103.11 | 104.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 103.98 | 103.43 | 104.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 103.40 | 103.62 | 104.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:30:00 | 103.49 | 103.61 | 104.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:15:00 | 103.48 | 103.61 | 104.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 105.19 | 103.67 | 103.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 105.19 | 103.67 | 103.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 105.80 | 104.09 | 103.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 106.01 | 106.59 | 105.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 106.01 | 106.59 | 105.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 106.01 | 106.59 | 105.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:15:00 | 106.03 | 106.59 | 105.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 106.17 | 106.50 | 105.92 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 105.39 | 105.78 | 105.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 104.82 | 105.47 | 105.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 104.73 | 104.65 | 105.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 104.73 | 104.65 | 105.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 105.50 | 104.86 | 105.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 105.54 | 104.86 | 105.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 105.89 | 105.07 | 105.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 105.79 | 105.07 | 105.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 104.97 | 105.21 | 105.24 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 107.73 | 105.71 | 105.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 109.16 | 106.40 | 105.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 109.16 | 109.42 | 108.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 109.16 | 109.42 | 108.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 108.15 | 109.14 | 108.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 108.38 | 109.14 | 108.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 107.90 | 108.89 | 108.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 107.90 | 108.89 | 108.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 107.61 | 108.64 | 108.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 107.61 | 108.64 | 108.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 108.38 | 108.41 | 108.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:45:00 | 107.96 | 108.41 | 108.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 108.26 | 108.39 | 108.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 107.77 | 108.39 | 108.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 109.32 | 108.58 | 108.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:15:00 | 110.30 | 109.14 | 108.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 109.99 | 111.00 | 111.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 109.99 | 111.00 | 111.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 106.65 | 108.20 | 109.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 104.62 | 104.57 | 106.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 104.62 | 104.57 | 106.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 106.13 | 105.05 | 105.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 106.13 | 105.05 | 105.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 105.38 | 105.12 | 105.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 104.37 | 104.97 | 105.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 106.21 | 104.95 | 105.33 | SL hit (close>static) qty=1.00 sl=106.20 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 104.60 | 103.73 | 103.65 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 102.79 | 103.58 | 103.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 102.70 | 103.14 | 103.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 98.52 | 97.95 | 99.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:15:00 | 99.00 | 97.95 | 99.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 98.99 | 98.16 | 99.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:45:00 | 99.51 | 98.16 | 99.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 98.53 | 98.23 | 99.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 98.45 | 98.23 | 99.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:45:00 | 98.16 | 98.22 | 99.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 98.29 | 98.27 | 98.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 93.53 | 97.01 | 98.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 93.25 | 97.01 | 98.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 93.38 | 97.01 | 98.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 96.14 | 95.38 | 96.64 | SL hit (close>ema200) qty=0.50 sl=95.38 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 99.75 | 97.45 | 97.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 101.25 | 98.21 | 97.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 103.14 | 103.36 | 101.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 103.14 | 103.36 | 101.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 102.78 | 103.27 | 102.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 102.53 | 103.27 | 102.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 102.20 | 103.06 | 102.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 102.20 | 103.06 | 102.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 102.54 | 102.96 | 102.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:15:00 | 103.02 | 102.96 | 102.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 101.67 | 103.03 | 102.63 | SL hit (close<static) qty=1.00 sl=102.17 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 99.82 | 101.96 | 102.19 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 103.49 | 102.21 | 102.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 104.28 | 103.01 | 102.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 105.03 | 105.12 | 104.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 105.03 | 105.12 | 104.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 104.25 | 104.92 | 104.39 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 103.45 | 104.11 | 104.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 102.75 | 103.84 | 104.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 104.55 | 103.98 | 104.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 104.55 | 103.98 | 104.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 104.55 | 103.98 | 104.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 104.55 | 103.98 | 104.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 104.56 | 104.10 | 104.13 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 12:15:00 | 105.00 | 104.28 | 104.21 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 103.80 | 104.10 | 104.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 103.47 | 103.97 | 104.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 12:15:00 | 98.57 | 97.99 | 99.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 13:00:00 | 98.57 | 97.99 | 99.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 101.07 | 98.71 | 99.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 101.07 | 98.71 | 99.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 100.60 | 99.09 | 99.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:45:00 | 100.15 | 99.28 | 99.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 99.74 | 99.37 | 99.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 99.74 | 99.37 | 99.35 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 97.69 | 99.02 | 99.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 97.34 | 98.69 | 99.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 96.80 | 95.68 | 96.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 96.80 | 95.68 | 96.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 96.80 | 95.68 | 96.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 96.80 | 95.68 | 96.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 96.54 | 95.85 | 96.78 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 100.75 | 97.54 | 97.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 102.36 | 100.45 | 99.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 101.04 | 101.38 | 100.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 101.04 | 101.38 | 100.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 101.90 | 102.43 | 101.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 101.90 | 102.43 | 101.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 101.15 | 102.17 | 101.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 101.15 | 102.17 | 101.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 101.23 | 101.98 | 101.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 101.29 | 101.98 | 101.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 101.99 | 102.01 | 101.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 101.28 | 102.01 | 101.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 101.49 | 101.90 | 101.83 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 101.27 | 101.71 | 101.75 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 104.40 | 102.22 | 101.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 14:15:00 | 105.10 | 104.07 | 103.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 108.86 | 109.07 | 108.17 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 12:45:00 | 110.10 | 109.29 | 108.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 14:45:00 | 109.88 | 109.33 | 108.96 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 109.17 | 109.30 | 109.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:15:00 | 109.12 | 109.30 | 109.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 109.07 | 109.25 | 109.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:30:00 | 109.06 | 109.25 | 109.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 108.85 | 109.17 | 109.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-11 13:15:00 | 108.85 | 109.17 | 109.05 | SL hit (close<ema400) qty=1.00 sl=109.05 alert=retest1 |

### Cycle 46 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 107.73 | 108.76 | 108.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 107.48 | 108.08 | 108.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 106.64 | 106.37 | 107.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 106.64 | 106.37 | 107.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 107.82 | 106.70 | 107.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 108.06 | 106.70 | 107.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 107.01 | 106.76 | 107.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 106.77 | 106.76 | 107.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:45:00 | 106.79 | 106.81 | 107.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 106.83 | 107.02 | 107.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 106.70 | 106.96 | 107.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 106.80 | 106.93 | 107.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 107.03 | 106.93 | 107.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 101.43 | 103.48 | 104.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 101.45 | 103.48 | 104.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 101.49 | 103.48 | 104.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 101.36 | 103.48 | 104.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 103.75 | 103.47 | 104.49 | SL hit (close>ema200) qty=0.50 sl=103.47 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 100.60 | 100.30 | 100.28 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 100.08 | 100.25 | 100.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 10:15:00 | 100.05 | 100.21 | 100.24 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 100.92 | 100.24 | 100.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 102.43 | 100.92 | 100.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 101.41 | 101.53 | 101.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 101.41 | 101.53 | 101.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 98.94 | 100.99 | 100.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 98.94 | 100.99 | 100.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 97.74 | 100.34 | 100.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 97.07 | 98.98 | 99.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 98.48 | 98.41 | 99.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:30:00 | 98.54 | 98.41 | 99.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 91.85 | 91.08 | 92.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 91.60 | 91.08 | 92.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:45:00 | 91.74 | 91.25 | 92.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 93.21 | 91.64 | 92.45 | SL hit (close>static) qty=1.00 sl=92.55 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 95.35 | 92.92 | 92.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 96.84 | 94.78 | 93.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 96.92 | 96.96 | 95.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 96.92 | 96.96 | 95.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 99.50 | 99.93 | 99.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 99.49 | 99.93 | 99.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 98.24 | 99.52 | 99.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 98.24 | 99.52 | 99.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 97.66 | 99.15 | 98.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 97.66 | 99.15 | 98.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 96.70 | 98.66 | 98.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 96.22 | 98.17 | 98.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 98.10 | 97.85 | 98.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 98.10 | 97.85 | 98.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 98.34 | 97.98 | 98.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 98.34 | 97.98 | 98.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 98.27 | 98.04 | 98.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 98.65 | 98.04 | 98.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 97.98 | 98.03 | 98.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 97.35 | 97.81 | 98.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 98.40 | 97.95 | 98.09 | SL hit (close>static) qty=1.00 sl=98.30 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 93.30 | 92.57 | 92.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 94.03 | 92.87 | 92.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 92.75 | 93.01 | 92.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 92.75 | 93.01 | 92.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 92.75 | 93.01 | 92.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 92.70 | 93.01 | 92.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 92.09 | 92.83 | 92.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 92.09 | 92.83 | 92.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 93.00 | 92.86 | 92.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 93.24 | 92.86 | 92.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 90.45 | 92.33 | 92.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 90.45 | 92.33 | 92.50 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 92.89 | 92.02 | 92.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 95.40 | 93.00 | 92.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 94.88 | 95.13 | 94.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 94.88 | 95.13 | 94.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 94.56 | 94.95 | 94.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 94.56 | 94.95 | 94.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 95.21 | 94.95 | 94.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 95.21 | 94.95 | 94.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 94.99 | 94.95 | 94.51 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 93.98 | 94.30 | 94.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 92.59 | 93.96 | 94.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 93.14 | 93.03 | 93.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 93.14 | 93.03 | 93.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 91.38 | 92.73 | 93.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 91.31 | 92.38 | 93.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:00:00 | 90.99 | 92.38 | 93.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 86.74 | 88.60 | 89.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:15:00 | 86.44 | 88.60 | 89.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 85.00 | 84.98 | 86.49 | SL hit (close>ema200) qty=0.50 sl=84.98 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 86.36 | 85.50 | 85.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 87.30 | 85.86 | 85.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 87.15 | 87.23 | 86.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 87.15 | 87.23 | 86.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 87.36 | 87.26 | 86.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 87.03 | 87.26 | 86.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 86.16 | 87.04 | 86.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:45:00 | 87.01 | 86.84 | 86.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 87.01 | 86.86 | 86.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 86.08 | 86.66 | 86.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 86.08 | 86.66 | 86.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 85.64 | 86.45 | 86.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 80.90 | 80.65 | 82.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 80.90 | 80.65 | 82.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 81.75 | 81.15 | 81.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 81.75 | 81.15 | 81.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 82.62 | 81.45 | 81.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 82.62 | 81.45 | 81.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 81.80 | 81.52 | 81.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:45:00 | 81.75 | 81.55 | 81.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:30:00 | 81.47 | 81.67 | 81.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 83.50 | 82.12 | 82.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 83.50 | 82.12 | 82.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 84.30 | 82.55 | 82.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 84.90 | 85.27 | 84.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 84.90 | 85.27 | 84.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 84.55 | 85.13 | 84.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 84.55 | 85.13 | 84.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 84.63 | 85.03 | 84.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 84.48 | 85.03 | 84.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 85.12 | 85.05 | 84.65 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 83.63 | 84.47 | 84.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 83.37 | 84.13 | 84.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 83.40 | 83.29 | 83.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 83.58 | 83.29 | 83.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 82.94 | 83.22 | 83.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 82.70 | 83.14 | 83.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:00:00 | 82.86 | 82.64 | 83.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 82.83 | 82.70 | 83.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:30:00 | 82.74 | 82.70 | 82.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 82.71 | 82.68 | 82.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 82.84 | 82.68 | 82.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 83.30 | 82.71 | 82.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 83.30 | 82.71 | 82.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 83.33 | 82.83 | 82.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 83.33 | 82.83 | 82.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 83.80 | 83.03 | 82.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 89.62 | 90.10 | 88.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 89.62 | 90.10 | 88.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 89.14 | 89.77 | 88.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 88.75 | 89.77 | 88.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 88.71 | 89.56 | 88.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 88.71 | 89.56 | 88.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 89.10 | 89.47 | 88.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 89.18 | 88.98 | 88.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 87.82 | 88.66 | 88.61 | SL hit (close<static) qty=1.00 sl=88.37 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 87.36 | 88.40 | 88.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 87.29 | 88.18 | 88.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 88.80 | 88.18 | 88.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 88.80 | 88.18 | 88.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 88.80 | 88.18 | 88.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:45:00 | 88.92 | 88.18 | 88.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 89.16 | 88.38 | 88.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 89.12 | 88.38 | 88.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 89.65 | 88.43 | 88.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 90.46 | 89.03 | 88.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 89.11 | 89.36 | 88.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 89.11 | 89.36 | 88.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 89.03 | 89.29 | 88.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 89.03 | 89.29 | 88.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 88.98 | 89.23 | 88.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 88.98 | 89.23 | 88.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 89.11 | 89.20 | 88.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 89.52 | 89.20 | 88.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 89.82 | 89.33 | 89.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 90.73 | 89.93 | 89.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:30:00 | 90.89 | 90.31 | 89.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 15:15:00 | 90.60 | 91.29 | 91.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 90.60 | 91.29 | 91.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 86.01 | 90.23 | 90.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 88.40 | 87.85 | 89.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 88.40 | 87.85 | 89.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 88.40 | 87.85 | 89.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 88.95 | 87.85 | 89.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 88.76 | 88.04 | 88.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 88.76 | 88.04 | 88.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 88.99 | 88.31 | 88.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 89.07 | 88.31 | 88.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 89.20 | 88.48 | 88.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 89.06 | 88.48 | 88.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 89.24 | 88.64 | 89.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 89.24 | 88.64 | 89.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 89.80 | 88.87 | 89.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 88.45 | 88.87 | 89.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 88.83 | 88.53 | 88.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:45:00 | 88.58 | 88.53 | 88.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 89.35 | 88.70 | 88.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:00:00 | 89.35 | 88.70 | 88.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 88.40 | 88.64 | 88.84 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 90.39 | 89.20 | 89.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 90.92 | 89.54 | 89.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 98.56 | 100.19 | 98.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 98.56 | 100.19 | 98.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 98.56 | 100.19 | 98.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 98.56 | 100.19 | 98.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 98.45 | 99.84 | 98.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 97.95 | 99.84 | 98.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 98.71 | 99.61 | 98.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 99.21 | 99.50 | 98.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:45:00 | 99.33 | 99.50 | 98.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 97.59 | 99.09 | 99.07 | SL hit (close<static) qty=1.00 sl=98.45 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 96.96 | 98.66 | 98.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 96.10 | 97.42 | 98.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 97.80 | 97.50 | 98.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:30:00 | 97.49 | 97.50 | 98.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 98.27 | 97.65 | 98.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 98.27 | 97.65 | 98.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 98.57 | 97.83 | 98.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 98.84 | 97.83 | 98.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 98.59 | 98.20 | 98.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 100.78 | 98.20 | 98.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 100.15 | 98.59 | 98.41 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 97.39 | 98.44 | 98.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 14:15:00 | 96.92 | 97.46 | 97.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 15:15:00 | 97.35 | 97.20 | 97.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:15:00 | 96.65 | 97.20 | 97.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 95.57 | 96.87 | 97.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 95.22 | 96.54 | 97.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 95.35 | 94.44 | 94.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 96.75 | 95.01 | 94.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 09:15:00 | 96.75 | 95.01 | 94.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 11:15:00 | 98.30 | 96.04 | 95.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 104.92 | 105.20 | 103.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 104.92 | 105.20 | 103.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 107.41 | 108.02 | 107.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 107.41 | 108.02 | 107.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 107.12 | 107.84 | 107.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:15:00 | 106.82 | 107.84 | 107.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 107.17 | 107.71 | 107.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 106.63 | 107.71 | 107.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 107.40 | 107.52 | 107.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 107.40 | 107.52 | 107.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 106.30 | 107.28 | 107.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 106.30 | 107.28 | 107.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 107.33 | 107.29 | 107.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 107.46 | 107.41 | 107.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 106.75 | 107.22 | 107.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 106.75 | 107.22 | 107.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 106.20 | 107.02 | 107.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 107.06 | 106.95 | 107.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 107.06 | 106.95 | 107.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 107.06 | 106.95 | 107.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 107.06 | 106.95 | 107.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 107.75 | 107.11 | 107.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 107.75 | 107.11 | 107.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 107.56 | 107.20 | 107.19 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 106.97 | 107.21 | 107.21 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 107.53 | 107.27 | 107.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 107.84 | 107.39 | 107.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 110.19 | 110.31 | 109.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:45:00 | 110.26 | 110.31 | 109.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 109.78 | 110.36 | 109.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 109.78 | 110.36 | 109.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 110.11 | 110.31 | 109.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 109.56 | 110.31 | 109.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 111.05 | 110.46 | 109.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 111.45 | 110.81 | 110.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 115.75 | 116.11 | 116.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 115.75 | 116.11 | 116.13 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 118.44 | 116.52 | 116.31 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 116.95 | 117.27 | 117.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 116.29 | 116.99 | 117.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 111.54 | 111.50 | 112.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 111.54 | 111.50 | 112.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 107.64 | 106.28 | 107.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 107.64 | 106.28 | 107.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 107.28 | 106.48 | 107.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 107.01 | 106.48 | 107.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 106.89 | 106.82 | 107.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 107.80 | 107.56 | 107.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 107.80 | 107.56 | 107.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 110.72 | 108.19 | 107.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 109.63 | 110.45 | 109.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 109.63 | 110.45 | 109.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 109.63 | 110.45 | 109.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 109.72 | 110.45 | 109.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 109.03 | 110.17 | 109.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 109.03 | 110.17 | 109.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 109.72 | 110.05 | 109.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 109.48 | 110.05 | 109.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 109.90 | 110.02 | 109.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 109.85 | 110.02 | 109.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 111.12 | 110.24 | 109.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 112.20 | 110.39 | 110.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 111.30 | 110.57 | 110.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 111.27 | 111.22 | 110.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 113.57 | 114.23 | 114.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 113.57 | 114.23 | 114.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 113.10 | 114.00 | 114.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 114.25 | 113.85 | 114.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 10:15:00 | 114.25 | 113.85 | 114.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 114.25 | 113.85 | 114.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 114.25 | 113.85 | 114.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 113.89 | 113.86 | 113.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 113.22 | 113.85 | 113.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 114.58 | 112.61 | 112.81 | SL hit (close>static) qty=1.00 sl=114.25 alert=retest2 |

### Cycle 79 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 113.83 | 113.10 | 113.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 115.10 | 113.78 | 113.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 116.29 | 116.29 | 115.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 116.21 | 116.29 | 115.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 115.41 | 116.01 | 115.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 115.41 | 116.01 | 115.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 115.30 | 115.87 | 115.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:15:00 | 115.22 | 115.87 | 115.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 115.38 | 115.77 | 115.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:45:00 | 115.57 | 115.73 | 115.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 115.76 | 115.66 | 115.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 115.56 | 115.43 | 115.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 12:15:00 | 114.52 | 115.25 | 115.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 114.52 | 115.25 | 115.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 113.05 | 114.51 | 114.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 114.55 | 114.38 | 114.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:45:00 | 114.58 | 114.38 | 114.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 112.07 | 113.92 | 114.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 111.81 | 113.92 | 114.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 110.52 | 113.25 | 114.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 113.38 | 109.84 | 109.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 113.38 | 109.84 | 109.73 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 110.41 | 110.78 | 110.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 109.80 | 110.58 | 110.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 110.60 | 110.59 | 110.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 110.41 | 110.59 | 110.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 110.31 | 110.53 | 110.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 13:15:00 | 109.92 | 110.53 | 110.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 14:30:00 | 109.89 | 110.38 | 110.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 108.05 | 107.62 | 107.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 108.05 | 107.62 | 107.58 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 107.01 | 107.55 | 107.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 106.95 | 107.43 | 107.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 107.43 | 107.27 | 107.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 11:15:00 | 107.43 | 107.27 | 107.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 107.43 | 107.27 | 107.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 107.43 | 107.27 | 107.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 109.14 | 107.65 | 107.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 109.92 | 109.35 | 108.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 109.66 | 109.68 | 109.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 109.67 | 109.68 | 109.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 109.06 | 109.55 | 109.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 109.06 | 109.55 | 109.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 108.84 | 109.41 | 109.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 108.84 | 109.41 | 109.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 108.98 | 109.32 | 109.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:30:00 | 109.21 | 109.14 | 109.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 109.69 | 109.14 | 109.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 109.29 | 109.20 | 109.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 109.23 | 109.20 | 109.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 109.30 | 109.22 | 109.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 109.31 | 109.22 | 109.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 109.15 | 109.21 | 109.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 109.75 | 109.21 | 109.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:15:00 | 109.64 | 109.47 | 109.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 110.45 | 111.13 | 111.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 110.45 | 111.13 | 111.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 110.25 | 110.85 | 111.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 104.63 | 104.41 | 105.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 105.01 | 104.41 | 105.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 105.07 | 104.74 | 105.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 105.07 | 104.74 | 105.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 105.20 | 104.83 | 105.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 104.83 | 104.83 | 105.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 106.19 | 105.10 | 105.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 106.19 | 105.10 | 105.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 106.48 | 105.38 | 105.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 106.62 | 105.38 | 105.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 106.84 | 105.67 | 105.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 107.35 | 106.28 | 105.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 107.47 | 107.53 | 106.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 107.47 | 107.53 | 106.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 106.85 | 107.36 | 106.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 106.85 | 107.36 | 106.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 106.55 | 107.19 | 106.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 106.55 | 107.19 | 106.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 106.28 | 107.01 | 106.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 106.28 | 107.01 | 106.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 106.30 | 106.63 | 106.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 106.15 | 106.52 | 106.60 | Break + close below crossover candle low |

### Cycle 89 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 107.23 | 106.66 | 106.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 108.05 | 107.08 | 106.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 14:15:00 | 107.99 | 108.06 | 107.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 107.99 | 108.06 | 107.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 110.95 | 111.53 | 110.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:45:00 | 110.99 | 111.53 | 110.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 111.00 | 111.42 | 110.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 111.00 | 111.42 | 110.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 111.20 | 111.38 | 110.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 111.08 | 111.38 | 110.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 111.03 | 111.31 | 110.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 110.89 | 111.31 | 110.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 110.93 | 111.23 | 110.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 112.31 | 111.23 | 110.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-24 09:15:00 | 123.54 | 121.13 | 119.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 119.24 | 120.88 | 121.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 118.97 | 120.50 | 120.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 120.75 | 119.63 | 120.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 120.75 | 119.63 | 120.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 120.75 | 119.63 | 120.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 120.93 | 119.63 | 120.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 120.20 | 119.74 | 120.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 121.20 | 119.74 | 120.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 119.98 | 119.79 | 120.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 119.98 | 119.79 | 120.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 121.35 | 120.10 | 120.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 121.35 | 120.10 | 120.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 121.00 | 120.28 | 120.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 120.53 | 120.33 | 120.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 120.85 | 120.44 | 120.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 120.85 | 120.44 | 120.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 124.05 | 121.16 | 120.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 121.83 | 123.01 | 122.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 121.83 | 123.01 | 122.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 121.83 | 123.01 | 122.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 121.62 | 123.01 | 122.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 121.37 | 122.68 | 122.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 121.37 | 122.68 | 122.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 123.00 | 122.74 | 122.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:15:00 | 123.24 | 122.74 | 122.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 125.55 | 126.83 | 126.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 125.55 | 126.83 | 126.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 124.95 | 126.45 | 126.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 126.57 | 125.90 | 126.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 126.57 | 125.90 | 126.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 126.57 | 125.90 | 126.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:15:00 | 126.78 | 125.90 | 126.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 127.44 | 126.21 | 126.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 127.44 | 126.21 | 126.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 127.79 | 126.72 | 126.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 127.99 | 126.97 | 126.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 14:15:00 | 127.37 | 127.84 | 127.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 14:15:00 | 127.37 | 127.84 | 127.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 127.37 | 127.84 | 127.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 127.48 | 127.84 | 127.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 127.46 | 127.76 | 127.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 126.92 | 127.76 | 127.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 127.27 | 127.38 | 127.32 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 126.30 | 127.16 | 127.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 125.08 | 126.75 | 127.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 127.44 | 126.39 | 126.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 127.44 | 126.39 | 126.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 127.44 | 126.39 | 126.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 127.18 | 126.39 | 126.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 128.10 | 126.73 | 126.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 128.10 | 126.73 | 126.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 128.94 | 127.17 | 127.03 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 11:15:00 | 126.85 | 127.21 | 127.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 126.01 | 126.85 | 127.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 127.58 | 126.33 | 126.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 127.58 | 126.33 | 126.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 127.58 | 126.33 | 126.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 127.23 | 126.33 | 126.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 128.00 | 126.67 | 126.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 128.00 | 126.67 | 126.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 128.19 | 126.97 | 126.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 128.58 | 127.50 | 127.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 128.00 | 128.22 | 127.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 128.00 | 128.22 | 127.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 129.60 | 128.50 | 127.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 129.95 | 128.76 | 127.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:45:00 | 130.00 | 128.96 | 128.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 130.05 | 129.17 | 128.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 132.25 | 129.94 | 129.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 136.83 | 138.92 | 138.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 137.08 | 138.92 | 138.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 139.62 | 139.06 | 138.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:30:00 | 139.88 | 139.46 | 138.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 140.00 | 139.91 | 139.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-12 09:15:00 | 142.94 | 141.09 | 140.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 146.16 | 148.50 | 148.73 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 148.93 | 147.67 | 147.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 151.66 | 148.78 | 148.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 150.25 | 150.36 | 149.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 150.20 | 150.36 | 149.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 152.12 | 151.59 | 150.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 152.19 | 151.59 | 150.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 151.20 | 151.54 | 151.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 151.20 | 151.54 | 151.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 151.45 | 151.53 | 151.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 151.29 | 151.53 | 151.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 150.51 | 151.32 | 151.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 150.04 | 151.32 | 151.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 150.70 | 151.20 | 151.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 149.94 | 151.20 | 151.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 152.18 | 152.65 | 151.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 151.96 | 152.65 | 151.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 152.18 | 152.56 | 152.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 150.47 | 152.56 | 152.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 149.07 | 151.86 | 151.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 149.07 | 151.86 | 151.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 147.08 | 150.90 | 151.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 145.77 | 149.29 | 150.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 147.68 | 147.63 | 149.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:30:00 | 148.71 | 147.63 | 149.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 148.19 | 147.55 | 148.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 148.19 | 147.55 | 148.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 149.08 | 147.86 | 148.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 147.18 | 148.32 | 148.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 146.75 | 147.86 | 148.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:45:00 | 146.87 | 145.73 | 145.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 147.21 | 146.00 | 145.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 147.21 | 146.00 | 145.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 148.19 | 147.03 | 146.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 147.18 | 147.68 | 147.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 147.18 | 147.68 | 147.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 147.18 | 147.68 | 147.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 147.18 | 147.68 | 147.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 146.81 | 147.50 | 147.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 146.81 | 147.50 | 147.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 147.08 | 147.42 | 147.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 147.35 | 147.42 | 147.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 147.31 | 147.31 | 147.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:15:00 | 147.28 | 147.31 | 147.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 147.49 | 147.27 | 147.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 147.05 | 147.23 | 147.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 150.00 | 147.23 | 147.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 147.60 | 148.89 | 148.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 11:15:00 | 147.60 | 148.89 | 148.91 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 149.47 | 148.92 | 148.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 150.17 | 149.33 | 149.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 150.15 | 150.15 | 149.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 150.55 | 150.15 | 149.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 150.28 | 150.18 | 149.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:30:00 | 150.87 | 150.19 | 149.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 14:15:00 | 149.50 | 149.96 | 149.86 | SL hit (close<static) qty=1.00 sl=149.69 alert=retest2 |

### Cycle 104 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 154.18 | 154.55 | 154.59 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 155.65 | 154.77 | 154.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 155.94 | 155.00 | 154.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 12:15:00 | 153.93 | 154.79 | 154.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 12:15:00 | 153.93 | 154.79 | 154.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 153.93 | 154.79 | 154.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 153.93 | 154.79 | 154.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 152.20 | 154.27 | 154.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 151.17 | 153.15 | 153.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 152.26 | 151.64 | 152.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 152.26 | 151.64 | 152.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 151.84 | 151.68 | 152.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 152.06 | 151.68 | 152.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 150.92 | 150.06 | 150.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 150.76 | 150.06 | 150.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 150.95 | 150.24 | 150.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 150.34 | 150.18 | 150.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 153.08 | 150.98 | 150.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 153.08 | 150.98 | 150.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 154.25 | 151.80 | 151.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 156.65 | 156.67 | 155.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 156.65 | 156.67 | 155.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 155.87 | 156.56 | 155.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 155.87 | 156.56 | 155.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 155.04 | 156.25 | 155.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 155.62 | 156.25 | 155.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 154.43 | 155.89 | 155.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 154.43 | 155.89 | 155.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 154.16 | 155.54 | 155.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:45:00 | 154.31 | 155.54 | 155.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 153.18 | 155.07 | 155.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 151.36 | 153.77 | 154.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 154.75 | 152.45 | 153.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 154.75 | 152.45 | 153.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 154.75 | 152.45 | 153.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 154.74 | 152.45 | 153.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 154.70 | 152.90 | 153.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 154.70 | 152.90 | 153.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 154.53 | 153.78 | 153.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 155.10 | 154.22 | 153.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 154.72 | 154.78 | 154.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 154.72 | 154.78 | 154.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 152.58 | 154.34 | 154.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 152.58 | 154.34 | 154.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 152.25 | 153.92 | 153.98 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 155.00 | 153.80 | 153.77 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 153.09 | 153.66 | 153.71 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 155.14 | 154.02 | 153.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 155.38 | 154.29 | 154.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 12:15:00 | 152.07 | 156.03 | 155.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 12:15:00 | 152.07 | 156.03 | 155.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 152.07 | 156.03 | 155.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 152.07 | 156.03 | 155.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 149.82 | 154.79 | 154.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 148.10 | 152.20 | 153.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 143.94 | 143.14 | 146.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:00:00 | 143.94 | 143.14 | 146.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 145.42 | 143.85 | 145.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 145.38 | 143.85 | 145.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 146.85 | 144.45 | 145.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 146.85 | 144.45 | 145.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 146.50 | 144.86 | 146.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 149.11 | 144.86 | 146.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 148.84 | 146.87 | 146.74 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 146.08 | 147.27 | 147.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 145.18 | 146.85 | 147.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 147.29 | 146.84 | 147.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 147.29 | 146.84 | 147.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 147.29 | 146.84 | 147.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 147.29 | 146.84 | 147.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 147.00 | 146.87 | 147.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 149.75 | 146.87 | 147.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 149.27 | 147.35 | 147.31 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 147.23 | 147.58 | 147.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 146.74 | 147.31 | 147.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 144.94 | 144.87 | 145.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 144.94 | 144.87 | 145.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 143.99 | 142.86 | 143.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 143.99 | 142.86 | 143.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 143.70 | 143.03 | 143.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:15:00 | 144.28 | 143.03 | 143.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 144.21 | 143.26 | 143.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 144.21 | 143.26 | 143.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 144.45 | 143.50 | 143.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 144.63 | 143.50 | 143.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 145.80 | 144.20 | 144.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 146.10 | 144.58 | 144.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 150.54 | 151.03 | 149.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 150.54 | 151.03 | 149.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 149.54 | 150.73 | 149.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 149.54 | 150.73 | 149.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 149.12 | 150.41 | 149.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 150.30 | 150.41 | 149.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 156.83 | 157.74 | 156.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 156.80 | 157.74 | 156.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 157.25 | 157.64 | 156.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:15:00 | 157.42 | 157.64 | 156.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:00:00 | 157.47 | 157.61 | 157.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:00:00 | 157.32 | 157.89 | 157.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 157.32 | 157.75 | 157.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 158.18 | 157.84 | 157.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 158.65 | 157.91 | 157.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 158.51 | 157.81 | 157.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 154.56 | 157.12 | 157.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 154.56 | 157.12 | 157.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 148.36 | 153.36 | 155.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 148.16 | 147.64 | 149.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 149.12 | 147.64 | 149.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 141.28 | 140.94 | 143.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 14:45:00 | 139.89 | 140.86 | 141.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 11:15:00 | 140.05 | 140.07 | 141.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 140.11 | 140.68 | 141.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 138.61 | 140.65 | 141.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 132.90 | 135.94 | 137.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 133.05 | 135.94 | 137.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 133.10 | 135.94 | 137.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 131.68 | 135.28 | 137.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 134.63 | 133.99 | 136.06 | SL hit (close>ema200) qty=0.50 sl=133.99 alert=retest2 |

### Cycle 121 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 137.31 | 136.00 | 135.95 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 135.35 | 136.16 | 136.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 134.35 | 135.80 | 136.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 138.79 | 135.47 | 135.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 138.79 | 135.47 | 135.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 138.79 | 135.47 | 135.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 139.03 | 135.47 | 135.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 138.98 | 136.17 | 136.01 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 130.55 | 135.42 | 135.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 130.29 | 134.39 | 135.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 132.09 | 131.65 | 133.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 132.09 | 131.65 | 133.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 132.09 | 131.65 | 133.28 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 136.74 | 133.60 | 133.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 137.15 | 134.80 | 134.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 131.53 | 135.07 | 134.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 131.53 | 135.07 | 134.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 131.53 | 135.07 | 134.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 131.63 | 135.07 | 134.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 131.68 | 134.39 | 134.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 131.46 | 134.39 | 134.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 130.91 | 133.69 | 134.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 130.19 | 131.90 | 133.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 127.02 | 126.35 | 128.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 127.02 | 126.35 | 128.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 127.02 | 126.35 | 128.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 126.67 | 126.35 | 128.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 123.02 | 127.39 | 128.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 129.84 | 127.47 | 127.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 129.84 | 127.47 | 127.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 130.12 | 128.00 | 127.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 128.73 | 128.98 | 128.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 128.73 | 128.98 | 128.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 136.95 | 139.08 | 137.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 137.41 | 139.08 | 137.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 137.70 | 138.84 | 137.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 141.46 | 143.97 | 144.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 141.46 | 143.97 | 144.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 139.50 | 141.85 | 142.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 140.66 | 140.53 | 141.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:45:00 | 140.72 | 140.53 | 141.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 140.75 | 140.68 | 141.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:30:00 | 140.45 | 140.60 | 141.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 140.61 | 140.52 | 141.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 133.58 | 136.28 | 137.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 137.00 | 135.41 | 136.45 | SL hit (close>ema200) qty=0.50 sl=135.41 alert=retest2 |

### Cycle 129 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 137.80 | 135.88 | 135.68 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 135.12 | 135.85 | 135.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 134.61 | 135.60 | 135.77 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 09:15:00 | 118.10 | 2024-05-27 09:15:00 | 116.15 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-05-27 12:15:00 | 117.85 | 2024-05-28 11:15:00 | 116.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-05-27 13:15:00 | 117.55 | 2024-05-28 11:15:00 | 116.20 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-05-28 09:15:00 | 118.30 | 2024-05-28 11:15:00 | 116.20 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-30 10:00:00 | 115.80 | 2024-05-31 11:15:00 | 116.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-05-30 11:30:00 | 115.65 | 2024-05-31 12:15:00 | 117.55 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-05-30 13:15:00 | 115.90 | 2024-05-31 12:15:00 | 117.55 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-05-30 14:15:00 | 115.85 | 2024-05-31 12:15:00 | 117.55 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-05-31 09:45:00 | 114.85 | 2024-05-31 12:15:00 | 117.55 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-06-24 14:45:00 | 118.43 | 2024-06-28 09:15:00 | 121.52 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-06-25 10:00:00 | 118.46 | 2024-06-28 09:15:00 | 121.52 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-06-26 14:30:00 | 118.85 | 2024-06-28 09:15:00 | 121.52 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-06-27 10:00:00 | 118.90 | 2024-06-28 09:15:00 | 121.52 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-06-27 12:15:00 | 117.92 | 2024-06-28 09:15:00 | 121.52 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-07-04 10:15:00 | 116.74 | 2024-07-05 11:15:00 | 117.66 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-07-09 14:15:00 | 115.85 | 2024-07-15 13:15:00 | 117.01 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 114.83 | 2024-07-23 12:15:00 | 109.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 14:00:00 | 114.75 | 2024-07-23 12:15:00 | 109.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 12:15:00 | 114.83 | 2024-07-24 15:15:00 | 112.73 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest2 | 2024-07-22 14:00:00 | 114.75 | 2024-07-24 15:15:00 | 112.73 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2024-07-22 15:00:00 | 114.70 | 2024-07-26 14:15:00 | 113.79 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2024-07-23 11:15:00 | 114.34 | 2024-07-26 14:15:00 | 113.79 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-07-23 12:15:00 | 110.74 | 2024-07-26 14:15:00 | 113.79 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-07-26 14:15:00 | 113.57 | 2024-07-26 14:15:00 | 113.79 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-08-01 09:30:00 | 114.94 | 2024-08-01 10:15:00 | 114.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-08-01 10:15:00 | 115.06 | 2024-08-01 10:15:00 | 114.63 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-06 13:30:00 | 106.26 | 2024-08-07 13:15:00 | 107.56 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-08 13:15:00 | 107.24 | 2024-08-09 09:15:00 | 108.22 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-08-08 13:45:00 | 107.27 | 2024-08-09 09:15:00 | 108.22 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-08-08 14:45:00 | 107.46 | 2024-08-09 09:15:00 | 108.22 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-08-16 10:15:00 | 106.70 | 2024-08-19 09:15:00 | 110.20 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-09-11 10:00:00 | 103.40 | 2024-09-13 09:15:00 | 105.19 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-09-11 10:30:00 | 103.49 | 2024-09-13 09:15:00 | 105.19 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-09-11 11:15:00 | 103.48 | 2024-09-13 09:15:00 | 105.19 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-09-26 15:15:00 | 110.30 | 2024-10-03 09:15:00 | 109.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-10-09 11:45:00 | 104.37 | 2024-10-10 09:15:00 | 106.21 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-10-10 13:30:00 | 104.60 | 2024-10-18 15:15:00 | 104.60 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-10-14 11:15:00 | 104.52 | 2024-10-18 15:15:00 | 104.60 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-10-15 10:15:00 | 104.53 | 2024-10-18 15:15:00 | 104.60 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-10-16 11:00:00 | 103.83 | 2024-10-18 15:15:00 | 104.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-16 14:00:00 | 103.92 | 2024-10-18 15:15:00 | 104.60 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-10-17 09:30:00 | 103.57 | 2024-10-18 15:15:00 | 104.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-24 12:15:00 | 98.45 | 2024-10-25 10:15:00 | 93.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:45:00 | 98.16 | 2024-10-25 10:15:00 | 93.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 14:15:00 | 98.29 | 2024-10-25 10:15:00 | 93.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:15:00 | 98.45 | 2024-10-28 09:15:00 | 96.14 | STOP_HIT | 0.50 | 2.35% |
| SELL | retest2 | 2024-10-24 12:45:00 | 98.16 | 2024-10-28 09:15:00 | 96.14 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2024-10-24 14:15:00 | 98.29 | 2024-10-28 09:15:00 | 96.14 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2024-10-28 12:15:00 | 98.45 | 2024-10-28 13:15:00 | 99.75 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-10-31 15:15:00 | 103.02 | 2024-11-04 09:15:00 | 101.67 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-11-19 11:45:00 | 100.15 | 2024-11-19 12:15:00 | 99.74 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest1 | 2024-12-09 12:45:00 | 110.10 | 2024-12-11 13:15:00 | 108.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2024-12-10 14:45:00 | 109.88 | 2024-12-11 13:15:00 | 108.85 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-16 12:00:00 | 106.77 | 2024-12-19 09:15:00 | 101.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:45:00 | 106.79 | 2024-12-19 09:15:00 | 101.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 106.83 | 2024-12-19 09:15:00 | 101.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:00:00 | 106.70 | 2024-12-19 09:15:00 | 101.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:00:00 | 106.77 | 2024-12-19 11:15:00 | 103.75 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2024-12-16 12:45:00 | 106.79 | 2024-12-19 11:15:00 | 103.75 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2024-12-17 09:15:00 | 106.83 | 2024-12-19 11:15:00 | 103.75 | STOP_HIT | 0.50 | 2.88% |
| SELL | retest2 | 2024-12-17 10:00:00 | 106.70 | 2024-12-19 11:15:00 | 103.75 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2024-12-26 12:15:00 | 100.75 | 2024-12-27 09:15:00 | 101.59 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-26 13:45:00 | 100.75 | 2024-12-27 09:15:00 | 101.59 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-26 14:45:00 | 100.79 | 2024-12-27 09:15:00 | 101.59 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-27 14:45:00 | 100.50 | 2025-01-01 15:15:00 | 100.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-12-30 13:15:00 | 99.50 | 2025-01-01 15:15:00 | 100.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-12-31 11:15:00 | 99.64 | 2025-01-01 15:15:00 | 100.60 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-12-31 14:15:00 | 99.84 | 2025-01-01 15:15:00 | 100.60 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-01 09:30:00 | 99.74 | 2025-01-01 15:15:00 | 100.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-01-14 12:15:00 | 91.60 | 2025-01-14 13:15:00 | 93.21 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-01-14 12:45:00 | 91.74 | 2025-01-14 13:15:00 | 93.21 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-01-24 09:30:00 | 97.35 | 2025-01-24 11:15:00 | 98.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-24 14:00:00 | 97.11 | 2025-01-27 14:15:00 | 92.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:00:00 | 97.11 | 2025-01-31 10:15:00 | 92.47 | STOP_HIT | 0.50 | 4.78% |
| BUY | retest2 | 2025-02-01 14:15:00 | 93.24 | 2025-02-03 09:15:00 | 90.45 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-02-11 10:30:00 | 91.31 | 2025-02-14 10:15:00 | 86.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:00:00 | 90.99 | 2025-02-14 10:15:00 | 86.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 10:30:00 | 91.31 | 2025-02-17 14:15:00 | 85.00 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest2 | 2025-02-11 11:00:00 | 90.99 | 2025-02-17 14:15:00 | 85.00 | STOP_HIT | 0.50 | 6.58% |
| BUY | retest2 | 2025-02-24 11:45:00 | 87.01 | 2025-02-25 10:15:00 | 86.08 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-02-24 14:30:00 | 87.01 | 2025-02-25 10:15:00 | 86.08 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-03-04 12:45:00 | 81.75 | 2025-03-05 09:15:00 | 83.50 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-03-04 13:30:00 | 81.47 | 2025-03-05 09:15:00 | 83.50 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-03-12 10:45:00 | 82.70 | 2025-03-18 10:15:00 | 83.33 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-03-13 10:00:00 | 82.86 | 2025-03-18 10:15:00 | 83.33 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-03-13 11:15:00 | 82.83 | 2025-03-18 10:15:00 | 83.33 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-03-17 09:30:00 | 82.74 | 2025-03-18 10:15:00 | 83.33 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-03-26 10:45:00 | 89.18 | 2025-03-26 12:15:00 | 87.82 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-04-02 12:15:00 | 90.73 | 2025-04-04 15:15:00 | 90.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-04-02 14:30:00 | 90.89 | 2025-04-04 15:15:00 | 90.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-04-23 13:15:00 | 99.21 | 2025-04-25 09:15:00 | 97.59 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-04-23 13:45:00 | 99.33 | 2025-04-25 09:15:00 | 97.59 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-05-06 11:00:00 | 95.22 | 2025-05-09 09:15:00 | 96.75 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-05-08 14:45:00 | 95.35 | 2025-05-09 09:15:00 | 96.75 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-05-21 14:30:00 | 107.46 | 2025-05-22 12:15:00 | 106.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-30 12:45:00 | 111.45 | 2025-06-06 14:15:00 | 115.75 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2025-06-20 12:15:00 | 107.01 | 2025-06-23 15:15:00 | 107.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-06-23 09:15:00 | 106.89 | 2025-06-23 15:15:00 | 107.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-06-27 09:15:00 | 112.20 | 2025-07-08 10:15:00 | 113.57 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-06-27 10:00:00 | 111.30 | 2025-07-08 10:15:00 | 113.57 | STOP_HIT | 1.00 | 2.04% |
| BUY | retest2 | 2025-06-27 15:15:00 | 111.27 | 2025-07-08 10:15:00 | 113.57 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2025-07-10 09:30:00 | 113.22 | 2025-07-14 09:15:00 | 114.58 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-17 14:45:00 | 115.57 | 2025-07-18 12:15:00 | 114.52 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-18 09:15:00 | 115.76 | 2025-07-18 12:15:00 | 114.52 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-18 11:30:00 | 115.56 | 2025-07-18 12:15:00 | 114.52 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-07-21 14:15:00 | 111.81 | 2025-07-24 14:15:00 | 113.38 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-07-22 09:15:00 | 110.52 | 2025-07-24 14:15:00 | 113.38 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-07-29 13:15:00 | 109.92 | 2025-08-05 11:15:00 | 108.05 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2025-07-29 14:30:00 | 109.89 | 2025-08-05 11:15:00 | 108.05 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2025-08-14 09:30:00 | 109.21 | 2025-08-22 11:15:00 | 110.45 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-08-14 10:15:00 | 109.69 | 2025-08-22 11:15:00 | 110.45 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-08-14 11:45:00 | 109.29 | 2025-08-22 11:15:00 | 110.45 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2025-08-14 13:30:00 | 109.23 | 2025-08-22 11:15:00 | 110.45 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2025-08-18 09:15:00 | 109.75 | 2025-08-22 11:15:00 | 110.45 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-08-18 13:15:00 | 109.64 | 2025-08-22 11:15:00 | 110.45 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-09-15 09:15:00 | 112.31 | 2025-09-24 09:15:00 | 123.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-29 15:00:00 | 120.53 | 2025-09-29 15:15:00 | 120.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-10-01 13:15:00 | 123.24 | 2025-10-14 10:15:00 | 125.55 | STOP_HIT | 1.00 | 1.87% |
| BUY | retest2 | 2025-10-28 12:30:00 | 129.95 | 2025-11-12 09:15:00 | 142.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-28 13:45:00 | 130.00 | 2025-11-12 09:15:00 | 143.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-28 15:00:00 | 130.05 | 2025-11-12 09:15:00 | 143.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 12:30:00 | 132.25 | 2025-11-14 09:15:00 | 145.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 12:30:00 | 139.88 | 2025-11-21 09:15:00 | 146.16 | STOP_HIT | 1.00 | 4.49% |
| BUY | retest2 | 2025-11-11 12:30:00 | 140.00 | 2025-11-21 09:15:00 | 146.16 | STOP_HIT | 1.00 | 4.40% |
| SELL | retest2 | 2025-12-08 09:15:00 | 147.18 | 2025-12-11 09:15:00 | 147.21 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-12-08 09:45:00 | 146.75 | 2025-12-11 09:15:00 | 147.21 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-10 09:45:00 | 146.87 | 2025-12-11 09:15:00 | 147.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-12-16 12:15:00 | 147.35 | 2025-12-19 11:15:00 | 147.60 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-12-16 13:30:00 | 147.31 | 2025-12-19 11:15:00 | 147.60 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-12-16 14:15:00 | 147.28 | 2025-12-19 11:15:00 | 147.60 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-12-16 14:45:00 | 147.49 | 2025-12-19 11:15:00 | 147.60 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-12-17 09:15:00 | 150.00 | 2025-12-19 11:15:00 | 147.60 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-12-24 10:30:00 | 150.87 | 2025-12-24 14:15:00 | 149.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-29 13:45:00 | 151.15 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-12-29 15:15:00 | 151.20 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2025-12-30 10:00:00 | 151.37 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-01-02 11:45:00 | 154.97 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-01-02 15:00:00 | 154.57 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-01-05 14:45:00 | 154.84 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-06 09:15:00 | 155.84 | 2026-01-06 15:15:00 | 154.18 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-13 11:45:00 | 150.34 | 2026-01-14 10:15:00 | 153.08 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-26 13:15:00 | 157.42 | 2026-03-02 09:15:00 | 154.56 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-26 14:00:00 | 157.47 | 2026-03-02 09:15:00 | 154.56 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-02-27 10:00:00 | 157.32 | 2026-03-02 09:15:00 | 154.56 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-27 11:45:00 | 157.32 | 2026-03-02 09:15:00 | 154.56 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-27 13:45:00 | 158.65 | 2026-03-02 09:15:00 | 154.56 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-02-27 14:30:00 | 158.51 | 2026-03-02 09:15:00 | 154.56 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-03-11 14:45:00 | 139.89 | 2026-03-16 09:15:00 | 132.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 11:15:00 | 140.05 | 2026-03-16 09:15:00 | 133.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:45:00 | 140.11 | 2026-03-16 09:15:00 | 133.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 138.61 | 2026-03-16 10:15:00 | 131.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:45:00 | 139.89 | 2026-03-16 14:15:00 | 134.63 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2026-03-12 11:15:00 | 140.05 | 2026-03-16 14:15:00 | 134.63 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2026-03-12 14:45:00 | 140.11 | 2026-03-16 14:15:00 | 134.63 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2026-03-13 09:15:00 | 138.61 | 2026-03-16 14:15:00 | 134.63 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2026-04-01 10:15:00 | 126.67 | 2026-04-06 12:15:00 | 129.84 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-02 09:15:00 | 123.02 | 2026-04-06 12:15:00 | 129.84 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2026-04-13 10:15:00 | 137.41 | 2026-04-23 12:15:00 | 141.46 | STOP_HIT | 1.00 | 2.95% |
| BUY | retest2 | 2026-04-13 10:45:00 | 137.70 | 2026-04-23 12:15:00 | 141.46 | STOP_HIT | 1.00 | 2.73% |
| SELL | retest2 | 2026-04-27 10:30:00 | 140.45 | 2026-04-30 10:15:00 | 133.58 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2026-04-27 10:30:00 | 140.45 | 2026-05-04 09:15:00 | 137.00 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2026-04-27 14:00:00 | 140.61 | 2026-05-05 09:15:00 | 133.43 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2026-04-27 14:00:00 | 140.61 | 2026-05-05 12:15:00 | 134.81 | STOP_HIT | 0.50 | 4.12% |
