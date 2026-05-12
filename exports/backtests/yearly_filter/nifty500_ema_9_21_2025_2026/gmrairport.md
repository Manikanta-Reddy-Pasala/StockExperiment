# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 101.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 50 |
| ALERT2 | 49 |
| ALERT2_SKIP | 19 |
| ALERT3 | 121 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 36 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 18
- **Target hits / Stop hits / Partials:** 0 / 37 / 2
- **Avg / median % per leg:** 0.37% / 0.30%
- **Sum % (uncompounded):** 14.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 10 | 47.6% | 0 | 21 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.90% | -2.9% |
| BUY @ 3rd Alert (retest2) | 20 | 10 | 50.0% | 0 | 20 | 0 | 0.15% | 2.9% |
| SELL (all) | 18 | 11 | 61.1% | 0 | 16 | 2 | 0.79% | 14.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 11 | 61.1% | 0 | 16 | 2 | 0.79% | 14.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.90% | -2.9% |
| retest2 (combined) | 38 | 21 | 55.3% | 0 | 36 | 2 | 0.45% | 17.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 88.70 | 86.10 | 85.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 10:15:00 | 89.00 | 88.12 | 87.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 89.96 | 89.98 | 89.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 89.91 | 89.98 | 89.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 89.15 | 89.71 | 89.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 89.15 | 89.71 | 89.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 88.80 | 89.53 | 89.16 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 87.46 | 88.86 | 88.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 87.05 | 87.83 | 88.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 87.06 | 87.01 | 87.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 87.06 | 87.01 | 87.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 86.65 | 87.01 | 87.59 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 88.30 | 87.75 | 87.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 88.69 | 88.10 | 87.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 09:15:00 | 86.93 | 88.04 | 87.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 86.93 | 88.04 | 87.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 86.93 | 88.04 | 87.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 86.97 | 88.04 | 87.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 87.16 | 87.87 | 87.86 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 11:15:00 | 87.28 | 87.75 | 87.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 86.97 | 87.59 | 87.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 86.89 | 86.80 | 87.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 86.89 | 86.80 | 87.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 86.89 | 86.80 | 87.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 85.85 | 86.33 | 86.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:00:00 | 85.95 | 86.26 | 86.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:00:00 | 86.05 | 86.22 | 86.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 85.96 | 86.26 | 86.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 86.15 | 86.24 | 86.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 85.76 | 86.30 | 86.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 85.50 | 84.90 | 84.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 85.50 | 84.90 | 84.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 86.27 | 85.35 | 85.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 86.35 | 86.35 | 85.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 85.91 | 86.35 | 85.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 86.10 | 86.30 | 85.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 85.98 | 86.30 | 85.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 85.78 | 86.20 | 85.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 85.78 | 86.20 | 85.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 85.60 | 86.08 | 85.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 85.60 | 86.08 | 85.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 85.48 | 85.92 | 85.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 85.48 | 85.92 | 85.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 85.11 | 85.76 | 85.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 84.27 | 85.24 | 85.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 81.96 | 81.55 | 82.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:30:00 | 81.90 | 81.55 | 82.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 82.32 | 81.70 | 82.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 82.36 | 81.70 | 82.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 82.75 | 81.91 | 82.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 82.75 | 81.91 | 82.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 83.55 | 82.24 | 82.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 83.55 | 82.24 | 82.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 83.80 | 82.75 | 82.75 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 82.64 | 82.82 | 82.82 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 83.24 | 82.90 | 82.86 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 82.23 | 82.74 | 82.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 81.61 | 82.52 | 82.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 81.49 | 81.30 | 81.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 81.49 | 81.30 | 81.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 81.42 | 81.17 | 81.52 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 83.17 | 81.75 | 81.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 83.60 | 82.54 | 82.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 84.88 | 84.90 | 84.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 84.88 | 84.90 | 84.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 84.88 | 84.90 | 84.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 84.88 | 84.90 | 84.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 84.64 | 84.86 | 84.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 84.70 | 84.86 | 84.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 84.73 | 84.84 | 84.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 84.51 | 84.84 | 84.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 92.10 | 92.24 | 91.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 92.39 | 92.24 | 91.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 91.45 | 92.08 | 91.80 | SL hit (close<static) qty=1.00 sl=91.65 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 91.12 | 91.58 | 91.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 90.68 | 91.35 | 91.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 91.06 | 91.02 | 91.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 91.06 | 91.02 | 91.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 91.93 | 91.20 | 91.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:30:00 | 92.24 | 91.20 | 91.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 91.65 | 91.29 | 91.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 91.42 | 91.37 | 91.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 91.13 | 91.30 | 91.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 91.91 | 91.36 | 91.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 91.91 | 91.36 | 91.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 92.20 | 91.53 | 91.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 94.25 | 94.36 | 93.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:15:00 | 96.24 | 94.36 | 93.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 94.29 | 94.58 | 94.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 93.51 | 94.58 | 94.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 93.45 | 94.35 | 94.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 93.45 | 94.35 | 94.06 | SL hit (close<ema400) qty=1.00 sl=94.06 alert=retest1 |

### Cycle 14 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 93.34 | 93.86 | 93.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 92.81 | 93.47 | 93.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 91.76 | 91.56 | 92.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:30:00 | 91.86 | 91.56 | 92.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 91.88 | 91.62 | 92.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 92.10 | 91.62 | 92.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 91.97 | 91.65 | 92.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 91.97 | 91.65 | 92.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 91.90 | 91.70 | 91.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 91.15 | 91.54 | 91.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 90.73 | 90.27 | 90.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 90.73 | 90.27 | 90.26 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 90.00 | 90.21 | 90.23 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 90.59 | 90.31 | 90.28 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 89.27 | 90.10 | 90.19 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 90.54 | 90.19 | 90.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 91.15 | 90.39 | 90.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 91.98 | 92.09 | 91.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 14:00:00 | 91.98 | 92.09 | 91.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 91.00 | 91.87 | 91.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 91.00 | 91.87 | 91.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 91.00 | 91.70 | 91.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 90.70 | 91.70 | 91.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 89.42 | 91.02 | 91.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 89.15 | 89.86 | 90.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 89.75 | 89.59 | 90.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:00:00 | 89.75 | 89.59 | 90.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 88.81 | 88.66 | 89.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 88.81 | 88.66 | 89.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 89.03 | 88.73 | 89.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:15:00 | 88.76 | 88.73 | 89.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 88.64 | 88.71 | 88.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 89.98 | 88.98 | 89.06 | SL hit (close>static) qty=1.00 sl=89.32 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 89.80 | 89.14 | 89.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 14:15:00 | 90.88 | 89.93 | 89.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 89.85 | 90.12 | 89.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 89.85 | 90.12 | 89.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 90.25 | 90.15 | 89.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 89.67 | 90.15 | 89.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 90.35 | 90.39 | 90.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 90.36 | 90.39 | 90.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 91.59 | 90.63 | 90.34 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 89.55 | 90.35 | 90.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 89.42 | 90.17 | 90.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 15:15:00 | 89.64 | 89.63 | 89.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:15:00 | 89.62 | 89.63 | 89.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 89.95 | 89.70 | 89.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 89.95 | 89.70 | 89.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 90.04 | 89.76 | 89.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 89.62 | 89.76 | 89.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 87.95 | 87.22 | 87.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 87.95 | 87.22 | 87.18 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 86.90 | 87.14 | 87.17 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 87.41 | 87.20 | 87.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 87.87 | 87.37 | 87.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 87.35 | 87.37 | 87.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 87.35 | 87.37 | 87.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 86.95 | 87.28 | 87.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 86.95 | 87.28 | 87.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 87.04 | 87.23 | 87.23 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 86.60 | 87.11 | 87.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 86.52 | 86.99 | 87.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 86.30 | 86.20 | 86.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 86.30 | 86.20 | 86.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 86.49 | 86.20 | 86.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 86.47 | 86.20 | 86.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 86.40 | 86.24 | 86.49 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 88.04 | 86.72 | 86.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 88.74 | 88.22 | 87.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 88.13 | 88.50 | 88.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 13:15:00 | 88.13 | 88.50 | 88.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 88.13 | 88.50 | 88.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 88.13 | 88.50 | 88.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 88.77 | 88.55 | 88.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 89.78 | 88.55 | 88.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 91.00 | 92.01 | 92.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 91.00 | 92.01 | 92.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 90.56 | 91.61 | 91.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 89.90 | 89.88 | 90.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:15:00 | 89.90 | 89.88 | 90.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 87.11 | 86.91 | 87.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 88.46 | 86.91 | 87.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 87.96 | 87.12 | 87.44 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 89.28 | 87.92 | 87.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 89.95 | 88.32 | 87.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 87.86 | 88.85 | 88.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 87.86 | 88.85 | 88.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 87.86 | 88.85 | 88.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 87.86 | 88.85 | 88.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 88.02 | 88.69 | 88.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 88.04 | 88.69 | 88.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 88.02 | 88.43 | 88.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 87.81 | 88.43 | 88.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 88.09 | 88.36 | 88.38 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 88.55 | 88.40 | 88.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 88.71 | 88.48 | 88.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 88.48 | 88.48 | 88.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 88.48 | 88.48 | 88.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 88.48 | 88.48 | 88.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 88.54 | 88.48 | 88.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 88.63 | 88.51 | 88.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 88.54 | 88.51 | 88.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 88.63 | 88.54 | 88.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 88.66 | 88.54 | 88.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 88.31 | 88.49 | 88.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 88.31 | 88.49 | 88.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 88.40 | 88.47 | 88.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 88.13 | 88.47 | 88.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 87.82 | 88.34 | 88.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 87.22 | 88.12 | 88.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 87.92 | 87.70 | 87.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 87.92 | 87.70 | 87.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 87.92 | 87.70 | 87.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 87.92 | 87.70 | 87.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 89.26 | 88.01 | 88.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 89.26 | 88.01 | 88.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 90.13 | 88.44 | 88.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 13:15:00 | 90.42 | 89.11 | 88.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 89.98 | 90.05 | 89.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 89.98 | 90.05 | 89.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 89.10 | 89.88 | 89.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 89.10 | 89.88 | 89.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 88.78 | 89.66 | 89.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 88.78 | 89.66 | 89.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 89.42 | 89.48 | 89.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 89.47 | 89.48 | 89.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 89.85 | 89.56 | 89.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 89.30 | 89.56 | 89.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 89.42 | 89.62 | 89.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 89.42 | 89.62 | 89.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 88.53 | 89.40 | 89.42 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 89.75 | 89.27 | 89.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 90.17 | 89.55 | 89.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 90.25 | 90.49 | 90.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 11:15:00 | 90.25 | 90.49 | 90.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 90.25 | 90.49 | 90.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 90.25 | 90.49 | 90.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 90.12 | 90.41 | 90.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 90.16 | 90.41 | 90.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 90.44 | 90.42 | 90.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 89.85 | 90.42 | 90.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 90.19 | 90.37 | 90.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 90.19 | 90.37 | 90.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 90.07 | 90.31 | 90.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 90.30 | 90.31 | 90.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 91.97 | 92.50 | 92.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 91.97 | 92.50 | 92.51 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 92.75 | 92.51 | 92.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 93.43 | 92.69 | 92.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 13:15:00 | 94.68 | 94.75 | 94.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:15:00 | 94.54 | 94.75 | 94.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 93.89 | 94.58 | 94.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 93.89 | 94.58 | 94.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 93.83 | 94.43 | 94.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 94.00 | 94.43 | 94.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 93.67 | 94.28 | 94.06 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 93.11 | 93.89 | 93.91 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 95.84 | 94.07 | 93.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 96.32 | 95.64 | 95.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 12:15:00 | 95.75 | 95.77 | 95.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:00:00 | 95.75 | 95.77 | 95.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 95.65 | 95.75 | 95.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 95.49 | 95.75 | 95.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 95.45 | 95.69 | 95.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 95.51 | 95.69 | 95.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 94.51 | 95.45 | 95.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 94.51 | 95.45 | 95.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 94.66 | 95.29 | 95.31 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 95.97 | 95.40 | 95.33 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 95.23 | 95.30 | 95.30 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 95.59 | 95.35 | 95.32 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 95.17 | 95.30 | 95.30 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 95.44 | 95.33 | 95.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 97.04 | 95.88 | 95.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 103.26 | 103.36 | 102.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:45:00 | 102.95 | 103.36 | 102.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 104.88 | 103.61 | 102.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 105.40 | 103.97 | 102.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 105.08 | 104.37 | 103.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 105.00 | 104.43 | 103.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 105.19 | 103.98 | 103.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 104.09 | 104.44 | 104.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 104.09 | 104.44 | 104.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 104.57 | 104.47 | 104.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 106.04 | 104.47 | 104.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 106.70 | 107.51 | 107.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 106.70 | 107.51 | 107.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 106.61 | 107.12 | 107.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 103.03 | 102.91 | 104.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:45:00 | 103.00 | 102.91 | 104.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 100.10 | 98.68 | 99.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 100.02 | 98.68 | 99.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 101.04 | 99.15 | 99.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 101.04 | 99.15 | 99.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 101.74 | 99.67 | 99.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 104.00 | 100.92 | 100.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 104.45 | 104.56 | 103.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 104.45 | 104.56 | 103.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 103.36 | 104.22 | 103.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 103.31 | 104.22 | 103.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 103.01 | 103.98 | 103.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 103.01 | 103.98 | 103.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 102.89 | 103.76 | 103.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 102.89 | 103.76 | 103.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 103.57 | 103.72 | 103.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 103.07 | 103.72 | 103.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 103.21 | 103.62 | 103.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 103.57 | 103.62 | 103.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 102.89 | 103.47 | 103.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 102.89 | 103.47 | 103.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 102.49 | 103.28 | 103.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 102.49 | 103.28 | 103.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 101.92 | 103.01 | 103.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 101.74 | 102.75 | 102.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 101.03 | 100.71 | 101.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 101.03 | 100.71 | 101.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 101.03 | 100.71 | 101.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 101.52 | 100.71 | 101.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 101.37 | 100.84 | 101.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 100.78 | 100.84 | 101.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:00:00 | 100.95 | 100.88 | 101.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 101.58 | 101.02 | 101.27 | SL hit (close>static) qty=1.00 sl=101.53 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 102.72 | 101.59 | 101.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 13:15:00 | 104.15 | 102.53 | 102.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 103.83 | 104.01 | 103.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:00:00 | 103.83 | 104.01 | 103.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 103.15 | 103.77 | 103.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 103.15 | 103.77 | 103.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 103.17 | 103.65 | 103.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 103.99 | 103.65 | 103.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 103.58 | 103.63 | 103.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 104.20 | 103.63 | 103.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 102.45 | 103.26 | 103.21 | SL hit (close<static) qty=1.00 sl=102.70 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 102.40 | 103.09 | 103.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 101.46 | 102.76 | 102.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 102.15 | 102.07 | 102.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:30:00 | 102.17 | 102.07 | 102.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 102.77 | 102.22 | 102.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 102.77 | 102.22 | 102.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 103.04 | 102.39 | 102.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 103.50 | 102.39 | 102.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 102.94 | 102.50 | 102.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 103.37 | 102.67 | 102.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 105.41 | 105.42 | 104.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:00:00 | 105.41 | 105.42 | 104.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 105.03 | 105.65 | 105.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 105.10 | 105.65 | 105.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 105.88 | 105.69 | 105.31 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 104.78 | 105.08 | 105.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 103.93 | 104.85 | 105.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 104.83 | 104.60 | 104.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 104.83 | 104.60 | 104.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 104.83 | 104.60 | 104.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 104.93 | 104.60 | 104.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 105.25 | 104.73 | 104.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 104.96 | 104.73 | 104.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 104.93 | 104.77 | 104.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 103.80 | 104.51 | 104.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 98.61 | 100.73 | 102.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 100.07 | 99.84 | 100.92 | SL hit (close>ema200) qty=0.50 sl=99.84 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 101.40 | 100.17 | 100.04 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 100.00 | 100.25 | 100.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 98.91 | 99.98 | 100.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 95.69 | 94.99 | 96.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 95.69 | 94.99 | 96.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 95.69 | 94.99 | 96.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 94.32 | 94.75 | 95.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 10:15:00 | 89.60 | 92.02 | 93.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 92.49 | 91.51 | 92.81 | SL hit (close>ema200) qty=0.50 sl=91.51 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 94.20 | 93.31 | 93.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 94.25 | 93.59 | 93.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 93.66 | 93.74 | 93.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 93.66 | 93.74 | 93.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 93.86 | 93.84 | 93.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:15:00 | 93.99 | 93.83 | 93.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 94.00 | 93.86 | 93.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 94.00 | 93.86 | 93.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 92.72 | 93.63 | 93.61 | SL hit (close<static) qty=1.00 sl=93.60 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 91.89 | 93.28 | 93.45 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 93.96 | 93.07 | 93.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 95.59 | 93.72 | 93.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 97.07 | 97.25 | 96.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 97.07 | 97.25 | 96.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 96.62 | 97.26 | 96.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 95.98 | 97.26 | 96.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 96.99 | 97.21 | 96.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 96.46 | 97.21 | 96.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 96.34 | 97.02 | 96.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 96.34 | 97.02 | 96.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 96.62 | 96.94 | 96.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 96.89 | 97.07 | 96.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 96.80 | 97.42 | 97.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 96.84 | 97.31 | 97.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 96.84 | 97.31 | 97.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 95.92 | 96.51 | 96.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 99.25 | 95.54 | 95.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 99.25 | 95.54 | 95.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 99.25 | 95.54 | 95.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 99.25 | 95.54 | 95.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 99.70 | 96.38 | 96.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 101.25 | 98.35 | 97.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 99.76 | 99.96 | 98.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 99.76 | 99.96 | 98.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 100.83 | 100.11 | 99.16 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 98.80 | 99.66 | 99.68 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 99.74 | 99.68 | 99.68 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 99.57 | 99.67 | 99.68 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 100.89 | 99.92 | 99.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 102.10 | 100.82 | 100.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 101.56 | 101.97 | 101.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 101.56 | 101.97 | 101.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 102.00 | 101.97 | 101.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 102.10 | 102.01 | 101.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:00:00 | 102.13 | 102.01 | 101.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 102.28 | 102.04 | 101.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 100.97 | 101.88 | 101.64 | SL hit (close<static) qty=1.00 sl=101.51 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 100.41 | 101.47 | 101.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 99.75 | 101.12 | 101.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 95.87 | 95.79 | 97.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 14:00:00 | 95.87 | 95.79 | 97.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 96.77 | 96.13 | 96.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 96.87 | 96.13 | 96.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 96.66 | 96.23 | 96.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 96.83 | 96.23 | 96.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 97.34 | 96.45 | 96.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 97.34 | 96.45 | 96.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 98.25 | 96.81 | 97.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 98.79 | 96.81 | 97.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 97.46 | 97.19 | 97.16 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 96.73 | 97.10 | 97.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 94.95 | 96.60 | 96.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 94.16 | 92.97 | 94.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 94.16 | 92.97 | 94.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 94.16 | 92.97 | 94.26 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 96.35 | 94.83 | 94.74 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 93.58 | 94.70 | 94.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 93.08 | 94.24 | 94.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 94.47 | 94.28 | 94.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 94.47 | 94.28 | 94.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 94.47 | 94.28 | 94.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 94.47 | 94.28 | 94.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 95.00 | 94.43 | 94.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 95.00 | 94.43 | 94.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 94.14 | 94.37 | 94.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:30:00 | 94.23 | 94.37 | 94.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 94.50 | 94.39 | 94.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 94.83 | 94.39 | 94.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 93.28 | 94.17 | 94.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 91.85 | 94.00 | 94.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 92.12 | 91.18 | 91.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 92.12 | 91.18 | 91.07 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 89.19 | 90.77 | 90.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 88.90 | 90.39 | 90.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 92.18 | 89.90 | 90.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 92.18 | 89.90 | 90.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 92.18 | 89.90 | 90.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 92.19 | 89.90 | 90.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 92.64 | 90.45 | 90.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 92.75 | 90.45 | 90.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 91.95 | 90.75 | 90.60 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 86.59 | 90.06 | 90.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 85.99 | 89.25 | 89.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 87.59 | 86.55 | 88.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 87.59 | 86.55 | 88.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 87.59 | 86.55 | 88.00 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 91.23 | 88.86 | 88.61 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 88.23 | 89.12 | 89.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 86.31 | 88.54 | 88.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 88.97 | 86.66 | 87.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 88.97 | 86.66 | 87.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 88.97 | 86.66 | 87.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 88.97 | 86.66 | 87.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 88.12 | 86.96 | 87.52 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 90.76 | 88.24 | 88.04 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 86.92 | 88.05 | 88.10 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 88.99 | 88.19 | 88.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 89.23 | 88.46 | 88.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 94.46 | 94.68 | 93.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 94.40 | 94.68 | 93.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 94.53 | 95.05 | 94.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 94.89 | 95.05 | 94.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 94.93 | 95.05 | 94.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 97.41 | 97.52 | 97.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 97.41 | 97.52 | 97.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 96.36 | 97.25 | 97.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 95.52 | 95.22 | 95.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 95.52 | 95.22 | 95.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 95.52 | 95.22 | 95.87 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 97.05 | 96.09 | 96.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 97.79 | 97.00 | 96.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 09:15:00 | 96.87 | 97.14 | 96.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 96.87 | 97.14 | 96.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 96.87 | 97.14 | 96.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 96.58 | 97.14 | 96.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 97.49 | 97.21 | 96.83 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 94.22 | 96.29 | 96.52 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 99.29 | 96.80 | 96.52 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 96.31 | 96.98 | 97.06 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 98.99 | 97.46 | 97.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 99.75 | 97.91 | 97.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-29 10:00:00 | 85.85 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-05-29 11:00:00 | 85.95 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-05-29 12:00:00 | 86.05 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-05-29 12:30:00 | 85.96 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-05-30 09:15:00 | 85.76 | 2025-06-06 09:15:00 | 85.50 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-07-10 14:15:00 | 92.39 | 2025-07-11 09:15:00 | 91.45 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-14 15:15:00 | 91.42 | 2025-07-15 14:15:00 | 91.91 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-15 09:30:00 | 91.13 | 2025-07-15 14:15:00 | 91.91 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest1 | 2025-07-18 09:15:00 | 96.24 | 2025-07-21 09:15:00 | 93.45 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-25 09:30:00 | 91.15 | 2025-07-31 12:15:00 | 90.73 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-08-13 14:15:00 | 88.76 | 2025-08-14 09:15:00 | 89.98 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-13 15:00:00 | 88.64 | 2025-08-14 09:15:00 | 89.98 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-08-25 11:15:00 | 89.62 | 2025-09-02 11:15:00 | 87.95 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-09-15 15:15:00 | 89.78 | 2025-09-22 15:15:00 | 91.00 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-10-20 09:15:00 | 90.30 | 2025-10-29 11:15:00 | 91.97 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2025-11-21 11:00:00 | 105.40 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-11-21 13:00:00 | 105.08 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-11-21 13:45:00 | 105.00 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2025-11-25 09:30:00 | 105.19 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-11-26 09:15:00 | 106.04 | 2025-12-02 14:15:00 | 106.70 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-12-19 11:15:00 | 100.78 | 2025-12-19 14:15:00 | 101.58 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-19 14:00:00 | 100.95 | 2025-12-19 14:15:00 | 101.58 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-26 10:15:00 | 104.20 | 2025-12-26 14:15:00 | 102.45 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-08 10:00:00 | 103.80 | 2026-01-12 09:15:00 | 98.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 103.80 | 2026-01-12 15:15:00 | 100.07 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-01-23 11:30:00 | 94.32 | 2026-01-27 10:15:00 | 89.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:30:00 | 94.32 | 2026-01-27 14:15:00 | 92.49 | STOP_HIT | 0.50 | 1.94% |
| BUY | retest2 | 2026-01-30 15:15:00 | 93.99 | 2026-02-01 11:15:00 | 92.72 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-01 10:30:00 | 94.00 | 2026-02-01 11:15:00 | 92.72 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-01 11:15:00 | 94.00 | 2026-02-01 11:15:00 | 92.72 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-06 14:30:00 | 96.89 | 2026-02-11 10:15:00 | 96.84 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-02-11 10:15:00 | 96.80 | 2026-02-11 10:15:00 | 96.84 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-02-26 13:30:00 | 102.10 | 2026-02-27 09:15:00 | 100.97 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-26 14:00:00 | 102.13 | 2026-02-27 09:15:00 | 100.97 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-26 14:30:00 | 102.28 | 2026-02-27 09:15:00 | 100.97 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-02-27 13:45:00 | 102.13 | 2026-02-27 14:15:00 | 100.41 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-03-13 09:15:00 | 91.85 | 2026-03-18 13:15:00 | 92.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-04-13 10:15:00 | 94.89 | 2026-04-22 14:15:00 | 97.41 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-04-13 10:45:00 | 94.93 | 2026-04-22 14:15:00 | 97.41 | STOP_HIT | 1.00 | 2.61% |
