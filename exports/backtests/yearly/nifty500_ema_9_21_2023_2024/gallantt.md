# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 866.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 229 |
| ALERT1 | 139 |
| ALERT2 | 134 |
| ALERT2_SKIP | 74 |
| ALERT3 | 377 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 201 |
| PARTIAL | 43 |
| TARGET_HIT | 24 |
| STOP_HIT | 180 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 247 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 110 / 137
- **Target hits / Stop hits / Partials:** 24 / 180 / 43
- **Avg / median % per leg:** 0.94% / -0.54%
- **Sum % (uncompounded):** 232.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 32 | 48.5% | 19 | 46 | 1 | 2.46% | 162.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.05% | 0.2% |
| BUY @ 3rd Alert (retest2) | 62 | 30 | 48.4% | 19 | 43 | 0 | 2.62% | 162.2% |
| SELL (all) | 181 | 78 | 43.1% | 5 | 134 | 42 | 0.39% | 70.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 181 | 78 | 43.1% | 5 | 134 | 42 | 0.39% | 70.6% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.05% | 0.2% |
| retest2 (combined) | 243 | 108 | 44.4% | 24 | 177 | 42 | 0.96% | 232.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 12:15:00 | 53.95 | 53.40 | 53.33 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 15:15:00 | 53.10 | 53.30 | 53.32 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 53.40 | 53.34 | 53.34 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 11:15:00 | 53.10 | 53.29 | 53.32 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 53.25 | 52.88 | 52.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 13:15:00 | 53.75 | 53.41 | 53.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 52.95 | 53.40 | 53.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 52.95 | 53.40 | 53.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 52.95 | 53.40 | 53.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 53.05 | 53.40 | 53.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 53.25 | 53.37 | 53.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 12:30:00 | 53.45 | 53.28 | 53.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 14:00:00 | 53.65 | 53.36 | 53.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 15:15:00 | 53.50 | 53.35 | 53.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 09:15:00 | 52.95 | 53.30 | 53.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 09:15:00 | 52.95 | 53.30 | 53.31 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 53.95 | 53.31 | 53.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 10:15:00 | 54.00 | 53.45 | 53.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 11:15:00 | 53.40 | 53.44 | 53.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 12:00:00 | 53.40 | 53.44 | 53.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 53.10 | 53.38 | 53.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 13:30:00 | 53.00 | 53.38 | 53.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 53.50 | 53.40 | 53.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 14:30:00 | 53.15 | 53.40 | 53.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 53.35 | 53.39 | 53.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:45:00 | 54.05 | 53.54 | 53.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 14:30:00 | 54.25 | 53.79 | 53.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 15:15:00 | 57.05 | 57.49 | 57.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 15:15:00 | 57.05 | 57.49 | 57.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 15:15:00 | 56.25 | 57.07 | 57.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 57.80 | 57.22 | 57.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 57.80 | 57.22 | 57.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 57.80 | 57.22 | 57.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:45:00 | 57.60 | 57.22 | 57.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 57.75 | 57.32 | 57.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 11:15:00 | 57.10 | 57.32 | 57.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 11:45:00 | 57.20 | 57.30 | 57.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 09:30:00 | 57.45 | 57.27 | 57.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 11:15:00 | 57.80 | 57.42 | 57.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 11:15:00 | 57.80 | 57.42 | 57.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 12:15:00 | 58.30 | 57.60 | 57.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 15:15:00 | 58.50 | 58.62 | 58.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:45:00 | 59.35 | 58.64 | 58.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 58.30 | 58.59 | 58.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-19 10:15:00 | 58.30 | 58.59 | 58.45 | SL hit (close<ema400) qty=1.00 sl=58.45 alert=retest1 |

### Cycle 10 — SELL (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 14:15:00 | 58.15 | 58.45 | 58.45 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 15:15:00 | 58.75 | 58.51 | 58.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 09:15:00 | 59.10 | 58.63 | 58.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 10:15:00 | 58.80 | 59.12 | 58.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 58.80 | 59.12 | 58.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 58.80 | 59.12 | 58.92 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 13:15:00 | 58.35 | 58.74 | 58.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 57.60 | 58.41 | 58.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 57.35 | 57.28 | 57.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 57.35 | 57.28 | 57.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 57.35 | 57.28 | 57.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 57.80 | 57.28 | 57.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 57.35 | 57.29 | 57.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:45:00 | 57.55 | 57.29 | 57.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 58.30 | 57.50 | 57.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:00:00 | 58.30 | 57.50 | 57.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 58.90 | 57.78 | 57.77 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 14:15:00 | 57.55 | 57.80 | 57.81 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 15:15:00 | 57.95 | 57.83 | 57.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 09:15:00 | 59.40 | 58.14 | 57.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 13:15:00 | 59.90 | 60.07 | 59.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 13:30:00 | 60.00 | 60.07 | 59.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 62.30 | 61.74 | 60.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:30:00 | 60.95 | 61.74 | 60.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 71.00 | 72.75 | 71.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:00:00 | 71.00 | 72.75 | 71.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 71.35 | 72.47 | 71.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 14:45:00 | 71.60 | 72.47 | 71.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 71.35 | 72.24 | 71.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 71.05 | 72.24 | 71.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 73.45 | 72.48 | 71.41 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 69.60 | 71.30 | 71.41 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 10:15:00 | 73.20 | 71.54 | 71.47 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 10:15:00 | 71.20 | 72.12 | 72.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 11:15:00 | 70.00 | 71.69 | 71.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 70.85 | 70.77 | 71.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-19 09:30:00 | 70.40 | 70.77 | 71.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 70.30 | 70.67 | 71.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 10:30:00 | 70.50 | 70.67 | 71.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 70.30 | 70.13 | 70.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:00:00 | 70.30 | 70.13 | 70.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 70.85 | 70.28 | 70.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:45:00 | 71.45 | 70.28 | 70.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 70.75 | 70.37 | 70.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:30:00 | 71.25 | 70.37 | 70.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 70.00 | 70.30 | 70.60 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 10:15:00 | 72.85 | 71.12 | 70.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 10:15:00 | 73.40 | 72.36 | 71.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 14:15:00 | 72.10 | 72.54 | 72.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-24 15:00:00 | 72.10 | 72.54 | 72.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 72.20 | 72.47 | 72.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:15:00 | 73.05 | 72.47 | 72.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-26 12:15:00 | 80.36 | 76.81 | 74.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 86.20 | 88.15 | 88.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 13:15:00 | 85.50 | 86.85 | 87.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 87.70 | 87.02 | 87.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 14:15:00 | 87.70 | 87.02 | 87.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 87.70 | 87.02 | 87.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 15:00:00 | 87.70 | 87.02 | 87.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 88.00 | 87.22 | 87.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:15:00 | 87.35 | 87.22 | 87.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 87.85 | 87.29 | 87.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:30:00 | 87.40 | 87.29 | 87.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 87.35 | 87.30 | 87.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 11:45:00 | 85.50 | 86.90 | 87.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-09 10:00:00 | 85.50 | 84.79 | 85.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 12:15:00 | 85.85 | 84.83 | 84.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 12:15:00 | 85.85 | 84.83 | 84.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 15:15:00 | 90.00 | 86.49 | 85.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 09:15:00 | 85.75 | 86.34 | 85.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 09:15:00 | 85.75 | 86.34 | 85.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 85.75 | 86.34 | 85.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 85.55 | 86.34 | 85.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 84.90 | 86.05 | 85.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:45:00 | 85.30 | 86.05 | 85.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 85.40 | 85.92 | 85.56 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 15:15:00 | 84.70 | 85.25 | 85.32 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 89.10 | 86.02 | 85.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 09:15:00 | 91.00 | 88.48 | 87.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 13:15:00 | 91.30 | 91.31 | 90.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 13:45:00 | 91.45 | 91.31 | 90.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 89.45 | 90.93 | 90.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 15:00:00 | 89.45 | 90.93 | 90.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 90.40 | 90.83 | 90.14 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 12:15:00 | 89.00 | 89.81 | 89.82 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 13:15:00 | 90.15 | 89.88 | 89.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 91.20 | 90.12 | 89.97 | Break + close above crossover candle high |

### Cycle 26 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 88.45 | 90.09 | 90.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 11:15:00 | 87.00 | 87.99 | 88.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 15:15:00 | 88.00 | 87.06 | 87.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 15:15:00 | 88.00 | 87.06 | 87.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 88.00 | 87.06 | 87.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:00:00 | 88.30 | 87.31 | 87.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 87.55 | 87.36 | 87.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 10:30:00 | 87.60 | 87.36 | 87.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 87.70 | 87.43 | 87.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 11:30:00 | 87.65 | 87.43 | 87.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 87.50 | 87.44 | 87.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 13:30:00 | 86.90 | 87.27 | 87.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 09:15:00 | 89.45 | 87.63 | 87.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 89.45 | 87.63 | 87.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 11:15:00 | 92.30 | 88.74 | 88.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 15:15:00 | 92.80 | 92.98 | 91.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 11:15:00 | 93.60 | 92.96 | 91.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 13:15:00 | 98.28 | 94.18 | 92.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-09-05 12:15:00 | 96.85 | 96.87 | 94.91 | SL hit (close<ema200) qty=0.50 sl=96.87 alert=retest1 |

### Cycle 28 — SELL (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 12:15:00 | 94.75 | 95.62 | 95.63 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 96.90 | 95.87 | 95.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 13:15:00 | 101.95 | 97.22 | 96.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 96.10 | 98.11 | 97.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 96.10 | 98.11 | 97.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 96.10 | 98.11 | 97.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 94.70 | 98.11 | 97.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 97.85 | 98.05 | 97.27 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 93.00 | 96.62 | 96.73 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 96.75 | 95.60 | 95.57 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 94.70 | 95.82 | 95.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 94.05 | 95.25 | 95.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 09:15:00 | 94.90 | 94.87 | 95.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 94.90 | 94.87 | 95.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 94.90 | 94.87 | 95.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:30:00 | 95.35 | 94.87 | 95.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 95.05 | 94.85 | 95.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:30:00 | 95.20 | 94.85 | 95.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 94.75 | 94.83 | 95.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:30:00 | 95.10 | 94.83 | 95.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 95.20 | 94.90 | 95.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 14:45:00 | 95.30 | 94.90 | 95.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 95.10 | 94.94 | 95.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 09:15:00 | 95.40 | 94.94 | 95.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 94.05 | 94.76 | 95.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 13:00:00 | 93.90 | 94.50 | 94.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 13:30:00 | 93.65 | 94.29 | 94.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 10:00:00 | 93.65 | 94.21 | 94.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 11:00:00 | 93.80 | 94.13 | 94.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 93.30 | 92.59 | 93.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:15:00 | 93.45 | 92.59 | 93.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 92.80 | 92.63 | 93.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 93.45 | 92.63 | 93.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 13:15:00 | 92.85 | 92.61 | 92.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 13:45:00 | 92.80 | 92.61 | 92.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 92.70 | 92.63 | 92.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 92.70 | 92.63 | 92.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 93.00 | 92.70 | 92.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 93.00 | 92.70 | 92.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 92.60 | 92.68 | 92.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 12:00:00 | 92.15 | 92.55 | 92.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 14:00:00 | 91.95 | 92.45 | 92.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:15:00 | 89.20 | 90.34 | 91.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:15:00 | 88.97 | 90.34 | 91.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:15:00 | 88.97 | 90.34 | 91.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:15:00 | 89.11 | 90.34 | 91.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-29 11:15:00 | 90.45 | 90.37 | 91.15 | SL hit (close>ema200) qty=0.50 sl=90.37 alert=retest2 |

### Cycle 33 — BUY (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 13:15:00 | 88.95 | 88.68 | 88.65 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 87.00 | 88.40 | 88.53 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 88.95 | 87.86 | 87.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 10:15:00 | 89.15 | 88.77 | 88.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 88.90 | 89.11 | 88.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 10:15:00 | 88.90 | 89.11 | 88.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 88.90 | 89.11 | 88.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:45:00 | 88.65 | 89.11 | 88.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 88.00 | 88.89 | 88.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:00:00 | 88.00 | 88.89 | 88.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 88.80 | 88.87 | 88.75 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 87.90 | 88.62 | 88.65 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 93.45 | 89.47 | 89.02 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 90.55 | 91.22 | 91.26 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 09:15:00 | 91.50 | 91.28 | 91.28 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 90.65 | 91.15 | 91.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 11:15:00 | 90.20 | 90.96 | 91.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 14:15:00 | 90.70 | 90.55 | 90.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 14:15:00 | 90.70 | 90.55 | 90.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 90.70 | 90.55 | 90.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:45:00 | 90.70 | 90.55 | 90.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 89.20 | 90.24 | 90.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:45:00 | 88.15 | 89.68 | 90.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:00:00 | 87.90 | 89.33 | 90.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:30:00 | 88.20 | 88.80 | 89.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 11:15:00 | 88.50 | 88.84 | 89.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 13:15:00 | 83.79 | 86.11 | 87.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 13:15:00 | 84.08 | 86.11 | 87.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 14:15:00 | 83.74 | 85.59 | 87.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 14:15:00 | 83.50 | 85.59 | 87.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-26 09:15:00 | 79.34 | 84.20 | 86.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 15:15:00 | 85.40 | 84.69 | 84.68 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 11:15:00 | 84.15 | 84.69 | 84.75 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 85.00 | 84.74 | 84.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 12:15:00 | 85.65 | 84.93 | 84.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 14:15:00 | 85.00 | 85.06 | 84.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 14:15:00 | 85.00 | 85.06 | 84.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 85.00 | 85.06 | 84.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:45:00 | 85.75 | 85.06 | 84.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 86.00 | 85.25 | 85.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 88.45 | 85.25 | 85.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-03 09:15:00 | 97.30 | 87.32 | 85.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 10:15:00 | 98.15 | 98.38 | 98.39 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 98.55 | 98.42 | 98.41 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 15:15:00 | 98.25 | 98.40 | 98.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 97.75 | 98.21 | 98.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 97.65 | 97.58 | 97.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 10:15:00 | 97.55 | 97.57 | 97.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 97.55 | 97.57 | 97.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:45:00 | 97.55 | 97.57 | 97.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 13:15:00 | 98.55 | 97.71 | 97.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 13:30:00 | 99.50 | 97.71 | 97.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 14:15:00 | 97.70 | 97.71 | 97.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 15:15:00 | 97.25 | 97.71 | 97.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 13:00:00 | 97.30 | 97.64 | 97.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 13:30:00 | 97.20 | 97.52 | 97.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 11:00:00 | 97.00 | 97.15 | 97.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 97.05 | 96.83 | 97.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:15:00 | 97.80 | 96.83 | 97.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 96.90 | 96.84 | 97.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 11:30:00 | 96.60 | 96.87 | 97.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:15:00 | 96.20 | 96.87 | 97.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-01 10:15:00 | 97.40 | 96.68 | 96.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 97.40 | 96.68 | 96.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 12:15:00 | 98.10 | 97.12 | 96.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 12:15:00 | 97.50 | 97.77 | 97.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-04 13:00:00 | 97.50 | 97.77 | 97.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 96.65 | 97.52 | 97.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 15:00:00 | 96.65 | 97.52 | 97.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 97.05 | 97.42 | 97.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:30:00 | 97.65 | 97.60 | 97.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 10:00:00 | 98.30 | 97.60 | 97.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 12:30:00 | 97.65 | 97.58 | 97.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:15:00 | 97.70 | 97.44 | 97.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 09:15:00 | 97.00 | 97.36 | 97.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 09:15:00 | 97.00 | 97.36 | 97.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 10:15:00 | 96.60 | 97.20 | 97.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 14:15:00 | 96.95 | 96.94 | 97.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-06 15:00:00 | 96.95 | 96.94 | 97.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 97.00 | 96.83 | 97.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:30:00 | 97.15 | 96.83 | 97.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 97.00 | 96.86 | 97.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 15:15:00 | 96.10 | 96.84 | 96.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:00:00 | 96.20 | 96.71 | 96.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:45:00 | 96.30 | 96.65 | 96.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 12:45:00 | 96.40 | 96.43 | 96.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 96.70 | 96.48 | 96.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 15:00:00 | 96.70 | 96.48 | 96.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 96.15 | 96.41 | 96.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:15:00 | 97.20 | 96.41 | 96.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 97.00 | 96.53 | 96.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:30:00 | 97.50 | 96.53 | 96.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-12-12 10:15:00 | 97.00 | 96.62 | 96.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 97.00 | 96.62 | 96.61 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 12:15:00 | 96.55 | 96.59 | 96.59 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 15:15:00 | 97.60 | 96.76 | 96.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 09:15:00 | 110.00 | 99.41 | 97.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 12:15:00 | 110.30 | 110.34 | 106.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 13:00:00 | 110.30 | 110.34 | 106.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 109.30 | 110.95 | 109.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 111.65 | 110.95 | 109.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 111.15 | 110.99 | 109.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 11:30:00 | 113.55 | 111.42 | 109.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-20 09:15:00 | 124.91 | 116.82 | 114.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 172.95 | 177.62 | 177.84 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 15:15:00 | 182.00 | 178.39 | 178.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 182.65 | 180.30 | 179.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 186.40 | 188.46 | 185.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 10:15:00 | 186.40 | 188.46 | 185.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 186.40 | 188.46 | 185.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:45:00 | 185.50 | 188.46 | 185.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 185.80 | 187.93 | 185.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 13:30:00 | 187.05 | 187.18 | 185.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 15:00:00 | 186.90 | 187.13 | 185.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 191.40 | 186.79 | 185.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 12:45:00 | 187.70 | 191.94 | 191.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 13:15:00 | 187.70 | 191.09 | 191.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 13:15:00 | 187.70 | 191.09 | 191.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 12:15:00 | 185.00 | 189.23 | 190.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 180.35 | 180.27 | 183.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 180.35 | 180.27 | 183.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 180.35 | 180.27 | 183.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:30:00 | 180.80 | 180.27 | 183.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 180.45 | 180.31 | 183.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:30:00 | 183.85 | 180.31 | 183.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 179.00 | 180.39 | 182.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 13:15:00 | 176.00 | 180.39 | 182.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 183.45 | 180.86 | 182.19 | SL hit (close>static) qty=1.00 sl=183.30 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 180.00 | 178.97 | 178.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 11:15:00 | 188.05 | 180.99 | 179.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 09:15:00 | 180.15 | 183.47 | 181.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 180.15 | 183.47 | 181.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 180.15 | 183.47 | 181.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 180.15 | 183.47 | 181.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 180.55 | 182.89 | 181.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 179.85 | 182.89 | 181.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 178.50 | 182.01 | 181.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 178.50 | 182.01 | 181.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 177.80 | 180.40 | 180.74 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 182.50 | 180.52 | 180.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 195.30 | 184.36 | 182.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 189.40 | 190.90 | 187.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 11:00:00 | 189.40 | 190.90 | 187.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 192.35 | 192.22 | 190.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 206.45 | 192.17 | 190.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-02 09:15:00 | 227.09 | 208.79 | 201.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 214.00 | 215.68 | 215.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 211.60 | 214.87 | 215.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 13:15:00 | 216.00 | 215.03 | 215.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 13:15:00 | 216.00 | 215.03 | 215.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 216.00 | 215.03 | 215.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 13:30:00 | 218.00 | 215.03 | 215.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 212.00 | 214.42 | 215.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:30:00 | 215.00 | 214.42 | 215.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 207.00 | 212.67 | 214.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 10:15:00 | 205.00 | 212.67 | 214.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 11:15:00 | 204.00 | 211.54 | 213.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 09:30:00 | 204.00 | 207.18 | 210.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 09:15:00 | 194.75 | 198.02 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 09:15:00 | 193.80 | 198.02 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 09:15:00 | 193.80 | 198.02 | 203.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-02-14 09:15:00 | 184.50 | 195.64 | 199.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 59 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 203.15 | 201.17 | 200.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 213.05 | 203.54 | 202.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 10:15:00 | 226.00 | 228.48 | 223.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 11:00:00 | 226.00 | 228.48 | 223.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 225.90 | 226.78 | 224.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:30:00 | 224.95 | 226.78 | 224.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 224.00 | 226.23 | 224.79 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 221.25 | 223.98 | 224.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 11:15:00 | 221.05 | 222.87 | 223.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 222.95 | 222.60 | 223.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:45:00 | 223.00 | 222.60 | 223.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 222.00 | 222.48 | 223.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:30:00 | 221.75 | 222.48 | 223.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 223.50 | 222.69 | 223.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 227.00 | 223.55 | 223.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 224.95 | 223.83 | 223.69 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 11:15:00 | 220.00 | 223.06 | 223.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 11:15:00 | 218.30 | 221.89 | 222.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-26 12:15:00 | 223.00 | 222.11 | 222.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 12:15:00 | 223.00 | 222.11 | 222.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 223.00 | 222.11 | 222.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 13:00:00 | 223.00 | 222.11 | 222.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 219.25 | 221.54 | 222.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 14:30:00 | 215.00 | 219.96 | 221.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 13:15:00 | 218.00 | 213.85 | 213.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 218.00 | 213.85 | 213.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 219.00 | 215.56 | 214.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 213.00 | 215.65 | 214.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 213.00 | 215.65 | 214.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 213.00 | 215.65 | 214.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:45:00 | 212.00 | 215.65 | 214.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 217.05 | 215.93 | 215.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:30:00 | 215.10 | 215.93 | 215.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 218.95 | 216.71 | 215.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:00:00 | 219.00 | 216.71 | 215.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:30:00 | 220.70 | 217.52 | 216.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 212.05 | 217.14 | 217.08 | SL hit (close<static) qty=1.00 sl=215.10 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 12:15:00 | 211.25 | 215.96 | 216.55 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 15:15:00 | 217.00 | 215.70 | 215.68 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 212.85 | 215.13 | 215.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 210.00 | 214.10 | 214.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 193.00 | 190.63 | 196.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 09:30:00 | 188.00 | 190.63 | 196.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 193.15 | 191.95 | 195.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 193.00 | 191.95 | 195.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 196.40 | 192.84 | 195.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:45:00 | 196.40 | 192.84 | 195.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 193.00 | 192.87 | 195.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:30:00 | 192.40 | 192.87 | 195.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 193.00 | 192.90 | 195.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 197.00 | 192.90 | 195.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 193.75 | 193.07 | 195.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 13:00:00 | 187.20 | 190.66 | 192.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 09:15:00 | 198.80 | 193.47 | 193.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 09:15:00 | 198.80 | 193.47 | 193.20 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 10:15:00 | 190.50 | 193.73 | 193.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 195.00 | 193.62 | 193.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 10:15:00 | 197.50 | 194.56 | 194.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 195.00 | 196.11 | 195.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 195.00 | 196.11 | 195.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 195.00 | 196.11 | 195.29 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 15:15:00 | 192.00 | 194.44 | 194.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 09:15:00 | 191.50 | 193.85 | 194.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 193.40 | 192.55 | 193.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 193.40 | 192.55 | 193.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 193.40 | 192.55 | 193.35 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 196.00 | 193.78 | 193.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 201.85 | 197.63 | 195.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 10:15:00 | 202.10 | 203.66 | 201.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 10:30:00 | 202.10 | 203.66 | 201.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 200.00 | 202.83 | 201.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:00:00 | 200.00 | 202.83 | 201.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 203.00 | 202.87 | 201.92 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 12:15:00 | 200.05 | 201.96 | 201.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 198.00 | 201.17 | 201.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 14:15:00 | 202.35 | 201.41 | 201.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 14:15:00 | 202.35 | 201.41 | 201.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 202.35 | 201.41 | 201.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 15:00:00 | 202.35 | 201.41 | 201.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 200.00 | 201.12 | 201.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 09:15:00 | 200.00 | 201.12 | 201.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 199.70 | 200.84 | 201.36 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 203.80 | 200.90 | 200.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 205.40 | 201.80 | 201.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 11:15:00 | 204.50 | 204.59 | 203.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 11:45:00 | 204.50 | 204.59 | 203.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 206.15 | 205.61 | 204.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:45:00 | 208.00 | 205.61 | 204.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 205.00 | 205.87 | 204.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 10:45:00 | 210.00 | 206.36 | 204.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-18 09:15:00 | 231.00 | 227.58 | 220.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 297.00 | 309.26 | 309.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 09:15:00 | 290.95 | 296.80 | 301.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 12:15:00 | 280.10 | 276.13 | 282.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 12:15:00 | 280.10 | 276.13 | 282.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 280.10 | 276.13 | 282.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:30:00 | 275.00 | 276.13 | 282.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 282.80 | 277.46 | 282.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 283.35 | 277.46 | 282.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 282.80 | 278.53 | 282.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:30:00 | 282.80 | 278.53 | 282.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 283.80 | 279.59 | 282.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 277.10 | 279.59 | 282.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 275.20 | 278.71 | 281.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 270.00 | 277.10 | 280.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 11:00:00 | 270.65 | 277.10 | 280.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 12:30:00 | 272.35 | 276.88 | 280.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 285.90 | 278.98 | 280.06 | SL hit (close>static) qty=1.00 sl=283.80 alert=retest2 |

### Cycle 75 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 286.35 | 280.80 | 280.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 287.95 | 283.35 | 282.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 286.00 | 287.45 | 285.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 14:15:00 | 286.00 | 287.45 | 285.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 286.00 | 287.45 | 285.31 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 281.05 | 283.87 | 284.24 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 287.60 | 284.36 | 284.28 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 13:15:00 | 282.00 | 283.92 | 284.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 15:15:00 | 280.30 | 283.38 | 283.83 | Break + close below crossover candle low |

### Cycle 79 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 288.00 | 284.30 | 284.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 289.95 | 285.43 | 284.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 285.00 | 286.08 | 285.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 285.00 | 286.08 | 285.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 285.00 | 286.08 | 285.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 285.00 | 286.08 | 285.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 284.70 | 285.80 | 285.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 286.90 | 285.80 | 285.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 287.60 | 286.16 | 285.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:30:00 | 282.00 | 286.16 | 285.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 285.00 | 285.93 | 285.34 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 282.00 | 284.68 | 284.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 275.00 | 282.32 | 283.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 14:15:00 | 275.00 | 274.34 | 277.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 15:00:00 | 275.00 | 274.34 | 277.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 273.20 | 274.18 | 276.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:30:00 | 268.15 | 273.29 | 275.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 10:45:00 | 269.00 | 271.76 | 273.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:15:00 | 269.00 | 271.76 | 273.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 269.05 | 271.77 | 273.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 270.10 | 271.36 | 273.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:30:00 | 271.50 | 271.36 | 273.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 275.00 | 272.09 | 273.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:15:00 | 271.50 | 272.09 | 273.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 267.55 | 271.18 | 272.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 14:45:00 | 264.10 | 268.58 | 270.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:30:00 | 264.50 | 267.05 | 268.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:30:00 | 263.00 | 266.64 | 268.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 12:30:00 | 263.35 | 266.31 | 267.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 266.00 | 265.78 | 267.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 266.00 | 265.78 | 267.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 267.95 | 266.21 | 267.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 265.00 | 266.21 | 267.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 271.85 | 267.34 | 267.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 274.00 | 267.34 | 267.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 268.50 | 267.57 | 267.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:15:00 | 265.10 | 267.57 | 267.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:00:00 | 264.05 | 266.94 | 267.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:15:00 | 265.00 | 266.82 | 267.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 277.75 | 268.71 | 268.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 277.75 | 268.71 | 268.15 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 263.00 | 268.78 | 269.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 255.00 | 263.77 | 266.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 261.00 | 260.10 | 263.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:30:00 | 260.00 | 260.10 | 263.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 267.00 | 261.48 | 263.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 271.90 | 261.48 | 263.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 271.60 | 263.50 | 264.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 271.80 | 263.50 | 264.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 272.30 | 265.26 | 265.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 279.95 | 271.96 | 268.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 10:15:00 | 314.00 | 314.32 | 309.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 13:15:00 | 311.00 | 313.11 | 310.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 311.00 | 313.11 | 310.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 13:45:00 | 313.25 | 313.11 | 310.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 316.80 | 313.85 | 310.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 316.80 | 313.85 | 310.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 343.90 | 343.36 | 334.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 347.00 | 344.00 | 337.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 355.95 | 357.04 | 357.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 355.95 | 357.04 | 357.19 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 359.95 | 357.62 | 357.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 10:15:00 | 363.75 | 359.88 | 358.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 12:15:00 | 377.05 | 377.11 | 372.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 12:30:00 | 377.05 | 377.11 | 372.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 378.00 | 377.21 | 373.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 373.40 | 377.21 | 373.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 372.95 | 376.36 | 373.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 370.55 | 376.36 | 373.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 365.00 | 374.09 | 372.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 369.70 | 374.09 | 372.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 364.85 | 372.24 | 372.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:45:00 | 361.90 | 372.24 | 372.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 12:15:00 | 359.00 | 369.59 | 370.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 339.90 | 359.25 | 365.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 09:15:00 | 344.85 | 335.98 | 340.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 344.85 | 335.98 | 340.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 344.85 | 335.98 | 340.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:30:00 | 345.00 | 335.98 | 340.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 338.00 | 336.38 | 340.34 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 350.00 | 342.24 | 341.79 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 340.00 | 343.83 | 344.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 323.05 | 339.08 | 341.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 335.05 | 333.00 | 337.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 13:15:00 | 335.05 | 333.00 | 337.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 335.05 | 333.00 | 337.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 335.00 | 333.00 | 337.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 335.50 | 333.50 | 337.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:45:00 | 344.40 | 333.50 | 337.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 331.70 | 332.14 | 335.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 331.70 | 332.14 | 335.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 317.80 | 311.48 | 316.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 318.90 | 311.48 | 316.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 323.20 | 313.83 | 317.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 323.20 | 313.83 | 317.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 320.00 | 315.06 | 317.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 323.00 | 315.06 | 317.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 319.00 | 315.85 | 317.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:15:00 | 313.05 | 316.68 | 317.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:00:00 | 314.00 | 316.14 | 317.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:30:00 | 318.70 | 315.51 | 316.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:45:00 | 315.10 | 315.28 | 316.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 318.75 | 315.97 | 316.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 319.00 | 315.97 | 316.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 318.00 | 316.38 | 316.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 322.55 | 317.87 | 317.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 322.55 | 317.87 | 317.41 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 10:15:00 | 321.00 | 323.03 | 323.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 11:15:00 | 313.00 | 321.03 | 322.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 09:15:00 | 311.90 | 310.73 | 314.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 311.90 | 310.73 | 314.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 311.90 | 310.73 | 314.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 303.00 | 311.01 | 311.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 11:00:00 | 304.00 | 308.84 | 310.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 09:15:00 | 288.80 | 303.82 | 306.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 300.00 | 303.73 | 306.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:00:00 | 303.35 | 303.73 | 306.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-07 09:15:00 | 288.18 | 296.54 | 301.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 299.25 | 296.84 | 300.63 | SL hit (close>ema200) qty=0.50 sl=296.84 alert=retest2 |

### Cycle 91 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 311.60 | 301.67 | 301.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 327.15 | 311.89 | 306.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 10:15:00 | 350.00 | 356.34 | 346.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 11:00:00 | 350.00 | 356.34 | 346.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 347.85 | 354.64 | 346.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:30:00 | 346.05 | 354.64 | 346.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 355.00 | 354.71 | 347.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:30:00 | 347.65 | 354.71 | 347.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 342.10 | 353.42 | 349.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 342.10 | 353.42 | 349.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 342.10 | 351.16 | 348.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 342.10 | 351.16 | 348.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 13:15:00 | 342.10 | 346.74 | 347.11 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 352.05 | 347.58 | 347.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 359.20 | 351.04 | 349.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 367.05 | 368.76 | 362.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:45:00 | 371.25 | 368.76 | 362.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 364.95 | 369.47 | 365.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 364.95 | 369.47 | 365.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 363.00 | 368.17 | 365.42 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 359.50 | 363.56 | 363.85 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 368.00 | 364.18 | 364.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 369.95 | 367.75 | 366.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 12:15:00 | 371.05 | 371.20 | 369.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 13:00:00 | 371.05 | 371.20 | 369.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 371.35 | 371.91 | 370.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:15:00 | 371.00 | 371.91 | 370.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 371.00 | 371.73 | 370.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 15:00:00 | 375.50 | 372.20 | 371.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:00:00 | 379.65 | 381.66 | 380.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 13:00:00 | 376.00 | 380.53 | 379.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 373.10 | 378.16 | 378.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 373.10 | 378.16 | 378.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 366.00 | 375.54 | 377.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 12:15:00 | 378.85 | 375.23 | 376.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 12:15:00 | 378.85 | 375.23 | 376.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 378.85 | 375.23 | 376.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:45:00 | 374.00 | 375.23 | 376.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 376.00 | 375.39 | 376.68 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 379.50 | 377.47 | 377.35 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 375.00 | 377.42 | 377.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 15:15:00 | 362.10 | 368.52 | 370.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 375.45 | 369.91 | 370.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 375.45 | 369.91 | 370.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 375.45 | 369.91 | 370.76 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 375.00 | 371.37 | 371.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 10:15:00 | 376.20 | 373.42 | 372.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 375.05 | 375.87 | 374.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 375.05 | 375.87 | 374.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 375.05 | 375.87 | 374.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:30:00 | 380.00 | 375.87 | 374.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 379.00 | 378.07 | 376.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 386.00 | 381.17 | 378.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:45:00 | 384.95 | 381.94 | 379.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 11:30:00 | 386.00 | 382.75 | 379.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 11:30:00 | 389.65 | 385.23 | 383.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 384.40 | 385.06 | 383.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:00:00 | 393.40 | 386.56 | 384.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 393.00 | 391.84 | 388.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 393.85 | 390.38 | 388.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:45:00 | 391.85 | 389.90 | 388.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 391.50 | 390.22 | 388.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:30:00 | 390.65 | 390.22 | 388.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 391.85 | 390.55 | 388.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:15:00 | 392.00 | 389.66 | 388.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 384.00 | 387.90 | 388.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 384.00 | 387.90 | 388.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 11:15:00 | 383.00 | 386.29 | 387.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 11:15:00 | 382.70 | 382.48 | 384.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:30:00 | 383.40 | 382.48 | 384.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 384.75 | 383.03 | 384.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 388.60 | 383.03 | 384.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 384.95 | 383.42 | 384.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:15:00 | 376.10 | 383.52 | 384.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 377.15 | 383.52 | 384.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:00:00 | 377.25 | 382.27 | 383.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 14:15:00 | 376.80 | 382.01 | 383.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 377.95 | 379.46 | 381.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 363.50 | 372.29 | 376.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 357.30 | 369.06 | 374.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 358.29 | 369.06 | 374.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 358.39 | 369.06 | 374.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 357.96 | 369.06 | 374.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 362.00 | 361.20 | 366.23 | SL hit (close>ema200) qty=0.50 sl=361.20 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 358.05 | 352.51 | 352.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 11:15:00 | 362.00 | 357.78 | 355.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 359.00 | 362.32 | 359.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 359.00 | 362.32 | 359.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 359.00 | 362.32 | 359.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 359.00 | 362.32 | 359.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 361.50 | 362.16 | 359.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:30:00 | 369.85 | 362.03 | 359.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 14:15:00 | 356.10 | 360.76 | 359.35 | SL hit (close<static) qty=1.00 sl=359.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 356.55 | 358.82 | 358.92 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 359.95 | 359.05 | 359.01 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 356.50 | 358.54 | 358.78 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 361.50 | 359.13 | 359.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 361.95 | 359.69 | 359.30 | Break + close above crossover candle high |

### Cycle 106 — SELL (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 10:15:00 | 356.20 | 359.00 | 359.02 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 363.05 | 359.81 | 359.38 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 355.00 | 358.48 | 358.88 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 359.75 | 358.32 | 358.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 364.70 | 359.86 | 358.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 355.50 | 359.16 | 358.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 11:15:00 | 355.50 | 359.16 | 358.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 355.50 | 359.16 | 358.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 355.50 | 359.16 | 358.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 355.50 | 358.43 | 358.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 349.05 | 355.54 | 357.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 10:15:00 | 349.95 | 344.59 | 347.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 10:15:00 | 349.95 | 344.59 | 347.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 349.95 | 344.59 | 347.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:45:00 | 345.40 | 344.59 | 347.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 346.70 | 345.01 | 347.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 338.50 | 345.01 | 347.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:00:00 | 337.00 | 343.01 | 345.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 337.00 | 343.17 | 345.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 321.57 | 333.10 | 337.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 320.15 | 333.10 | 337.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 320.15 | 333.10 | 337.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 12:15:00 | 325.00 | 324.41 | 329.24 | SL hit (close>ema200) qty=0.50 sl=324.41 alert=retest2 |

### Cycle 111 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 328.70 | 323.85 | 323.82 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 15:15:00 | 321.00 | 323.28 | 323.56 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 333.80 | 325.39 | 324.50 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 325.00 | 326.15 | 326.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 323.00 | 325.49 | 325.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 15:15:00 | 319.95 | 318.07 | 320.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 09:15:00 | 318.00 | 318.07 | 320.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 310.45 | 316.54 | 319.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:45:00 | 300.90 | 311.19 | 315.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:15:00 | 304.30 | 311.19 | 315.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 11:45:00 | 304.50 | 308.26 | 313.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:00:00 | 301.50 | 306.29 | 311.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 304.60 | 304.39 | 309.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:30:00 | 300.00 | 304.41 | 308.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:45:00 | 300.00 | 303.53 | 307.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:45:00 | 300.65 | 303.02 | 306.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 301.00 | 303.28 | 306.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 302.75 | 303.18 | 306.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 322.70 | 309.24 | 307.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 322.70 | 309.24 | 307.84 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 307.75 | 310.44 | 310.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 13:15:00 | 304.00 | 309.15 | 310.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 15:15:00 | 311.50 | 308.80 | 309.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 15:15:00 | 311.50 | 308.80 | 309.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 311.50 | 308.80 | 309.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:15:00 | 307.80 | 308.80 | 309.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 315.25 | 310.09 | 310.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 315.25 | 310.09 | 310.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 312.45 | 310.56 | 310.48 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 309.65 | 310.43 | 310.44 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 312.05 | 310.75 | 310.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 320.95 | 312.79 | 311.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 318.05 | 318.27 | 315.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:45:00 | 317.40 | 318.27 | 315.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 318.60 | 318.34 | 316.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 11:15:00 | 321.05 | 318.34 | 316.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 11:45:00 | 319.05 | 318.45 | 316.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 15:15:00 | 322.00 | 318.56 | 316.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:30:00 | 320.65 | 319.54 | 317.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 336.90 | 330.34 | 326.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:30:00 | 341.85 | 333.85 | 328.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 09:15:00 | 353.16 | 346.14 | 337.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 11:15:00 | 375.10 | 377.86 | 378.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 12:15:00 | 374.85 | 377.26 | 377.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 14:15:00 | 377.00 | 376.83 | 377.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 15:00:00 | 377.00 | 376.83 | 377.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 360.05 | 373.50 | 375.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:15:00 | 359.00 | 373.50 | 375.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 15:15:00 | 360.00 | 367.84 | 371.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 11:30:00 | 360.00 | 366.36 | 369.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:45:00 | 360.00 | 363.55 | 367.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 362.15 | 362.54 | 366.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:45:00 | 355.15 | 360.39 | 363.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:30:00 | 355.10 | 360.32 | 363.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 354.00 | 360.25 | 363.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 13:15:00 | 356.50 | 358.08 | 361.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 341.05 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 342.00 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 342.00 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 342.00 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 337.39 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 337.35 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 336.30 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:15:00 | 338.68 | 353.97 | 357.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 343.95 | 347.37 | 352.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 352.50 | 347.71 | 350.97 | SL hit (close>ema200) qty=0.50 sl=347.71 alert=retest2 |

### Cycle 121 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 354.35 | 352.45 | 352.29 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 12:15:00 | 350.10 | 351.98 | 352.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 13:15:00 | 349.75 | 351.53 | 351.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 349.90 | 348.98 | 350.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:45:00 | 350.00 | 348.98 | 350.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 348.00 | 348.67 | 349.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 349.95 | 348.67 | 349.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 348.90 | 348.72 | 349.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:45:00 | 341.25 | 345.96 | 348.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 352.30 | 348.49 | 348.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 13:15:00 | 352.30 | 348.49 | 348.25 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 13:15:00 | 346.15 | 348.09 | 348.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 14:15:00 | 345.00 | 347.47 | 348.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 345.00 | 344.28 | 345.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 12:15:00 | 345.00 | 344.28 | 345.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 345.00 | 344.28 | 345.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:00:00 | 345.00 | 344.28 | 345.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 351.80 | 345.92 | 346.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 351.80 | 345.92 | 346.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 348.00 | 346.33 | 346.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 344.55 | 346.33 | 346.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 346.20 | 346.25 | 346.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 346.20 | 346.25 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 346.30 | 346.26 | 346.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:15:00 | 348.70 | 346.26 | 346.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 346.50 | 346.31 | 346.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 348.50 | 346.31 | 346.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 346.95 | 346.44 | 346.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 346.95 | 346.44 | 346.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 350.00 | 347.15 | 346.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 15:15:00 | 353.90 | 348.50 | 347.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 341.85 | 347.17 | 346.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 341.85 | 347.17 | 346.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 341.85 | 347.17 | 346.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 341.85 | 347.17 | 346.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 341.70 | 346.08 | 346.45 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 13:15:00 | 348.85 | 346.95 | 346.76 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 342.85 | 346.13 | 346.40 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 347.00 | 346.40 | 346.36 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 345.75 | 346.27 | 346.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 345.45 | 346.10 | 346.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 09:15:00 | 346.05 | 345.94 | 346.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 346.05 | 345.94 | 346.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 346.05 | 345.94 | 346.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 346.05 | 345.94 | 346.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 343.95 | 345.54 | 345.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:15:00 | 353.45 | 345.54 | 345.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 353.55 | 347.14 | 346.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 359.05 | 353.35 | 350.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 357.00 | 358.73 | 355.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 11:15:00 | 357.00 | 358.73 | 355.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 357.00 | 358.73 | 355.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 357.00 | 358.73 | 355.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 360.00 | 358.98 | 355.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:45:00 | 360.95 | 358.98 | 355.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 348.15 | 357.73 | 356.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 345.15 | 357.73 | 356.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 343.50 | 354.88 | 355.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 343.00 | 352.50 | 354.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 10:15:00 | 340.80 | 340.19 | 344.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 10:15:00 | 340.80 | 340.19 | 344.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 340.80 | 340.19 | 344.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 343.50 | 340.19 | 344.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 337.05 | 337.41 | 341.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 337.00 | 337.41 | 341.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 340.00 | 338.63 | 341.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:30:00 | 340.90 | 338.63 | 341.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 341.75 | 339.26 | 341.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:45:00 | 341.75 | 339.26 | 341.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 340.30 | 339.47 | 341.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:45:00 | 338.10 | 339.08 | 340.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 15:15:00 | 345.85 | 340.58 | 341.10 | SL hit (close>static) qty=1.00 sl=341.75 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 327.00 | 320.31 | 320.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 334.00 | 323.88 | 321.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 324.80 | 327.52 | 324.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 324.80 | 327.52 | 324.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 324.80 | 327.52 | 324.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 324.80 | 327.52 | 324.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 318.00 | 325.62 | 324.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 318.00 | 325.62 | 324.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 317.95 | 324.08 | 323.70 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 316.85 | 322.64 | 323.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 310.00 | 317.87 | 320.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 311.70 | 311.41 | 315.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 311.70 | 311.41 | 315.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 317.75 | 312.68 | 315.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 321.45 | 312.68 | 315.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 320.55 | 314.25 | 316.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 13:30:00 | 313.75 | 316.51 | 317.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:15:00 | 315.10 | 314.95 | 315.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 298.06 | 306.81 | 310.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 299.35 | 306.81 | 310.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 303.55 | 302.45 | 305.73 | SL hit (close>ema200) qty=0.50 sl=302.45 alert=retest2 |

### Cycle 135 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 316.30 | 307.08 | 306.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 320.35 | 315.04 | 311.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 316.25 | 320.86 | 317.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 316.25 | 320.86 | 317.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 316.25 | 320.86 | 317.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 316.05 | 320.86 | 317.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 317.00 | 320.09 | 317.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 317.00 | 320.09 | 317.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 311.10 | 318.29 | 317.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:45:00 | 311.25 | 318.29 | 317.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 310.85 | 315.65 | 316.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 307.60 | 312.26 | 314.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 316.45 | 311.42 | 312.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 14:15:00 | 316.45 | 311.42 | 312.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 316.45 | 311.42 | 312.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 316.45 | 311.42 | 312.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 319.95 | 313.13 | 313.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 312.90 | 313.13 | 313.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 325.70 | 315.64 | 314.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 14:15:00 | 332.45 | 321.66 | 318.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 15:15:00 | 348.75 | 348.95 | 342.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 09:15:00 | 345.45 | 348.95 | 342.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 345.65 | 347.69 | 342.68 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 325.50 | 337.97 | 339.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 322.45 | 331.97 | 336.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 320.70 | 319.02 | 326.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 322.50 | 319.02 | 326.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 317.75 | 318.83 | 323.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:00:00 | 313.25 | 317.72 | 322.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 312.60 | 316.55 | 321.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:00:00 | 311.90 | 316.55 | 321.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:30:00 | 313.05 | 315.14 | 319.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 318.00 | 311.06 | 314.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 303.30 | 309.51 | 313.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 320.00 | 313.38 | 313.84 | SL hit (close>static) qty=1.00 sl=318.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 320.15 | 313.38 | 312.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 15:15:00 | 322.60 | 318.02 | 315.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 323.05 | 323.84 | 320.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 323.05 | 323.84 | 320.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 323.05 | 323.84 | 320.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 13:45:00 | 334.90 | 327.08 | 323.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 318.50 | 323.27 | 323.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 318.50 | 323.27 | 323.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 315.80 | 320.18 | 322.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 309.55 | 305.39 | 309.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 309.55 | 305.39 | 309.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 309.55 | 305.39 | 309.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 309.55 | 305.39 | 309.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 319.00 | 308.11 | 310.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 319.00 | 308.11 | 310.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 312.45 | 308.98 | 310.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:45:00 | 317.40 | 308.98 | 310.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 314.75 | 311.49 | 311.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 324.15 | 314.54 | 312.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 338.70 | 339.19 | 332.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 338.70 | 339.19 | 332.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 330.90 | 336.41 | 333.87 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 326.20 | 332.35 | 332.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 325.15 | 330.91 | 331.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 316.15 | 312.99 | 315.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 316.15 | 312.99 | 315.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 316.15 | 312.99 | 315.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:15:00 | 314.00 | 313.82 | 315.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:30:00 | 314.05 | 313.48 | 315.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 313.45 | 314.08 | 314.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 14:30:00 | 312.95 | 314.28 | 314.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 315.55 | 314.54 | 314.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 324.10 | 314.54 | 314.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 323.00 | 316.23 | 315.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 323.00 | 316.23 | 315.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 336.45 | 324.73 | 320.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 15:15:00 | 370.30 | 371.28 | 364.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 09:15:00 | 377.50 | 371.28 | 364.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 353.00 | 369.59 | 367.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 353.00 | 369.59 | 367.06 | SL hit (close<ema400) qty=1.00 sl=367.06 alert=retest1 |

### Cycle 144 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 382.65 | 412.14 | 414.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 375.70 | 404.85 | 411.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 398.35 | 397.03 | 405.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 398.35 | 397.03 | 405.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 410.55 | 399.39 | 405.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 400.00 | 402.59 | 404.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 421.90 | 402.72 | 402.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 421.90 | 402.72 | 402.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 428.70 | 407.92 | 404.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 463.85 | 467.61 | 456.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:00:00 | 463.85 | 467.61 | 456.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 490.95 | 470.94 | 462.14 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 465.90 | 467.71 | 467.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 463.05 | 466.49 | 467.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 454.15 | 450.81 | 455.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 454.15 | 450.81 | 455.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 454.15 | 450.81 | 455.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 454.15 | 450.81 | 455.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 456.80 | 452.01 | 455.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:30:00 | 456.20 | 452.01 | 455.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 456.50 | 452.91 | 455.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 455.55 | 452.91 | 455.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 456.85 | 453.70 | 455.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 457.25 | 453.70 | 455.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 458.35 | 454.63 | 456.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:45:00 | 458.25 | 454.63 | 456.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 454.55 | 454.61 | 455.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 15:15:00 | 453.00 | 454.61 | 455.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 447.25 | 452.85 | 454.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 456.60 | 443.52 | 443.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 456.60 | 443.52 | 443.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 464.80 | 453.57 | 448.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 452.10 | 456.28 | 451.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 452.10 | 456.28 | 451.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 452.10 | 456.28 | 451.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 452.50 | 456.28 | 451.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 447.65 | 454.55 | 451.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 447.65 | 454.55 | 451.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 442.10 | 452.06 | 450.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 442.10 | 452.06 | 450.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 434.55 | 448.56 | 449.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 427.35 | 442.31 | 446.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 441.35 | 440.24 | 443.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 13:15:00 | 441.35 | 440.24 | 443.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 441.35 | 440.24 | 443.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:45:00 | 440.00 | 440.24 | 443.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 442.00 | 440.59 | 443.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 442.00 | 440.59 | 443.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 444.95 | 441.46 | 443.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 439.65 | 441.46 | 443.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 440.55 | 441.28 | 443.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 432.30 | 439.48 | 441.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 431.35 | 436.48 | 440.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 410.69 | 429.90 | 436.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 409.78 | 429.90 | 436.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 450.00 | 425.99 | 429.87 | SL hit (close>ema200) qty=0.50 sl=425.99 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 458.60 | 437.33 | 434.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 463.70 | 446.18 | 439.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 462.60 | 463.56 | 453.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 460.65 | 463.56 | 453.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 467.50 | 472.30 | 466.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 468.40 | 472.30 | 466.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 464.05 | 470.65 | 466.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 463.10 | 470.65 | 466.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 464.05 | 469.33 | 466.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 466.90 | 469.33 | 466.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 467.70 | 467.59 | 466.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 470.00 | 469.18 | 467.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 467.40 | 472.71 | 472.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 13:15:00 | 470.05 | 472.18 | 472.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 470.05 | 472.18 | 472.43 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 479.00 | 473.69 | 473.08 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 470.65 | 472.80 | 472.80 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 473.10 | 472.80 | 472.79 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 472.10 | 472.66 | 472.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 10:15:00 | 470.70 | 472.27 | 472.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 472.55 | 472.33 | 472.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 472.55 | 472.33 | 472.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 472.55 | 472.33 | 472.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 474.90 | 472.33 | 472.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 475.35 | 472.93 | 472.80 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 471.25 | 472.89 | 472.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 467.55 | 471.53 | 472.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 458.80 | 458.30 | 463.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:00:00 | 458.80 | 458.30 | 463.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 459.00 | 459.00 | 462.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 461.05 | 459.00 | 462.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 459.50 | 459.12 | 461.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 456.45 | 459.12 | 461.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 457.00 | 458.70 | 461.10 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 465.15 | 462.43 | 462.10 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 448.85 | 460.96 | 462.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 444.75 | 450.99 | 455.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 441.45 | 437.59 | 440.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 441.45 | 437.59 | 440.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 441.45 | 437.59 | 440.91 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 450.75 | 443.46 | 442.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 464.10 | 449.56 | 446.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 465.00 | 465.20 | 458.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:15:00 | 466.65 | 465.20 | 458.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 458.00 | 464.79 | 462.19 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 450.60 | 458.76 | 459.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 449.80 | 455.77 | 458.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 455.50 | 453.61 | 455.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 455.50 | 453.61 | 455.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 455.20 | 453.93 | 455.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 454.60 | 453.93 | 455.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 453.35 | 454.11 | 455.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 459.40 | 449.13 | 450.12 | SL hit (close>static) qty=1.00 sl=456.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 460.45 | 451.77 | 451.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 13:15:00 | 462.65 | 455.41 | 452.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 544.80 | 546.52 | 532.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 540.10 | 546.52 | 532.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 540.05 | 542.34 | 535.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:00:00 | 540.05 | 542.34 | 535.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 537.45 | 541.36 | 535.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 539.85 | 541.36 | 535.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 536.80 | 540.45 | 536.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 541.40 | 540.45 | 536.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 539.05 | 540.17 | 536.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 533.50 | 538.84 | 536.03 | SL hit (close<static) qty=1.00 sl=535.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 12:15:00 | 525.50 | 534.27 | 534.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 523.55 | 528.22 | 530.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 529.15 | 528.41 | 530.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 529.15 | 528.41 | 530.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 525.10 | 527.99 | 529.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 521.60 | 526.47 | 528.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 523.00 | 525.88 | 527.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 550.00 | 530.07 | 529.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 550.00 | 530.07 | 529.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 555.65 | 537.58 | 532.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 549.60 | 549.62 | 542.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:45:00 | 550.65 | 549.62 | 542.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 542.95 | 546.80 | 542.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:30:00 | 540.70 | 546.80 | 542.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 546.60 | 546.76 | 542.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 549.85 | 546.76 | 542.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 540.00 | 548.11 | 546.53 | SL hit (close<static) qty=1.00 sl=540.25 alert=retest2 |

### Cycle 164 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 541.75 | 545.34 | 545.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 13:15:00 | 540.90 | 544.45 | 545.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 547.10 | 544.98 | 545.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 547.10 | 544.98 | 545.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 547.10 | 544.98 | 545.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 547.10 | 544.98 | 545.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 549.90 | 545.97 | 545.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 553.00 | 547.38 | 546.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 13:15:00 | 579.00 | 579.09 | 565.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 579.00 | 579.09 | 565.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 579.55 | 583.22 | 579.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 579.55 | 583.22 | 579.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 575.95 | 581.76 | 578.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 575.95 | 581.76 | 578.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 575.60 | 580.53 | 578.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 575.60 | 580.53 | 578.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 579.85 | 580.19 | 578.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 574.50 | 580.19 | 578.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 581.90 | 580.53 | 579.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:30:00 | 586.00 | 581.13 | 579.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:00:00 | 583.35 | 581.57 | 579.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 13:45:00 | 583.00 | 581.51 | 579.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 584.50 | 582.10 | 580.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 582.85 | 582.69 | 580.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 576.95 | 582.69 | 580.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 587.70 | 588.15 | 585.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:45:00 | 592.00 | 589.16 | 586.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:45:00 | 591.35 | 588.66 | 586.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 593.85 | 586.25 | 586.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-21 10:15:00 | 641.69 | 598.88 | 592.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 758.15 | 771.54 | 772.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 750.50 | 762.70 | 766.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 768.30 | 763.82 | 767.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 768.30 | 763.82 | 767.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 768.30 | 763.82 | 767.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 766.60 | 763.82 | 767.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 768.85 | 764.83 | 767.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 775.05 | 764.83 | 767.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 768.35 | 766.02 | 767.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 768.35 | 766.02 | 767.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 770.30 | 766.87 | 767.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 770.30 | 766.87 | 767.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 775.50 | 768.60 | 768.39 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 757.00 | 767.61 | 768.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 750.55 | 764.20 | 766.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 633.45 | 628.53 | 644.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 633.45 | 628.53 | 644.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 636.85 | 629.95 | 642.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 638.70 | 629.95 | 642.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 636.35 | 632.61 | 641.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:30:00 | 640.75 | 632.61 | 641.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 639.85 | 633.48 | 639.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 639.85 | 633.48 | 639.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 639.70 | 634.73 | 639.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 632.40 | 634.73 | 639.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 600.78 | 609.79 | 617.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 585.80 | 582.80 | 589.91 | SL hit (close>ema200) qty=0.50 sl=582.80 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 616.15 | 592.94 | 592.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 638.65 | 619.00 | 613.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 15:15:00 | 725.95 | 728.75 | 710.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:15:00 | 721.05 | 728.75 | 710.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 709.85 | 722.05 | 711.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 709.85 | 722.05 | 711.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 714.10 | 720.46 | 712.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 709.35 | 720.46 | 712.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 708.45 | 717.17 | 711.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 708.25 | 717.17 | 711.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 710.00 | 715.74 | 711.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 706.25 | 715.74 | 711.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 711.20 | 714.83 | 711.70 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 677.75 | 705.24 | 707.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 676.40 | 692.70 | 700.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 681.90 | 681.90 | 690.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:15:00 | 679.55 | 681.90 | 690.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 662.70 | 668.02 | 673.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:15:00 | 660.60 | 664.23 | 667.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 660.70 | 663.53 | 666.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 685.00 | 667.58 | 668.07 | SL hit (close>static) qty=1.00 sl=674.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 681.55 | 670.37 | 669.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 689.75 | 676.57 | 672.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 671.00 | 677.92 | 673.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 671.00 | 677.92 | 673.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 671.00 | 677.92 | 673.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 674.40 | 677.92 | 673.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 669.65 | 676.26 | 673.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 666.80 | 676.26 | 673.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 660.80 | 671.21 | 671.59 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 675.10 | 670.45 | 670.42 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 668.25 | 670.01 | 670.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 663.85 | 668.95 | 669.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 657.90 | 657.48 | 661.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 657.90 | 657.48 | 661.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 657.90 | 657.48 | 661.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 662.85 | 657.48 | 661.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 668.35 | 658.79 | 660.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 672.35 | 658.79 | 660.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 660.20 | 659.07 | 660.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 656.50 | 659.07 | 660.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 658.50 | 654.76 | 656.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 656.60 | 655.57 | 656.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 664.35 | 658.63 | 657.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 664.35 | 658.63 | 657.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 667.85 | 660.48 | 658.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 667.35 | 671.68 | 666.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 667.35 | 671.68 | 666.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 667.35 | 671.68 | 666.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 670.10 | 671.68 | 666.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 669.40 | 671.22 | 667.09 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 652.55 | 665.76 | 665.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 651.05 | 658.43 | 661.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 658.85 | 658.52 | 661.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 658.85 | 658.52 | 661.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 663.75 | 656.01 | 658.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 664.60 | 656.01 | 658.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 667.85 | 658.38 | 658.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 667.35 | 658.38 | 658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 665.50 | 659.80 | 659.55 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 657.95 | 660.39 | 660.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 656.10 | 659.53 | 660.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 12:15:00 | 657.30 | 655.92 | 657.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 12:15:00 | 657.30 | 655.92 | 657.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 657.30 | 655.92 | 657.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 657.30 | 655.92 | 657.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 657.30 | 656.20 | 657.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 656.90 | 656.20 | 657.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 655.40 | 656.04 | 657.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 652.95 | 654.82 | 656.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:30:00 | 652.45 | 651.88 | 653.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 661.30 | 653.43 | 653.70 | SL hit (close>static) qty=1.00 sl=658.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 656.00 | 653.94 | 653.91 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 651.15 | 655.04 | 655.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 638.10 | 651.15 | 653.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 526.45 | 522.91 | 538.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:45:00 | 526.60 | 522.91 | 538.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 537.20 | 525.77 | 538.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 537.20 | 525.77 | 538.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 530.85 | 526.78 | 537.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 532.60 | 526.78 | 537.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 536.60 | 528.75 | 537.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 536.60 | 528.75 | 537.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 533.75 | 529.75 | 537.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 532.85 | 529.75 | 537.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 533.10 | 531.51 | 536.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 532.95 | 531.51 | 536.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 539.30 | 533.07 | 536.92 | SL hit (close>static) qty=1.00 sl=538.50 alert=retest2 |

### Cycle 181 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 542.25 | 529.51 | 529.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 544.55 | 534.13 | 531.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 552.90 | 552.99 | 543.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:00:00 | 552.90 | 552.99 | 543.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 561.80 | 553.21 | 545.90 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 531.20 | 541.82 | 542.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 523.40 | 535.93 | 539.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 534.65 | 531.25 | 535.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 534.65 | 531.25 | 535.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 534.65 | 531.25 | 535.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 539.00 | 531.25 | 535.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 537.65 | 532.53 | 535.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 537.65 | 532.53 | 535.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 545.00 | 535.03 | 536.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 545.00 | 535.03 | 536.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 554.95 | 539.01 | 538.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 621.00 | 556.93 | 546.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 647.50 | 656.34 | 629.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 647.50 | 656.34 | 629.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 624.15 | 649.90 | 629.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 624.15 | 649.90 | 629.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 623.60 | 644.64 | 628.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 620.50 | 644.64 | 628.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 598.80 | 618.64 | 621.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 593.65 | 606.16 | 613.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 597.30 | 597.13 | 604.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 597.30 | 597.13 | 604.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 588.60 | 595.21 | 602.25 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 636.30 | 607.41 | 605.09 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 609.15 | 615.05 | 615.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 601.25 | 612.29 | 614.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 593.95 | 593.25 | 599.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 593.95 | 593.25 | 599.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 591.70 | 592.37 | 597.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 597.10 | 592.37 | 597.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 591.55 | 590.40 | 593.92 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 15:15:00 | 595.90 | 593.38 | 593.36 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 592.25 | 593.28 | 593.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 590.50 | 592.72 | 593.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 592.85 | 592.41 | 592.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 592.85 | 592.41 | 592.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 592.85 | 592.41 | 592.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 592.75 | 592.41 | 592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 588.55 | 591.64 | 592.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 589.95 | 591.64 | 592.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 590.50 | 591.41 | 592.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 590.50 | 591.41 | 592.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 592.35 | 591.60 | 592.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 591.50 | 591.60 | 592.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 590.80 | 591.44 | 592.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 590.10 | 591.51 | 592.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 590.00 | 591.51 | 592.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 590.55 | 591.71 | 592.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:15:00 | 590.40 | 591.71 | 592.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 590.75 | 591.52 | 591.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 593.75 | 592.21 | 592.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 593.75 | 592.21 | 592.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 09:15:00 | 595.90 | 592.95 | 592.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 11:15:00 | 592.05 | 592.78 | 592.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 592.05 | 592.78 | 592.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 592.05 | 592.78 | 592.53 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 15:15:00 | 590.80 | 592.12 | 592.29 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 595.10 | 592.71 | 592.55 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 590.00 | 592.20 | 592.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 587.85 | 591.33 | 591.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 592.30 | 591.31 | 591.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 592.30 | 591.31 | 591.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 592.30 | 591.31 | 591.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 589.05 | 591.12 | 591.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 588.55 | 591.12 | 591.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:30:00 | 589.10 | 590.80 | 591.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 597.45 | 591.76 | 591.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 597.45 | 591.76 | 591.58 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 592.30 | 596.08 | 596.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 591.85 | 595.24 | 595.91 | Break + close below crossover candle low |

### Cycle 195 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 604.20 | 596.79 | 596.48 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 595.45 | 596.32 | 596.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 592.80 | 595.44 | 595.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 592.00 | 591.35 | 592.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 591.95 | 591.35 | 592.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 592.20 | 591.52 | 592.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 592.20 | 591.52 | 592.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 596.05 | 592.43 | 593.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 596.80 | 592.43 | 593.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 599.60 | 593.86 | 593.74 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 555.40 | 587.03 | 590.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 526.80 | 574.98 | 585.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 524.25 | 519.20 | 530.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 524.25 | 519.20 | 530.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 524.85 | 520.90 | 528.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 516.85 | 521.24 | 525.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 526.60 | 523.10 | 522.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 526.60 | 523.10 | 522.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 15:15:00 | 530.00 | 525.21 | 523.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 521.80 | 524.53 | 523.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 521.80 | 524.53 | 523.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 521.80 | 524.53 | 523.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 520.20 | 524.53 | 523.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 521.00 | 523.82 | 523.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 521.00 | 523.82 | 523.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 521.05 | 523.55 | 523.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 521.05 | 523.55 | 523.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 522.10 | 523.26 | 523.24 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 516.55 | 521.92 | 522.63 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 10:15:00 | 528.45 | 522.89 | 522.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 539.20 | 526.16 | 524.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 15:15:00 | 528.05 | 529.29 | 526.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 528.05 | 529.29 | 526.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 528.05 | 529.29 | 526.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 523.50 | 529.29 | 526.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 524.55 | 528.34 | 526.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 529.50 | 527.48 | 526.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 535.35 | 527.99 | 527.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 560.05 | 567.68 | 568.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 12:15:00 | 560.05 | 567.68 | 568.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 11:15:00 | 558.00 | 563.14 | 565.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 561.15 | 560.20 | 563.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 561.15 | 560.20 | 563.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 555.15 | 552.89 | 557.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 15:00:00 | 555.15 | 552.89 | 557.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 565.00 | 555.32 | 557.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 550.20 | 556.64 | 557.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 562.30 | 558.60 | 558.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 562.30 | 558.60 | 558.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 567.30 | 560.54 | 559.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 563.05 | 564.58 | 561.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 11:00:00 | 563.05 | 564.58 | 561.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 560.55 | 563.77 | 561.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:45:00 | 562.45 | 563.77 | 561.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 556.75 | 562.37 | 561.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 556.75 | 562.37 | 561.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 552.10 | 560.31 | 560.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 550.00 | 556.98 | 558.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 551.20 | 548.30 | 552.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 551.20 | 548.30 | 552.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 551.20 | 548.30 | 552.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 552.35 | 548.30 | 552.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 535.35 | 536.73 | 541.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 527.55 | 536.73 | 541.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 501.17 | 513.44 | 517.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 516.80 | 513.45 | 516.00 | SL hit (close>ema200) qty=0.50 sl=513.45 alert=retest2 |

### Cycle 205 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 531.75 | 518.19 | 517.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 540.00 | 522.55 | 519.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 591.00 | 591.07 | 576.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 554.20 | 583.61 | 580.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 554.20 | 583.61 | 580.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 554.20 | 583.61 | 580.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 557.70 | 578.42 | 578.31 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 11:15:00 | 561.20 | 574.98 | 576.76 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 590.00 | 577.57 | 576.60 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 576.05 | 577.31 | 577.39 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 580.00 | 577.85 | 577.63 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 575.00 | 577.45 | 577.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 568.15 | 575.59 | 576.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 552.40 | 550.72 | 558.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 554.05 | 550.72 | 558.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 569.20 | 553.36 | 555.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 566.85 | 553.36 | 555.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 571.80 | 557.04 | 556.86 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 554.85 | 565.31 | 566.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 548.05 | 556.97 | 561.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 552.00 | 551.12 | 556.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 552.00 | 551.12 | 556.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 552.00 | 551.12 | 556.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 552.00 | 551.12 | 556.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 560.00 | 553.07 | 556.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 560.00 | 553.07 | 556.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 558.10 | 554.08 | 556.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 560.70 | 554.08 | 556.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 558.00 | 556.16 | 556.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 558.15 | 556.16 | 556.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 559.40 | 556.81 | 557.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 559.40 | 556.81 | 557.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 558.45 | 557.14 | 557.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 563.95 | 557.14 | 557.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 564.05 | 558.52 | 557.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 11:15:00 | 577.45 | 562.81 | 559.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 574.60 | 575.05 | 569.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:30:00 | 574.95 | 575.05 | 569.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 581.10 | 576.26 | 571.88 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 551.20 | 567.89 | 569.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 14:15:00 | 539.00 | 553.75 | 561.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 548.15 | 546.32 | 553.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:15:00 | 546.70 | 546.32 | 553.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 541.00 | 545.49 | 551.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 534.00 | 543.56 | 549.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 539.15 | 535.69 | 536.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 565.00 | 536.29 | 535.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 565.00 | 536.29 | 535.28 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 539.20 | 541.20 | 541.40 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 542.70 | 540.98 | 540.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 548.10 | 542.41 | 541.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 542.95 | 543.09 | 542.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:00:00 | 542.95 | 543.09 | 542.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 548.20 | 549.54 | 546.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 557.30 | 547.60 | 546.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 13:15:00 | 554.25 | 552.71 | 549.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 560.00 | 552.72 | 550.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 541.85 | 551.71 | 550.36 | SL hit (close<static) qty=1.00 sl=544.40 alert=retest2 |

### Cycle 218 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 538.30 | 549.03 | 549.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 532.05 | 543.87 | 546.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 540.00 | 539.36 | 542.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 540.00 | 539.36 | 542.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 540.00 | 539.36 | 542.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 540.00 | 539.36 | 542.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 544.00 | 540.28 | 542.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 537.50 | 539.07 | 541.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 552.20 | 541.52 | 542.42 | SL hit (close>static) qty=1.00 sl=545.45 alert=retest2 |

### Cycle 219 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 547.75 | 543.91 | 543.42 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 538.05 | 542.93 | 543.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 534.20 | 541.18 | 542.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 551.95 | 540.10 | 540.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 551.95 | 540.10 | 540.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 551.95 | 540.10 | 540.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 551.95 | 540.10 | 540.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 531.50 | 538.38 | 540.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 554.85 | 538.38 | 540.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 544.20 | 539.54 | 540.49 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 548.50 | 542.10 | 541.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 563.85 | 548.56 | 544.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 549.00 | 557.56 | 552.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 549.00 | 557.56 | 552.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 549.00 | 557.56 | 552.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 546.25 | 557.56 | 552.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 547.85 | 555.61 | 552.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 553.15 | 555.36 | 552.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-07 09:15:00 | 608.47 | 576.84 | 565.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 838.35 | 856.11 | 858.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 827.65 | 845.27 | 852.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 844.85 | 835.55 | 842.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 844.85 | 835.55 | 842.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 844.85 | 835.55 | 842.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 844.85 | 835.55 | 842.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 861.25 | 840.69 | 844.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 861.00 | 840.69 | 844.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 872.95 | 850.82 | 848.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 896.80 | 868.88 | 858.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 864.25 | 873.24 | 864.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 864.25 | 873.24 | 864.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 864.25 | 873.24 | 864.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 864.25 | 873.24 | 864.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 855.60 | 869.71 | 863.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 855.60 | 869.71 | 863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 850.00 | 865.77 | 862.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 858.40 | 864.00 | 862.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 857.40 | 860.70 | 860.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 857.40 | 860.70 | 860.99 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 865.30 | 861.62 | 861.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 871.00 | 863.49 | 862.26 | Break + close above crossover candle high |

### Cycle 226 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 841.00 | 858.99 | 860.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 840.75 | 855.35 | 858.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 858.15 | 853.79 | 856.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 858.15 | 853.79 | 856.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 858.15 | 853.79 | 856.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 850.60 | 853.79 | 856.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 858.95 | 854.82 | 856.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 849.85 | 854.82 | 856.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 851.75 | 854.21 | 856.31 | EMA400 retest candle locked (from downside) |

### Cycle 227 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 873.40 | 858.05 | 857.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 886.95 | 873.63 | 866.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 877.35 | 877.41 | 870.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 877.35 | 877.41 | 870.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 877.35 | 877.41 | 870.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:30:00 | 872.15 | 877.41 | 870.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 844.95 | 871.33 | 868.91 | EMA400 retest candle locked (from upside) |

### Cycle 228 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 835.25 | 864.12 | 865.85 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 865.50 | 861.04 | 860.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 881.00 | 866.20 | 863.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 870.65 | 871.72 | 867.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 870.65 | 871.72 | 867.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 866.00 | 870.58 | 867.50 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-26 12:30:00 | 53.45 | 2023-05-30 09:15:00 | 52.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-05-26 14:00:00 | 53.65 | 2023-05-30 09:15:00 | 52.95 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-05-29 15:15:00 | 53.50 | 2023-05-30 09:15:00 | 52.95 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-06-02 09:45:00 | 54.05 | 2023-06-09 15:15:00 | 57.05 | STOP_HIT | 1.00 | 5.55% |
| BUY | retest2 | 2023-06-02 14:30:00 | 54.25 | 2023-06-09 15:15:00 | 57.05 | STOP_HIT | 1.00 | 5.16% |
| SELL | retest2 | 2023-06-13 11:15:00 | 57.10 | 2023-06-14 11:15:00 | 57.80 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-06-13 11:45:00 | 57.20 | 2023-06-14 11:15:00 | 57.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-06-14 09:30:00 | 57.45 | 2023-06-14 11:15:00 | 57.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2023-06-16 09:45:00 | 59.35 | 2023-06-19 10:15:00 | 58.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2023-07-25 09:15:00 | 73.05 | 2023-07-26 12:15:00 | 80.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-07 11:45:00 | 85.50 | 2023-08-11 12:15:00 | 85.85 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-08-09 10:00:00 | 85.50 | 2023-08-11 12:15:00 | 85.85 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-08-30 13:30:00 | 86.90 | 2023-08-31 09:15:00 | 89.45 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest1 | 2023-09-04 11:15:00 | 93.60 | 2023-09-04 13:15:00 | 98.28 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-09-04 11:15:00 | 93.60 | 2023-09-05 12:15:00 | 96.85 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2023-09-21 13:00:00 | 93.90 | 2023-09-29 10:15:00 | 89.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-21 13:30:00 | 93.65 | 2023-09-29 10:15:00 | 88.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-22 10:00:00 | 93.65 | 2023-09-29 10:15:00 | 88.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-22 11:00:00 | 93.80 | 2023-09-29 10:15:00 | 89.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-21 13:00:00 | 93.90 | 2023-09-29 11:15:00 | 90.45 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2023-09-21 13:30:00 | 93.65 | 2023-09-29 11:15:00 | 90.45 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2023-09-22 10:00:00 | 93.65 | 2023-09-29 11:15:00 | 90.45 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2023-09-22 11:00:00 | 93.80 | 2023-09-29 11:15:00 | 90.45 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2023-09-27 12:00:00 | 92.15 | 2023-10-04 10:15:00 | 87.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-27 14:00:00 | 91.95 | 2023-10-04 10:15:00 | 87.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-27 12:00:00 | 92.15 | 2023-10-04 13:15:00 | 88.40 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2023-09-27 14:00:00 | 91.95 | 2023-10-04 13:15:00 | 88.40 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2023-10-20 11:45:00 | 88.15 | 2023-10-25 13:15:00 | 83.79 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2023-10-20 13:00:00 | 87.90 | 2023-10-25 13:15:00 | 84.08 | PARTIAL | 0.50 | 4.35% |
| SELL | retest2 | 2023-10-23 09:30:00 | 88.20 | 2023-10-25 14:15:00 | 83.74 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2023-10-23 11:15:00 | 88.50 | 2023-10-25 14:15:00 | 83.50 | PARTIAL | 0.50 | 5.64% |
| SELL | retest2 | 2023-10-20 11:45:00 | 88.15 | 2023-10-26 09:15:00 | 79.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-20 13:00:00 | 87.90 | 2023-10-26 09:15:00 | 79.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-23 09:30:00 | 88.20 | 2023-10-26 09:15:00 | 79.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-23 11:15:00 | 88.50 | 2023-10-26 09:15:00 | 79.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-27 12:30:00 | 84.85 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-27 13:00:00 | 84.50 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-10-27 13:45:00 | 84.95 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-10-27 14:15:00 | 84.90 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-10-30 09:15:00 | 83.55 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2023-10-30 10:45:00 | 84.20 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-10-30 11:30:00 | 84.20 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-10-30 12:30:00 | 83.25 | 2023-10-30 15:15:00 | 85.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2023-11-03 09:15:00 | 88.45 | 2023-11-03 09:15:00 | 97.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-11-23 15:15:00 | 97.25 | 2023-12-01 10:15:00 | 97.40 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-11-24 13:00:00 | 97.30 | 2023-12-01 10:15:00 | 97.40 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2023-11-24 13:30:00 | 97.20 | 2023-12-01 10:15:00 | 97.40 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2023-11-28 11:00:00 | 97.00 | 2023-12-01 10:15:00 | 97.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-11-29 11:30:00 | 96.60 | 2023-12-01 10:15:00 | 97.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-11-29 12:15:00 | 96.20 | 2023-12-01 10:15:00 | 97.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2023-12-05 09:30:00 | 97.65 | 2023-12-06 09:15:00 | 97.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-12-05 10:00:00 | 98.30 | 2023-12-06 09:15:00 | 97.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-12-05 12:30:00 | 97.65 | 2023-12-06 09:15:00 | 97.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-12-06 09:15:00 | 97.70 | 2023-12-06 09:15:00 | 97.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-12-07 15:15:00 | 96.10 | 2023-12-12 10:15:00 | 97.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-12-08 12:00:00 | 96.20 | 2023-12-12 10:15:00 | 97.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-12-08 12:45:00 | 96.30 | 2023-12-12 10:15:00 | 97.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-12-11 12:45:00 | 96.40 | 2023-12-12 10:15:00 | 97.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-12-18 11:30:00 | 113.55 | 2023-12-20 09:15:00 | 124.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-05 13:30:00 | 187.05 | 2024-01-10 13:15:00 | 187.70 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-01-05 15:00:00 | 186.90 | 2024-01-10 13:15:00 | 187.70 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2024-01-08 09:15:00 | 191.40 | 2024-01-10 13:15:00 | 187.70 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-01-10 12:45:00 | 187.70 | 2024-01-10 13:15:00 | 187.70 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-01-16 13:15:00 | 176.00 | 2024-01-17 09:15:00 | 183.45 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2024-01-17 12:45:00 | 177.70 | 2024-01-19 15:15:00 | 180.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-01-17 13:30:00 | 178.00 | 2024-01-19 15:15:00 | 180.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-01-19 09:30:00 | 178.30 | 2024-01-19 15:15:00 | 180.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-02-01 09:15:00 | 206.45 | 2024-02-02 09:15:00 | 227.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-09 10:15:00 | 205.00 | 2024-02-13 09:15:00 | 194.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-09 11:15:00 | 204.00 | 2024-02-13 09:15:00 | 193.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-12 09:30:00 | 204.00 | 2024-02-13 09:15:00 | 193.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-09 10:15:00 | 205.00 | 2024-02-14 09:15:00 | 184.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-02-09 11:15:00 | 204.00 | 2024-02-14 09:15:00 | 200.00 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2024-02-12 09:30:00 | 204.00 | 2024-02-14 09:15:00 | 200.00 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2024-02-27 14:30:00 | 215.00 | 2024-03-01 13:15:00 | 218.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-03-05 11:00:00 | 219.00 | 2024-03-06 11:15:00 | 212.05 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-03-05 11:30:00 | 220.70 | 2024-03-06 11:15:00 | 212.05 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2024-03-18 13:00:00 | 187.20 | 2024-03-19 09:15:00 | 198.80 | STOP_HIT | 1.00 | -6.20% |
| BUY | retest2 | 2024-04-15 10:45:00 | 210.00 | 2024-04-18 09:15:00 | 231.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-13 10:30:00 | 270.00 | 2024-05-14 09:15:00 | 285.90 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest2 | 2024-05-13 11:00:00 | 270.65 | 2024-05-14 09:15:00 | 285.90 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2024-05-13 12:30:00 | 272.35 | 2024-05-14 09:15:00 | 285.90 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2024-05-24 12:30:00 | 268.15 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2024-05-27 10:45:00 | 269.00 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2024-05-27 11:15:00 | 269.00 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2024-05-27 11:45:00 | 269.05 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2024-05-28 14:45:00 | 264.10 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2024-05-30 10:30:00 | 264.50 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2024-05-30 11:30:00 | 263.00 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -5.61% |
| SELL | retest2 | 2024-05-30 12:30:00 | 263.35 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2024-05-31 11:15:00 | 265.10 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2024-05-31 13:00:00 | 264.05 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2024-05-31 15:15:00 | 265.00 | 2024-06-03 09:15:00 | 277.75 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-06-20 14:15:00 | 347.00 | 2024-06-27 13:15:00 | 355.95 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2024-07-22 14:15:00 | 313.05 | 2024-07-24 09:15:00 | 322.55 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-07-22 15:00:00 | 314.00 | 2024-07-24 09:15:00 | 322.55 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-07-23 11:30:00 | 318.70 | 2024-07-24 09:15:00 | 322.55 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-07-23 12:45:00 | 315.10 | 2024-07-24 09:15:00 | 322.55 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-08-05 09:15:00 | 303.00 | 2024-08-06 09:15:00 | 288.80 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2024-08-05 11:00:00 | 304.00 | 2024-08-07 09:15:00 | 288.18 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2024-08-05 09:15:00 | 303.00 | 2024-08-07 11:15:00 | 299.25 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2024-08-05 11:00:00 | 304.00 | 2024-08-07 11:15:00 | 299.25 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2024-08-06 10:30:00 | 300.00 | 2024-08-08 10:15:00 | 311.60 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-08-06 11:00:00 | 303.35 | 2024-08-08 10:15:00 | 311.60 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-08-29 15:00:00 | 375.50 | 2024-09-03 14:15:00 | 373.10 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-09-03 12:00:00 | 379.65 | 2024-09-03 14:15:00 | 373.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-09-03 13:00:00 | 376.00 | 2024-09-03 14:15:00 | 373.10 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-09-18 10:00:00 | 386.00 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-09-18 10:45:00 | 384.95 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-09-18 11:30:00 | 386.00 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-09-19 11:30:00 | 389.65 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-09-19 15:00:00 | 393.40 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-09-20 13:30:00 | 393.00 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-09-23 09:15:00 | 393.85 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-09-23 09:45:00 | 391.85 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-09-23 13:15:00 | 392.00 | 2024-09-24 09:15:00 | 384.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-09-26 11:15:00 | 376.10 | 2024-09-30 09:15:00 | 357.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 12:15:00 | 377.15 | 2024-09-30 09:15:00 | 358.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 13:00:00 | 377.25 | 2024-09-30 09:15:00 | 358.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 14:15:00 | 376.80 | 2024-09-30 09:15:00 | 357.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 11:15:00 | 376.10 | 2024-10-01 11:15:00 | 362.00 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2024-09-26 12:15:00 | 377.15 | 2024-10-01 11:15:00 | 362.00 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2024-09-26 13:00:00 | 377.25 | 2024-10-01 11:15:00 | 362.00 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2024-09-26 14:15:00 | 376.80 | 2024-10-01 11:15:00 | 362.00 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-09-27 15:15:00 | 363.50 | 2024-10-07 12:15:00 | 345.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 15:15:00 | 363.50 | 2024-10-08 09:15:00 | 349.00 | STOP_HIT | 0.50 | 3.99% |
| BUY | retest2 | 2024-10-11 12:30:00 | 369.85 | 2024-10-11 14:15:00 | 356.10 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2024-10-24 12:15:00 | 338.50 | 2024-10-28 09:15:00 | 321.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 15:00:00 | 337.00 | 2024-10-28 09:15:00 | 320.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 337.00 | 2024-10-28 09:15:00 | 320.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:15:00 | 338.50 | 2024-10-29 12:15:00 | 325.00 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2024-10-24 15:00:00 | 337.00 | 2024-10-29 12:15:00 | 325.00 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2024-10-25 09:15:00 | 337.00 | 2024-10-29 12:15:00 | 325.00 | STOP_HIT | 0.50 | 3.56% |
| SELL | retest2 | 2024-11-13 09:45:00 | 300.90 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -7.24% |
| SELL | retest2 | 2024-11-13 10:15:00 | 304.30 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest2 | 2024-11-13 11:45:00 | 304.50 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -5.98% |
| SELL | retest2 | 2024-11-13 14:00:00 | 301.50 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -7.03% |
| SELL | retest2 | 2024-11-14 11:30:00 | 300.00 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -7.57% |
| SELL | retest2 | 2024-11-14 12:45:00 | 300.00 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -7.57% |
| SELL | retest2 | 2024-11-14 13:45:00 | 300.65 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -7.33% |
| SELL | retest2 | 2024-11-18 09:15:00 | 301.00 | 2024-11-19 09:15:00 | 322.70 | STOP_HIT | 1.00 | -7.21% |
| BUY | retest2 | 2024-11-26 11:15:00 | 321.05 | 2024-12-02 09:15:00 | 353.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-26 11:45:00 | 319.05 | 2024-12-02 09:15:00 | 350.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-26 15:15:00 | 322.00 | 2024-12-02 09:15:00 | 354.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 09:30:00 | 320.65 | 2024-12-02 09:15:00 | 352.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 11:30:00 | 341.85 | 2024-12-03 09:15:00 | 376.04 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-10 10:15:00 | 359.00 | 2024-12-16 09:15:00 | 341.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 15:15:00 | 360.00 | 2024-12-16 09:15:00 | 342.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 11:30:00 | 360.00 | 2024-12-16 09:15:00 | 342.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 14:45:00 | 360.00 | 2024-12-16 09:15:00 | 342.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 13:45:00 | 355.15 | 2024-12-16 09:15:00 | 337.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 14:30:00 | 355.10 | 2024-12-16 09:15:00 | 337.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 354.00 | 2024-12-16 09:15:00 | 336.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 13:15:00 | 356.50 | 2024-12-16 09:15:00 | 338.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 10:15:00 | 359.00 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2024-12-10 15:15:00 | 360.00 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2024-12-11 11:30:00 | 360.00 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2024-12-11 14:45:00 | 360.00 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2024-12-12 13:45:00 | 355.15 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 0.75% |
| SELL | retest2 | 2024-12-12 14:30:00 | 355.10 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 0.73% |
| SELL | retest2 | 2024-12-13 09:15:00 | 354.00 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2024-12-13 13:15:00 | 356.50 | 2024-12-17 12:15:00 | 352.50 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2024-12-20 14:45:00 | 341.25 | 2024-12-23 13:15:00 | 352.30 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-01-09 13:45:00 | 338.10 | 2025-01-09 15:15:00 | 345.85 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-01-10 09:15:00 | 332.00 | 2025-01-14 09:15:00 | 315.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 332.00 | 2025-01-14 11:15:00 | 320.45 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-01-23 13:30:00 | 313.75 | 2025-01-28 09:15:00 | 298.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:15:00 | 315.10 | 2025-01-28 09:15:00 | 299.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:30:00 | 313.75 | 2025-01-29 09:15:00 | 303.55 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-01-24 13:15:00 | 315.10 | 2025-01-29 09:15:00 | 303.55 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-02-13 11:00:00 | 313.25 | 2025-02-17 15:15:00 | 320.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-02-13 11:30:00 | 312.60 | 2025-02-19 09:15:00 | 297.59 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-02-13 11:30:00 | 312.60 | 2025-02-19 09:15:00 | 314.55 | STOP_HIT | 0.50 | -0.62% |
| SELL | retest2 | 2025-02-13 12:00:00 | 311.90 | 2025-02-19 09:15:00 | 296.97 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-02-13 12:00:00 | 311.90 | 2025-02-19 09:15:00 | 314.55 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2025-02-14 09:30:00 | 313.05 | 2025-02-19 09:15:00 | 296.30 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-02-14 09:30:00 | 313.05 | 2025-02-19 09:15:00 | 314.55 | STOP_HIT | 0.50 | -0.48% |
| SELL | retest2 | 2025-02-17 10:00:00 | 303.30 | 2025-02-19 09:15:00 | 297.40 | PARTIAL | 0.50 | 1.95% |
| SELL | retest2 | 2025-02-17 10:00:00 | 303.30 | 2025-02-19 09:15:00 | 314.55 | STOP_HIT | 0.50 | -3.71% |
| BUY | retest2 | 2025-02-21 13:45:00 | 334.90 | 2025-02-27 09:15:00 | 318.50 | STOP_HIT | 1.00 | -4.90% |
| SELL | retest2 | 2025-03-17 12:15:00 | 314.00 | 2025-03-19 09:15:00 | 323.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-03-17 14:30:00 | 314.05 | 2025-03-19 09:15:00 | 323.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-03-18 12:30:00 | 313.45 | 2025-03-19 09:15:00 | 323.00 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-03-18 14:30:00 | 312.95 | 2025-03-19 09:15:00 | 323.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest1 | 2025-03-27 09:15:00 | 377.50 | 2025-03-27 14:15:00 | 353.00 | STOP_HIT | 1.00 | -6.49% |
| BUY | retest2 | 2025-03-28 09:15:00 | 366.60 | 2025-04-01 09:15:00 | 403.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 15:15:00 | 400.00 | 2025-04-11 09:15:00 | 421.90 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2025-04-28 15:15:00 | 453.00 | 2025-05-05 10:15:00 | 456.60 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-04-29 09:30:00 | 447.25 | 2025-05-05 10:15:00 | 456.60 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-05-08 13:45:00 | 432.30 | 2025-05-09 09:15:00 | 410.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:45:00 | 431.35 | 2025-05-09 09:15:00 | 409.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:45:00 | 432.30 | 2025-05-12 09:15:00 | 450.00 | STOP_HIT | 0.50 | -4.09% |
| SELL | retest2 | 2025-05-08 14:45:00 | 431.35 | 2025-05-12 09:15:00 | 450.00 | STOP_HIT | 0.50 | -4.32% |
| BUY | retest2 | 2025-05-15 15:15:00 | 466.90 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-05-16 11:15:00 | 467.70 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-05-19 10:00:00 | 470.00 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-05-21 12:45:00 | 467.40 | 2025-05-21 13:15:00 | 470.05 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-06-16 14:15:00 | 454.60 | 2025-06-18 15:15:00 | 459.40 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-17 10:15:00 | 453.35 | 2025-06-18 15:15:00 | 459.40 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-19 09:15:00 | 454.75 | 2025-06-19 11:15:00 | 460.45 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-19 09:45:00 | 454.25 | 2025-06-19 11:15:00 | 460.45 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-27 09:15:00 | 541.40 | 2025-06-27 10:15:00 | 533.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-27 10:00:00 | 539.05 | 2025-06-27 10:15:00 | 533.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-02 13:15:00 | 521.60 | 2025-07-03 09:15:00 | 550.00 | STOP_HIT | 1.00 | -5.44% |
| SELL | retest2 | 2025-07-02 14:15:00 | 523.00 | 2025-07-03 09:15:00 | 550.00 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2025-07-04 15:15:00 | 549.85 | 2025-07-07 15:15:00 | 540.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-15 11:30:00 | 586.00 | 2025-07-21 10:15:00 | 641.69 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2025-07-15 13:00:00 | 583.35 | 2025-07-21 10:15:00 | 641.30 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-07-15 13:45:00 | 583.00 | 2025-07-21 10:15:00 | 642.95 | TARGET_HIT | 1.00 | 10.28% |
| BUY | retest2 | 2025-07-15 15:00:00 | 584.50 | 2025-07-21 11:15:00 | 644.60 | TARGET_HIT | 1.00 | 10.28% |
| BUY | retest2 | 2025-07-17 14:45:00 | 592.00 | 2025-07-21 14:15:00 | 651.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-18 09:45:00 | 591.35 | 2025-07-21 14:15:00 | 650.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 09:15:00 | 593.85 | 2025-07-24 10:15:00 | 653.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 632.40 | 2025-08-26 15:15:00 | 600.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 632.40 | 2025-09-01 11:15:00 | 585.80 | STOP_HIT | 0.50 | 7.37% |
| SELL | retest2 | 2025-09-19 14:15:00 | 660.60 | 2025-09-22 09:15:00 | 685.00 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-09-19 15:00:00 | 660.70 | 2025-09-22 09:15:00 | 685.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-09-30 09:15:00 | 656.50 | 2025-10-01 12:15:00 | 664.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-01 09:30:00 | 658.50 | 2025-10-01 12:15:00 | 664.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-01 10:30:00 | 656.60 | 2025-10-01 12:15:00 | 664.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-14 10:30:00 | 652.95 | 2025-10-15 14:15:00 | 661.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-15 11:30:00 | 652.45 | 2025-10-15 14:15:00 | 661.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-29 09:15:00 | 532.95 | 2025-10-29 09:15:00 | 539.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-29 11:00:00 | 532.80 | 2025-11-03 09:15:00 | 539.85 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-10-29 11:30:00 | 532.00 | 2025-11-03 09:15:00 | 539.85 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-30 09:45:00 | 532.25 | 2025-11-03 09:15:00 | 539.85 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-01 14:30:00 | 590.10 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-01 15:15:00 | 590.00 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-02 09:30:00 | 590.55 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-02 10:15:00 | 590.40 | 2025-12-02 15:15:00 | 593.75 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-05 09:30:00 | 589.05 | 2025-12-08 09:15:00 | 597.45 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-05 10:15:00 | 588.55 | 2025-12-08 09:15:00 | 597.45 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-05 12:30:00 | 589.10 | 2025-12-08 09:15:00 | 597.45 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-12-22 15:15:00 | 516.85 | 2025-12-24 12:15:00 | 526.60 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-30 11:30:00 | 529.50 | 2026-01-12 12:15:00 | 560.05 | STOP_HIT | 1.00 | 5.77% |
| BUY | retest2 | 2025-12-31 09:15:00 | 535.35 | 2026-01-12 12:15:00 | 560.05 | STOP_HIT | 1.00 | 4.61% |
| SELL | retest2 | 2026-01-16 15:15:00 | 550.20 | 2026-01-19 11:15:00 | 562.30 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-01-27 09:15:00 | 527.55 | 2026-02-02 10:15:00 | 501.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 09:15:00 | 527.55 | 2026-02-02 14:15:00 | 516.80 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2026-03-06 10:45:00 | 534.00 | 2026-03-12 10:15:00 | 565.00 | STOP_HIT | 1.00 | -5.81% |
| SELL | retest2 | 2026-03-11 10:15:00 | 539.15 | 2026-03-12 10:15:00 | 565.00 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2026-03-20 09:15:00 | 557.30 | 2026-03-23 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-03-20 13:15:00 | 554.25 | 2026-03-23 09:15:00 | 541.85 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-03-20 15:15:00 | 560.00 | 2026-03-23 09:15:00 | 541.85 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-03-24 14:30:00 | 537.50 | 2026-03-25 09:15:00 | 552.20 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-04-02 11:30:00 | 553.15 | 2026-04-07 09:15:00 | 608.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 10:15:00 | 858.40 | 2026-04-29 13:15:00 | 857.40 | STOP_HIT | 1.00 | -0.12% |
