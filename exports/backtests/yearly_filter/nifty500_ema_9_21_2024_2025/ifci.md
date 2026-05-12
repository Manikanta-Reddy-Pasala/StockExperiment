# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 64.27
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 93 |
| ALERT2 | 89 |
| ALERT2_SKIP | 46 |
| ALERT3 | 215 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 71 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 52
- **Target hits / Stop hits / Partials:** 13 / 63 / 16
- **Avg / median % per leg:** 1.30% / -0.63%
- **Sum % (uncompounded):** 119.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 4 | 21.1% | 3 | 16 | 0 | -0.61% | -11.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 4 | 21.1% | 3 | 16 | 0 | -0.61% | -11.5% |
| SELL (all) | 73 | 36 | 49.3% | 10 | 47 | 16 | 1.79% | 130.7% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.06% | -15.3% |
| SELL @ 3rd Alert (retest2) | 68 | 36 | 52.9% | 10 | 42 | 16 | 2.15% | 146.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.06% | -15.3% |
| retest2 (combined) | 87 | 40 | 46.0% | 13 | 58 | 16 | 1.55% | 134.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 50.20 | 51.54 | 51.59 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 53.70 | 51.60 | 51.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 54.30 | 53.40 | 52.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 10:15:00 | 61.75 | 61.77 | 59.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 10:30:00 | 61.25 | 61.77 | 59.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 59.80 | 61.12 | 60.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 59.55 | 61.12 | 60.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 59.25 | 60.74 | 60.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 59.25 | 60.74 | 60.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 59.15 | 60.42 | 60.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:15:00 | 59.15 | 60.42 | 60.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 59.40 | 59.70 | 59.74 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 60.55 | 59.87 | 59.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 10:15:00 | 61.35 | 60.16 | 59.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 60.50 | 60.58 | 60.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:00:00 | 60.50 | 60.58 | 60.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 60.35 | 60.55 | 60.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 60.35 | 60.55 | 60.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 60.00 | 60.44 | 60.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:45:00 | 60.05 | 60.44 | 60.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 60.05 | 60.36 | 60.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 59.60 | 60.36 | 60.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 58.20 | 59.93 | 60.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 57.30 | 58.23 | 58.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 58.05 | 57.77 | 58.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 58.05 | 57.77 | 58.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 58.05 | 57.77 | 58.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 58.35 | 57.77 | 58.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 57.80 | 57.78 | 58.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 57.60 | 57.78 | 58.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 13:15:00 | 59.45 | 58.23 | 58.46 | SL hit (close>static) qty=1.00 sl=58.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 59.00 | 58.67 | 58.63 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 57.80 | 58.59 | 58.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 56.35 | 57.98 | 58.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 59.10 | 57.92 | 58.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 59.10 | 57.92 | 58.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 59.10 | 57.92 | 58.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 59.10 | 57.92 | 58.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 57.95 | 57.93 | 58.13 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 62.15 | 58.83 | 58.51 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 54.85 | 58.96 | 59.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 49.55 | 54.88 | 56.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 53.45 | 53.39 | 55.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:30:00 | 53.70 | 53.39 | 55.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 55.95 | 53.73 | 54.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 55.30 | 53.73 | 54.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 57.15 | 54.41 | 55.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 57.45 | 54.41 | 55.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 56.45 | 55.65 | 55.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 57.10 | 55.94 | 55.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 11:15:00 | 64.62 | 64.66 | 62.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:00:00 | 64.62 | 64.66 | 62.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 64.22 | 64.32 | 63.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 14:45:00 | 65.43 | 64.42 | 64.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 65.34 | 64.43 | 64.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 62.92 | 64.47 | 64.41 | SL hit (close<static) qty=1.00 sl=63.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 62.93 | 64.16 | 64.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 15:15:00 | 62.35 | 62.99 | 63.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 62.27 | 62.22 | 62.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 14:00:00 | 62.27 | 62.22 | 62.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 63.20 | 61.67 | 61.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 63.20 | 61.67 | 61.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 63.29 | 62.00 | 62.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:30:00 | 62.65 | 62.18 | 62.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 62.60 | 62.26 | 62.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 62.60 | 62.26 | 62.22 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 62.09 | 62.20 | 62.20 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 62.79 | 62.32 | 62.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 63.17 | 62.49 | 62.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 15:15:00 | 62.79 | 62.90 | 62.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 15:15:00 | 62.79 | 62.90 | 62.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 62.79 | 62.90 | 62.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 62.57 | 62.83 | 62.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 61.67 | 62.60 | 62.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 61.70 | 62.60 | 62.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 61.64 | 62.41 | 62.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 12:15:00 | 60.74 | 62.07 | 62.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 61.75 | 61.24 | 61.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 61.75 | 61.24 | 61.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 61.75 | 61.24 | 61.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 61.33 | 61.24 | 61.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 61.56 | 61.31 | 61.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 60.95 | 61.31 | 61.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 61.91 | 61.45 | 61.69 | SL hit (close>static) qty=1.00 sl=61.80 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 61.67 | 61.05 | 60.97 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 60.82 | 61.00 | 61.00 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 66.60 | 62.10 | 61.50 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 61.71 | 62.78 | 62.85 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 66.26 | 62.97 | 62.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 69.16 | 64.21 | 63.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 73.96 | 74.35 | 71.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 73.96 | 74.35 | 71.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 73.11 | 73.94 | 72.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 71.00 | 73.94 | 72.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 70.74 | 73.30 | 72.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 70.74 | 73.30 | 72.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 71.39 | 72.92 | 72.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 71.63 | 72.66 | 72.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:30:00 | 71.55 | 72.26 | 72.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 71.65 | 72.08 | 72.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 69.30 | 71.53 | 71.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 69.30 | 71.53 | 71.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 69.24 | 70.49 | 71.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 72.94 | 70.25 | 70.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 72.94 | 70.25 | 70.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 72.94 | 70.25 | 70.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 72.59 | 70.25 | 70.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 71.28 | 70.45 | 70.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 72.64 | 70.45 | 70.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 13:15:00 | 71.50 | 71.07 | 71.04 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 69.93 | 70.98 | 71.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 11:15:00 | 69.45 | 70.65 | 70.86 | Break + close below crossover candle low |

### Cycle 24 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 75.96 | 71.12 | 70.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 80.49 | 74.06 | 72.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 83.85 | 84.16 | 80.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 10:00:00 | 83.85 | 84.16 | 80.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 83.10 | 83.60 | 81.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:30:00 | 81.82 | 83.60 | 81.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 83.70 | 83.60 | 82.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:00:00 | 86.79 | 84.30 | 83.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 89.15 | 84.97 | 84.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:00:00 | 85.30 | 86.20 | 85.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 83.68 | 85.03 | 85.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 83.68 | 85.03 | 85.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 82.42 | 84.15 | 84.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 75.90 | 75.38 | 77.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 76.01 | 75.38 | 77.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 77.32 | 76.11 | 77.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:45:00 | 77.40 | 76.11 | 77.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 77.70 | 76.43 | 77.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 76.86 | 76.43 | 77.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:45:00 | 77.23 | 76.59 | 77.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 77.19 | 76.89 | 77.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 15:15:00 | 73.37 | 75.95 | 76.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 15:15:00 | 73.33 | 75.95 | 76.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:15:00 | 73.02 | 75.25 | 76.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-12 09:15:00 | 69.17 | 71.63 | 73.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 72.37 | 71.90 | 71.84 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 71.79 | 72.01 | 72.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 15:15:00 | 71.50 | 71.78 | 71.88 | Break + close below crossover candle low |

### Cycle 28 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 75.97 | 72.07 | 71.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 12:15:00 | 77.38 | 74.22 | 72.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 09:15:00 | 75.62 | 76.91 | 75.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 75.62 | 76.91 | 75.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 75.62 | 76.91 | 75.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 75.62 | 76.91 | 75.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 75.30 | 76.58 | 75.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 75.30 | 76.58 | 75.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 74.89 | 76.25 | 75.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 75.00 | 76.25 | 75.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 73.25 | 75.05 | 75.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 10:15:00 | 73.06 | 74.65 | 75.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 74.30 | 73.69 | 74.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 74.30 | 73.69 | 74.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 74.30 | 73.69 | 74.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 75.28 | 73.69 | 74.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 73.92 | 73.73 | 74.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:45:00 | 74.20 | 73.73 | 74.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 73.94 | 73.78 | 74.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:30:00 | 74.15 | 73.78 | 74.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 73.95 | 73.78 | 74.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 73.95 | 73.78 | 74.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 73.24 | 73.64 | 73.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 71.61 | 72.99 | 73.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 70.49 | 70.12 | 70.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 70.49 | 70.12 | 70.09 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 69.85 | 70.06 | 70.07 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 14:15:00 | 70.41 | 70.13 | 70.10 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 69.29 | 69.99 | 70.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 11:15:00 | 69.17 | 69.73 | 69.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 69.84 | 69.62 | 69.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 69.84 | 69.62 | 69.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 69.84 | 69.62 | 69.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:45:00 | 69.83 | 69.62 | 69.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 69.80 | 69.65 | 69.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 70.04 | 69.65 | 69.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 70.54 | 69.83 | 69.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 70.66 | 69.83 | 69.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 70.30 | 69.92 | 69.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 71.55 | 70.49 | 70.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 72.41 | 72.70 | 71.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 72.41 | 72.70 | 71.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 71.70 | 72.33 | 71.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 71.38 | 72.33 | 71.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 72.25 | 72.32 | 71.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 71.45 | 72.32 | 71.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 72.03 | 72.26 | 71.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:30:00 | 72.05 | 72.26 | 71.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 71.65 | 72.14 | 71.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 71.65 | 72.14 | 71.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 71.50 | 72.01 | 71.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 71.55 | 72.01 | 71.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 70.98 | 71.71 | 71.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 69.43 | 71.25 | 71.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 70.30 | 70.06 | 70.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:15:00 | 70.60 | 70.06 | 70.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 72.09 | 70.46 | 70.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 72.09 | 70.46 | 70.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 72.14 | 70.80 | 70.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:15:00 | 71.61 | 70.80 | 70.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:00:00 | 71.56 | 70.95 | 70.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 71.17 | 71.03 | 71.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 71.17 | 71.03 | 71.02 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 70.65 | 70.97 | 71.00 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 71.24 | 71.02 | 71.02 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 11:15:00 | 70.95 | 71.01 | 71.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 14:15:00 | 70.53 | 70.89 | 70.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 66.72 | 66.58 | 67.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 10:45:00 | 66.67 | 66.58 | 67.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 67.44 | 66.75 | 67.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 67.44 | 66.75 | 67.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 67.18 | 66.84 | 67.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 65.76 | 66.84 | 67.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:15:00 | 62.47 | 64.15 | 65.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-04 09:15:00 | 59.18 | 62.61 | 63.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 40 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 63.42 | 62.46 | 62.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 64.40 | 62.84 | 62.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 63.26 | 63.49 | 63.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 63.26 | 63.49 | 63.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 63.26 | 63.49 | 63.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 63.29 | 63.49 | 63.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 62.76 | 63.35 | 63.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:30:00 | 62.86 | 63.35 | 63.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 63.05 | 63.29 | 63.10 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 62.18 | 62.96 | 62.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 61.38 | 62.65 | 62.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 61.18 | 59.85 | 60.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 61.18 | 59.85 | 60.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 61.18 | 59.85 | 60.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 61.16 | 59.85 | 60.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 61.10 | 60.10 | 60.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 61.10 | 60.10 | 60.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 12:15:00 | 61.33 | 60.50 | 60.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 15:15:00 | 61.82 | 61.04 | 60.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 60.82 | 61.00 | 60.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 60.82 | 61.00 | 60.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 60.82 | 61.00 | 60.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 60.82 | 61.00 | 60.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 60.35 | 60.87 | 60.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 60.35 | 60.87 | 60.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 60.40 | 60.77 | 60.70 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 60.22 | 60.59 | 60.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 58.57 | 60.12 | 60.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 54.66 | 54.61 | 55.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:30:00 | 55.33 | 54.61 | 55.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 54.92 | 54.53 | 55.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:00:00 | 54.92 | 54.53 | 55.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 52.48 | 51.92 | 52.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 52.08 | 52.40 | 52.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:15:00 | 52.13 | 52.40 | 52.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:30:00 | 51.94 | 52.43 | 52.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 53.18 | 52.58 | 52.69 | SL hit (close>static) qty=1.00 sl=52.96 alert=retest2 |

### Cycle 44 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 54.64 | 53.12 | 52.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 55.05 | 53.51 | 53.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 56.78 | 57.54 | 56.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 56.78 | 57.54 | 56.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 56.52 | 57.08 | 56.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 56.45 | 57.08 | 56.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 56.26 | 56.91 | 56.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 56.26 | 56.91 | 56.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 56.23 | 56.78 | 56.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:45:00 | 56.09 | 56.78 | 56.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 62.68 | 64.90 | 63.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 62.68 | 64.90 | 63.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 62.94 | 64.51 | 63.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:15:00 | 62.80 | 64.51 | 63.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 62.31 | 64.07 | 63.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 62.31 | 64.07 | 63.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 66.40 | 63.89 | 63.23 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 60.67 | 62.99 | 63.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 60.03 | 62.01 | 62.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 61.02 | 60.80 | 61.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 13:30:00 | 58.86 | 60.13 | 61.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 10:15:00 | 58.48 | 59.37 | 60.48 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 60.69 | 59.57 | 60.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 60.69 | 59.57 | 60.20 | SL hit (close>ema400) qty=1.00 sl=60.20 alert=retest1 |

### Cycle 46 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 62.14 | 60.80 | 60.63 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 59.50 | 60.59 | 60.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 11:15:00 | 58.00 | 59.18 | 59.78 | Break + close below crossover candle low |

### Cycle 48 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 64.69 | 59.71 | 59.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 13:15:00 | 67.76 | 66.18 | 65.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 11:15:00 | 66.54 | 66.66 | 66.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 11:45:00 | 66.48 | 66.66 | 66.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 66.49 | 66.55 | 66.22 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 65.47 | 66.06 | 66.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 64.88 | 65.36 | 65.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 63.15 | 62.86 | 63.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 63.15 | 62.86 | 63.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 63.15 | 62.86 | 63.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 63.59 | 62.86 | 63.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 64.05 | 63.10 | 63.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 64.05 | 63.10 | 63.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 63.83 | 63.24 | 63.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 64.15 | 63.24 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 64.36 | 63.88 | 63.85 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 63.53 | 63.85 | 63.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 62.88 | 63.50 | 63.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 63.30 | 61.79 | 62.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 63.30 | 61.79 | 62.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 63.30 | 61.79 | 62.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 63.30 | 61.79 | 62.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 63.11 | 62.06 | 62.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 63.16 | 62.06 | 62.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 13:15:00 | 63.79 | 62.73 | 62.68 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 61.22 | 62.57 | 62.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 11:15:00 | 60.63 | 61.47 | 62.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 61.29 | 61.23 | 61.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 15:00:00 | 61.29 | 61.23 | 61.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 61.38 | 61.26 | 61.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:00:00 | 61.15 | 61.27 | 61.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 61.19 | 61.22 | 61.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:00:00 | 61.06 | 61.22 | 61.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 60.79 | 60.46 | 60.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 60.85 | 60.54 | 60.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 60.86 | 60.54 | 60.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 61.14 | 60.66 | 60.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 61.08 | 60.66 | 60.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 60.68 | 60.66 | 60.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 60.25 | 60.66 | 60.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:15:00 | 60.08 | 60.62 | 60.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 14:15:00 | 62.48 | 60.45 | 60.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 62.48 | 60.45 | 60.31 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 61.23 | 61.70 | 61.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 59.54 | 61.27 | 61.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 59.93 | 59.38 | 60.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 59.93 | 59.38 | 60.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 59.93 | 59.38 | 60.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 60.38 | 59.38 | 60.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 59.08 | 58.77 | 59.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 59.08 | 58.77 | 59.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 58.70 | 58.75 | 59.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 58.59 | 58.75 | 59.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 58.45 | 58.74 | 59.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 55.66 | 57.44 | 58.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 55.53 | 57.44 | 58.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 11:15:00 | 52.73 | 54.43 | 55.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 56 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 56.00 | 54.54 | 54.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 14:15:00 | 56.24 | 55.41 | 54.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 56.80 | 57.31 | 56.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 56.80 | 57.31 | 56.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 57.00 | 57.35 | 56.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 56.02 | 57.35 | 56.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 56.11 | 57.10 | 56.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 55.59 | 57.10 | 56.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 55.38 | 56.76 | 56.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 55.38 | 56.76 | 56.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 54.57 | 56.32 | 56.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 54.44 | 55.14 | 55.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 52.46 | 51.97 | 53.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 52.46 | 51.97 | 53.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 53.11 | 52.20 | 53.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 53.11 | 52.20 | 53.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 52.98 | 52.36 | 53.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 52.98 | 52.36 | 53.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 52.23 | 52.33 | 52.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:45:00 | 52.40 | 52.33 | 52.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 54.13 | 52.64 | 53.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 54.13 | 52.64 | 53.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 54.12 | 52.93 | 53.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 54.13 | 52.93 | 53.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 53.55 | 53.26 | 53.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 54.20 | 53.45 | 53.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 54.41 | 54.44 | 53.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 54.41 | 54.44 | 53.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 54.29 | 54.39 | 54.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 53.96 | 54.39 | 54.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 54.86 | 54.49 | 54.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 55.35 | 54.49 | 54.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 15:15:00 | 54.18 | 54.87 | 54.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 54.18 | 54.87 | 54.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 52.83 | 54.46 | 54.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 52.76 | 52.68 | 53.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:15:00 | 52.45 | 52.68 | 53.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 14:15:00 | 52.55 | 52.49 | 53.12 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 52.92 | 52.57 | 53.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 52.92 | 52.57 | 53.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 54.38 | 52.99 | 53.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 54.38 | 52.99 | 53.21 | SL hit (close>ema400) qty=1.00 sl=53.21 alert=retest1 |

### Cycle 60 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 54.05 | 53.41 | 53.37 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 53.28 | 53.48 | 53.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 53.05 | 53.39 | 53.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 53.57 | 53.43 | 53.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 14:15:00 | 53.57 | 53.43 | 53.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 53.57 | 53.43 | 53.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 53.57 | 53.43 | 53.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 53.80 | 53.50 | 53.48 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 10:15:00 | 53.45 | 53.47 | 53.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 52.61 | 53.19 | 53.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 49.30 | 48.67 | 49.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 49.30 | 48.67 | 49.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 49.26 | 48.97 | 49.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 49.61 | 48.97 | 49.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 45.75 | 43.75 | 44.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 45.75 | 43.75 | 44.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 45.60 | 44.12 | 44.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 45.84 | 44.12 | 44.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 45.81 | 44.92 | 44.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 46.02 | 45.22 | 45.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 46.00 | 46.37 | 45.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 46.00 | 46.37 | 45.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 46.00 | 46.37 | 45.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 46.00 | 46.37 | 45.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 46.05 | 46.31 | 45.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:30:00 | 46.66 | 46.45 | 46.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:30:00 | 46.68 | 46.57 | 46.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 45.34 | 46.27 | 46.08 | SL hit (close<static) qty=1.00 sl=45.80 alert=retest2 |

### Cycle 65 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 45.62 | 45.93 | 45.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 45.28 | 45.72 | 45.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 46.00 | 45.78 | 45.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 46.00 | 45.78 | 45.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 46.00 | 45.78 | 45.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 45.12 | 45.63 | 45.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 12:30:00 | 45.11 | 45.54 | 45.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 13:15:00 | 45.03 | 45.54 | 45.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:45:00 | 45.05 | 45.33 | 45.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 42.86 | 44.22 | 44.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 42.85 | 44.22 | 44.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 42.78 | 44.22 | 44.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 42.80 | 44.22 | 44.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 12:15:00 | 40.61 | 42.11 | 43.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 66 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 42.63 | 41.65 | 41.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 42.93 | 42.18 | 41.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 43.55 | 44.11 | 43.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 43.55 | 44.11 | 43.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 43.55 | 44.11 | 43.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 43.55 | 44.11 | 43.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 43.31 | 43.95 | 43.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 43.31 | 43.95 | 43.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 42.95 | 43.75 | 43.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 42.95 | 43.75 | 43.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 42.83 | 43.44 | 43.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 42.53 | 43.25 | 43.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 42.20 | 42.13 | 42.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 42.11 | 42.13 | 42.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 41.49 | 42.00 | 42.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 41.25 | 41.88 | 42.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 41.29 | 41.43 | 41.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 41.24 | 41.43 | 41.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 41.61 | 41.22 | 41.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 41.61 | 41.22 | 41.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 42.84 | 41.54 | 41.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 44.93 | 45.62 | 45.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 44.93 | 45.62 | 45.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 44.93 | 45.62 | 45.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 44.93 | 45.62 | 45.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 44.59 | 45.41 | 44.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 44.71 | 45.41 | 44.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 44.42 | 45.21 | 44.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 44.43 | 45.21 | 44.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 44.21 | 44.76 | 44.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 44.09 | 44.62 | 44.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 43.80 | 43.52 | 43.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 43.80 | 43.52 | 43.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 43.80 | 43.52 | 43.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 43.76 | 43.52 | 43.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 43.44 | 43.51 | 43.90 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 44.61 | 44.03 | 44.02 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 43.59 | 43.99 | 44.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 43.39 | 43.87 | 43.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 43.89 | 43.65 | 43.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 43.89 | 43.65 | 43.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 43.89 | 43.65 | 43.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 44.02 | 43.65 | 43.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 43.61 | 43.64 | 43.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 43.40 | 43.62 | 43.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 12:15:00 | 44.22 | 43.74 | 43.81 | SL hit (close>static) qty=1.00 sl=44.10 alert=retest2 |

### Cycle 72 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 44.55 | 43.90 | 43.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 45.00 | 44.49 | 44.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 43.17 | 44.51 | 44.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 43.17 | 44.51 | 44.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 43.17 | 44.51 | 44.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 43.17 | 44.51 | 44.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 43.45 | 44.30 | 44.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 42.81 | 43.67 | 43.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 41.06 | 40.87 | 41.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 41.06 | 40.87 | 41.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 41.06 | 40.87 | 41.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 40.75 | 40.90 | 41.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 40.44 | 41.24 | 41.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 42.25 | 41.06 | 40.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 42.25 | 41.06 | 40.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 43.01 | 41.90 | 41.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 43.72 | 43.74 | 43.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 09:15:00 | 43.66 | 43.74 | 43.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 44.76 | 45.37 | 44.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 44.76 | 45.37 | 44.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 44.98 | 45.29 | 44.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 44.79 | 45.29 | 44.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 45.18 | 45.27 | 44.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:00:00 | 45.34 | 45.28 | 44.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 44.09 | 45.57 | 45.50 | SL hit (close<static) qty=1.00 sl=44.92 alert=retest2 |

### Cycle 75 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 43.49 | 45.16 | 45.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 43.21 | 43.66 | 43.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 42.65 | 42.42 | 42.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:00:00 | 42.65 | 42.42 | 42.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 42.89 | 42.63 | 42.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 42.36 | 42.63 | 42.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 41.93 | 42.49 | 42.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 41.74 | 42.34 | 42.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 39.65 | 41.07 | 41.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 40.85 | 40.75 | 41.28 | SL hit (close>ema200) qty=0.50 sl=40.75 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 42.91 | 40.62 | 40.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 43.35 | 42.05 | 41.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 50.30 | 51.09 | 49.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:30:00 | 50.10 | 51.09 | 49.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 49.60 | 50.29 | 49.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 49.35 | 50.29 | 49.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 49.88 | 50.21 | 49.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 49.23 | 50.21 | 49.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 50.19 | 50.20 | 49.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 53.51 | 50.68 | 50.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-26 11:15:00 | 58.86 | 55.20 | 53.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 66.01 | 67.02 | 67.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 65.68 | 66.62 | 66.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 66.25 | 66.23 | 66.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 66.25 | 66.23 | 66.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 66.25 | 66.23 | 66.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 66.25 | 66.23 | 66.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 67.58 | 66.50 | 66.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:15:00 | 68.23 | 66.50 | 66.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 67.24 | 66.65 | 66.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 67.81 | 66.65 | 66.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 67.55 | 66.83 | 66.81 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 66.01 | 66.86 | 66.97 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 67.63 | 67.08 | 67.04 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 66.86 | 67.10 | 67.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 66.44 | 66.91 | 67.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 61.42 | 61.06 | 62.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 61.42 | 61.06 | 62.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 62.44 | 61.43 | 62.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 62.79 | 61.43 | 62.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 62.20 | 61.58 | 62.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 61.50 | 61.58 | 62.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 62.18 | 61.70 | 62.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 61.22 | 61.70 | 62.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 58.16 | 59.46 | 60.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 59.67 | 58.76 | 59.65 | SL hit (close>ema200) qty=0.50 sl=58.76 alert=retest2 |

### Cycle 82 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 60.38 | 59.90 | 59.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 61.47 | 60.36 | 60.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 61.39 | 61.56 | 60.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 61.39 | 61.56 | 60.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 61.39 | 61.56 | 60.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 61.48 | 61.56 | 60.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 63.05 | 61.89 | 61.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:30:00 | 61.76 | 61.89 | 61.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 63.30 | 63.12 | 62.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:00:00 | 64.00 | 63.29 | 62.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 63.85 | 64.18 | 64.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 63.85 | 64.18 | 64.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 63.71 | 64.02 | 64.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 63.62 | 63.55 | 63.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 63.62 | 63.55 | 63.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 63.62 | 63.55 | 63.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 62.93 | 63.41 | 63.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 62.75 | 61.72 | 61.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 62.95 | 61.22 | 61.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 62.95 | 61.22 | 61.21 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 62.03 | 62.28 | 62.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 61.66 | 62.06 | 62.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 60.60 | 60.56 | 60.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:00:00 | 60.60 | 60.56 | 60.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 61.77 | 60.80 | 60.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 61.77 | 60.80 | 60.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 61.88 | 61.02 | 61.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 61.83 | 61.02 | 61.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 61.81 | 61.17 | 61.10 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 60.74 | 61.15 | 61.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 60.04 | 60.93 | 61.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 58.93 | 58.47 | 59.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 58.93 | 58.47 | 59.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 58.93 | 58.47 | 59.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 58.93 | 58.47 | 59.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 58.81 | 58.54 | 59.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 58.98 | 58.54 | 59.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 58.95 | 58.62 | 59.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 58.70 | 58.62 | 59.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 59.15 | 58.73 | 59.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 59.11 | 58.73 | 59.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 59.25 | 58.83 | 59.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 59.25 | 58.83 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 59.27 | 58.92 | 59.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 59.27 | 58.92 | 59.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 58.93 | 58.96 | 59.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:45:00 | 58.71 | 58.89 | 59.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 55.77 | 56.94 | 57.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 54.60 | 54.50 | 55.42 | SL hit (close>ema200) qty=0.50 sl=54.50 alert=retest2 |

### Cycle 88 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 53.87 | 53.29 | 53.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 54.21 | 53.76 | 53.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 55.20 | 55.25 | 54.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:45:00 | 55.25 | 55.25 | 54.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 54.73 | 55.15 | 54.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 54.73 | 55.15 | 54.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 54.71 | 55.06 | 54.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 54.35 | 55.06 | 54.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 54.30 | 54.91 | 54.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 54.38 | 54.91 | 54.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 54.13 | 54.75 | 54.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 54.13 | 54.75 | 54.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 54.06 | 54.50 | 54.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 53.73 | 54.18 | 54.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 51.77 | 51.49 | 51.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 51.77 | 51.49 | 51.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 51.77 | 51.49 | 51.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 52.00 | 51.49 | 51.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 52.00 | 51.59 | 51.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 52.00 | 51.59 | 51.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 52.62 | 51.80 | 51.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 52.62 | 51.80 | 51.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 52.47 | 52.15 | 52.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 53.62 | 52.51 | 52.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 53.12 | 53.15 | 52.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:45:00 | 53.09 | 53.15 | 52.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 53.00 | 53.29 | 53.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 52.85 | 53.29 | 53.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 52.96 | 53.23 | 53.10 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 52.47 | 53.01 | 53.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 52.14 | 52.84 | 52.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 52.73 | 52.52 | 52.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 52.73 | 52.52 | 52.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 52.73 | 52.52 | 52.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 52.60 | 52.52 | 52.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 52.55 | 52.53 | 52.69 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 53.76 | 52.78 | 52.77 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 52.81 | 52.91 | 52.91 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 53.10 | 52.94 | 52.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 54.34 | 53.24 | 53.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 53.54 | 53.54 | 53.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:30:00 | 53.58 | 53.54 | 53.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 53.83 | 53.59 | 53.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:30:00 | 54.29 | 53.72 | 53.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 57.07 | 53.62 | 53.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-18 09:15:00 | 59.72 | 58.40 | 57.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 57.68 | 58.64 | 58.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 57.21 | 58.36 | 58.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 53.97 | 53.94 | 54.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 11:30:00 | 53.60 | 53.84 | 54.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 53.98 | 53.63 | 54.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 53.98 | 53.63 | 54.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 54.27 | 53.75 | 54.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 54.27 | 53.75 | 54.26 | SL hit (close>ema400) qty=1.00 sl=54.26 alert=retest1 |

### Cycle 96 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 54.68 | 54.25 | 54.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 55.19 | 54.44 | 54.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 57.30 | 57.38 | 56.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 57.45 | 57.38 | 56.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 56.59 | 57.11 | 56.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 56.59 | 57.11 | 56.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 56.84 | 57.05 | 56.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 58.17 | 56.87 | 56.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 56.83 | 57.91 | 57.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 56.83 | 57.91 | 57.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 56.55 | 57.64 | 57.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 55.30 | 55.30 | 55.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 15:00:00 | 55.30 | 55.30 | 55.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 55.75 | 55.39 | 55.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 56.00 | 55.48 | 55.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 55.66 | 55.52 | 55.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 55.53 | 55.52 | 55.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 55.58 | 55.50 | 55.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 11:15:00 | 55.93 | 55.28 | 55.35 | SL hit (close>static) qty=1.00 sl=55.90 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 55.96 | 55.42 | 55.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 56.67 | 56.14 | 55.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 56.35 | 56.36 | 56.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 56.35 | 56.36 | 56.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 56.41 | 56.37 | 56.11 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 55.61 | 55.95 | 55.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 55.45 | 55.82 | 55.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 11:15:00 | 56.06 | 55.82 | 55.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 56.06 | 55.82 | 55.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 56.06 | 55.82 | 55.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 56.06 | 55.82 | 55.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 56.60 | 55.98 | 55.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 58.55 | 57.12 | 56.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 57.55 | 57.66 | 57.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 58.08 | 57.66 | 57.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 58.04 | 58.14 | 57.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 57.88 | 58.14 | 57.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 57.44 | 58.00 | 57.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 57.44 | 58.00 | 57.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 57.67 | 57.93 | 57.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 57.73 | 57.87 | 57.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 14:15:00 | 57.35 | 57.76 | 57.71 | SL hit (close<static) qty=1.00 sl=57.40 alert=retest2 |

### Cycle 101 — SELL (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 15:15:00 | 57.30 | 57.67 | 57.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 56.28 | 57.39 | 57.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 54.48 | 54.37 | 54.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:30:00 | 54.74 | 54.37 | 54.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 54.89 | 54.47 | 54.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 54.96 | 54.47 | 54.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 54.98 | 54.58 | 54.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:15:00 | 54.85 | 54.58 | 54.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 54.82 | 54.62 | 54.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 54.80 | 54.68 | 54.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 55.60 | 54.47 | 54.58 | SL hit (close>static) qty=1.00 sl=55.25 alert=retest2 |

### Cycle 102 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 55.61 | 54.70 | 54.68 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 54.62 | 54.96 | 54.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 54.59 | 54.88 | 54.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 54.83 | 54.74 | 54.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 11:15:00 | 54.83 | 54.74 | 54.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 54.83 | 54.74 | 54.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 54.83 | 54.74 | 54.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 54.75 | 54.75 | 54.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:30:00 | 54.63 | 54.76 | 54.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 55.80 | 54.95 | 54.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 55.80 | 54.95 | 54.90 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 53.93 | 54.94 | 54.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 53.84 | 54.24 | 54.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 53.75 | 53.72 | 54.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 53.75 | 53.72 | 54.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 53.75 | 53.72 | 54.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:00:00 | 53.42 | 53.64 | 53.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 50.75 | 51.56 | 52.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 51.45 | 50.66 | 51.22 | SL hit (close>ema200) qty=0.50 sl=50.66 alert=retest2 |

### Cycle 106 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 51.79 | 51.45 | 51.44 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 51.25 | 51.48 | 51.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 50.78 | 51.34 | 51.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 51.19 | 51.10 | 51.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 51.19 | 51.10 | 51.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 51.19 | 51.10 | 51.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 51.01 | 51.13 | 51.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 51.55 | 51.29 | 51.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 51.55 | 51.29 | 51.27 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 50.90 | 51.27 | 51.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 50.40 | 51.07 | 51.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 48.05 | 47.88 | 48.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 48.05 | 47.88 | 48.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 48.55 | 48.20 | 48.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 49.28 | 48.20 | 48.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 49.53 | 48.47 | 48.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 49.60 | 48.47 | 48.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 48.74 | 48.52 | 48.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 48.55 | 48.52 | 48.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 48.72 | 48.37 | 48.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 48.72 | 48.37 | 48.33 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 48.03 | 48.36 | 48.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 47.95 | 48.16 | 48.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 47.48 | 47.32 | 47.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 47.48 | 47.32 | 47.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 47.48 | 47.32 | 47.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 47.48 | 47.32 | 47.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 47.15 | 47.09 | 47.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 47.44 | 47.09 | 47.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 47.27 | 47.16 | 47.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 47.30 | 47.16 | 47.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 47.51 | 47.23 | 47.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 47.51 | 47.23 | 47.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 47.89 | 47.36 | 47.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 47.89 | 47.36 | 47.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 49.27 | 47.82 | 47.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 49.44 | 48.88 | 48.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 53.20 | 53.34 | 52.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 53.02 | 53.34 | 52.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 52.62 | 53.20 | 52.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 52.55 | 53.20 | 52.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 52.14 | 52.99 | 52.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 52.14 | 52.99 | 52.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 51.89 | 52.77 | 52.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 51.76 | 52.77 | 52.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 51.63 | 52.22 | 52.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 51.43 | 52.07 | 52.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 53.97 | 51.46 | 51.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 53.97 | 51.46 | 51.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 53.97 | 51.46 | 51.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 54.00 | 51.46 | 51.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 53.58 | 51.88 | 51.78 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 52.48 | 53.06 | 53.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 52.16 | 52.88 | 52.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 51.60 | 49.96 | 50.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 51.60 | 49.96 | 50.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 51.60 | 49.96 | 50.47 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2026-01-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 12:15:00 | 54.20 | 51.43 | 51.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 13:15:00 | 55.05 | 52.16 | 51.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 56.20 | 56.74 | 55.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 56.20 | 56.74 | 55.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 60.09 | 61.30 | 60.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 59.95 | 61.30 | 60.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 59.89 | 61.02 | 60.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 59.89 | 61.02 | 60.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 59.45 | 60.71 | 59.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 59.45 | 60.71 | 59.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 58.35 | 60.24 | 59.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:30:00 | 58.00 | 60.24 | 59.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 58.01 | 59.37 | 59.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 56.92 | 58.00 | 58.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 57.06 | 56.31 | 57.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 57.06 | 56.31 | 57.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 57.06 | 56.31 | 57.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 57.72 | 56.31 | 57.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 57.09 | 56.46 | 57.03 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 57.89 | 57.31 | 57.27 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 56.44 | 57.14 | 57.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 56.09 | 56.79 | 57.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 59.71 | 56.27 | 56.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 59.71 | 56.27 | 56.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 59.71 | 56.27 | 56.31 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 58.96 | 56.81 | 56.55 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 55.31 | 56.65 | 56.66 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 56.84 | 56.48 | 56.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 57.36 | 56.78 | 56.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 58.90 | 59.26 | 58.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:30:00 | 59.00 | 59.26 | 58.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 63.14 | 64.23 | 63.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 62.92 | 64.23 | 63.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 62.55 | 63.89 | 63.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 62.55 | 63.89 | 63.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 62.56 | 63.63 | 63.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:15:00 | 62.56 | 63.63 | 63.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 61.96 | 62.87 | 62.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 60.33 | 61.74 | 62.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 61.75 | 61.74 | 62.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:45:00 | 61.46 | 61.74 | 62.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 62.67 | 61.92 | 62.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:45:00 | 62.04 | 61.92 | 62.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 62.77 | 62.09 | 62.24 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 15:15:00 | 62.84 | 62.36 | 62.35 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 62.16 | 62.31 | 62.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 61.96 | 62.24 | 62.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 62.39 | 62.20 | 62.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 62.39 | 62.20 | 62.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 62.39 | 62.20 | 62.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 62.98 | 62.20 | 62.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 61.93 | 62.15 | 62.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 63.85 | 62.15 | 62.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 63.76 | 62.47 | 62.37 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 62.31 | 62.73 | 62.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 61.81 | 62.43 | 62.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 59.10 | 58.56 | 59.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 59.10 | 58.56 | 59.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 59.10 | 58.56 | 59.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:15:00 | 59.18 | 58.56 | 59.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 59.00 | 58.65 | 59.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:15:00 | 61.01 | 58.65 | 59.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 60.22 | 58.96 | 59.13 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 60.34 | 59.24 | 59.24 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 57.15 | 59.34 | 59.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 56.08 | 57.91 | 58.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 54.42 | 54.23 | 55.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 54.42 | 54.23 | 55.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 55.43 | 54.53 | 55.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 55.42 | 54.53 | 55.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 54.78 | 54.58 | 55.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 54.60 | 54.73 | 55.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 51.87 | 54.21 | 54.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 52.97 | 52.66 | 53.51 | SL hit (close>ema200) qty=0.50 sl=52.66 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 54.31 | 53.67 | 53.64 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 52.14 | 53.38 | 53.54 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 54.00 | 53.68 | 53.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 09:15:00 | 57.06 | 54.44 | 54.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 54.80 | 56.10 | 55.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 54.80 | 56.10 | 55.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 54.80 | 56.10 | 55.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:15:00 | 54.42 | 56.10 | 55.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 54.20 | 55.72 | 55.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 54.60 | 55.72 | 55.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 55.83 | 55.48 | 55.25 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 55.03 | 55.15 | 55.16 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 55.79 | 55.28 | 55.22 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 54.93 | 55.35 | 55.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 54.65 | 55.21 | 55.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 54.86 | 54.83 | 55.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 54.86 | 54.83 | 55.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 54.86 | 54.83 | 55.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 54.29 | 54.59 | 54.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 51.58 | 53.90 | 54.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 52.24 | 51.62 | 52.48 | SL hit (close>ema200) qty=0.50 sl=51.62 alert=retest2 |

### Cycle 136 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 53.19 | 52.75 | 52.71 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 51.55 | 52.58 | 52.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 51.11 | 52.29 | 52.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 52.08 | 49.70 | 50.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 52.08 | 49.70 | 50.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 52.08 | 49.70 | 50.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 52.08 | 49.70 | 50.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 51.55 | 50.07 | 50.59 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 52.52 | 50.96 | 50.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 53.01 | 52.43 | 51.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 52.70 | 52.79 | 52.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 52.70 | 52.79 | 52.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 55.72 | 56.71 | 56.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 56.01 | 56.71 | 56.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 61.61 | 59.69 | 58.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 59.42 | 60.90 | 61.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 59.37 | 60.40 | 60.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 61.00 | 60.15 | 60.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 61.00 | 60.15 | 60.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 61.00 | 60.15 | 60.51 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 61.30 | 60.77 | 60.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 61.64 | 61.07 | 60.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 59.67 | 61.02 | 60.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 59.67 | 61.02 | 60.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 59.67 | 61.02 | 60.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 58.81 | 61.02 | 60.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 59.93 | 60.80 | 60.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 10:15:00 | 59.34 | 60.21 | 60.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 59.12 | 58.67 | 59.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 59.12 | 58.67 | 59.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 59.12 | 58.67 | 59.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 58.84 | 58.79 | 59.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 59.12 | 58.79 | 58.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 59.12 | 58.79 | 58.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 61.41 | 59.31 | 59.02 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-29 11:15:00 | 57.60 | 2024-05-29 13:15:00 | 59.45 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-06-14 14:45:00 | 65.43 | 2024-06-19 09:15:00 | 62.92 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2024-06-18 09:15:00 | 65.34 | 2024-06-19 09:15:00 | 62.92 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-06-25 11:30:00 | 62.65 | 2024-06-25 12:15:00 | 62.60 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-06-28 12:15:00 | 60.95 | 2024-06-28 13:15:00 | 61.91 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-07-01 14:15:00 | 61.03 | 2024-07-04 09:15:00 | 62.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-07-01 15:00:00 | 61.07 | 2024-07-04 09:15:00 | 62.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-18 13:00:00 | 71.63 | 2024-07-19 09:15:00 | 69.30 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-07-18 14:30:00 | 71.55 | 2024-07-19 09:15:00 | 69.30 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-07-19 09:15:00 | 71.65 | 2024-07-19 09:15:00 | 69.30 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-07-30 11:00:00 | 86.79 | 2024-08-01 13:15:00 | 83.68 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-07-31 09:15:00 | 89.15 | 2024-08-01 13:15:00 | 83.68 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest2 | 2024-08-01 11:00:00 | 85.30 | 2024-08-01 13:15:00 | 83.68 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-08-08 09:15:00 | 76.86 | 2024-08-08 15:15:00 | 73.37 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2024-08-08 10:45:00 | 77.23 | 2024-08-08 15:15:00 | 73.33 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2024-08-08 13:00:00 | 77.19 | 2024-08-09 09:15:00 | 73.02 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2024-08-08 09:15:00 | 76.86 | 2024-08-12 09:15:00 | 69.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-08 10:45:00 | 77.23 | 2024-08-12 09:15:00 | 69.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-08 13:00:00 | 77.19 | 2024-08-12 09:15:00 | 69.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-06 09:15:00 | 71.61 | 2024-09-11 12:15:00 | 70.49 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-09-20 12:15:00 | 71.61 | 2024-09-20 14:15:00 | 71.17 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2024-09-20 13:00:00 | 71.56 | 2024-09-20 14:15:00 | 71.17 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2024-09-30 09:15:00 | 65.76 | 2024-10-03 11:15:00 | 62.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 65.76 | 2024-10-04 09:15:00 | 59.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-29 10:30:00 | 52.08 | 2024-10-29 14:15:00 | 53.18 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-10-29 12:15:00 | 52.13 | 2024-10-29 14:15:00 | 53.18 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-10-29 13:30:00 | 51.94 | 2024-10-29 14:15:00 | 53.18 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest1 | 2024-11-14 13:30:00 | 58.86 | 2024-11-18 13:15:00 | 60.69 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest1 | 2024-11-18 10:15:00 | 58.48 | 2024-11-18 13:15:00 | 60.69 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2024-12-24 11:00:00 | 61.15 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-12-24 11:30:00 | 61.19 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-12-24 12:00:00 | 61.06 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-12-27 11:45:00 | 60.79 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2024-12-27 15:15:00 | 60.25 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-12-30 10:15:00 | 60.08 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-01-09 09:15:00 | 58.59 | 2025-01-10 09:15:00 | 55.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:15:00 | 58.45 | 2025-01-10 09:15:00 | 55.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 58.59 | 2025-01-13 11:15:00 | 52.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 10:15:00 | 58.45 | 2025-01-13 12:15:00 | 52.61 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 10:15:00 | 55.35 | 2025-02-01 15:15:00 | 54.18 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest1 | 2025-02-04 10:15:00 | 52.45 | 2025-02-05 09:15:00 | 54.38 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest1 | 2025-02-04 14:15:00 | 52.55 | 2025-02-05 09:15:00 | 54.38 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-02-21 12:30:00 | 46.66 | 2025-02-24 09:15:00 | 45.34 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-02-21 14:30:00 | 46.68 | 2025-02-24 09:15:00 | 45.34 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-02-25 11:45:00 | 45.12 | 2025-02-27 11:15:00 | 42.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 12:30:00 | 45.11 | 2025-02-27 11:15:00 | 42.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 13:15:00 | 45.03 | 2025-02-27 11:15:00 | 42.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:45:00 | 45.05 | 2025-02-27 11:15:00 | 42.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 45.12 | 2025-02-28 12:15:00 | 40.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 12:30:00 | 45.11 | 2025-02-28 12:15:00 | 40.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 13:15:00 | 45.03 | 2025-03-03 09:15:00 | 40.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 14:45:00 | 45.05 | 2025-03-03 09:15:00 | 40.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 12:00:00 | 41.05 | 2025-03-05 09:15:00 | 42.35 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-03-12 11:15:00 | 41.25 | 2025-03-18 15:15:00 | 41.61 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-03-13 10:45:00 | 41.29 | 2025-03-18 15:15:00 | 41.61 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-03-13 11:15:00 | 41.24 | 2025-03-18 15:15:00 | 41.61 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-01 11:30:00 | 43.40 | 2025-04-01 12:15:00 | 44.22 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-04-08 10:30:00 | 40.75 | 2025-04-15 10:15:00 | 42.25 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-04-09 09:15:00 | 40.44 | 2025-04-15 10:15:00 | 42.25 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2025-04-23 13:00:00 | 45.34 | 2025-04-25 09:15:00 | 44.09 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-05-06 11:00:00 | 41.74 | 2025-05-07 09:15:00 | 39.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:00:00 | 41.74 | 2025-05-07 15:15:00 | 40.85 | STOP_HIT | 0.50 | 2.13% |
| BUY | retest2 | 2025-05-23 11:30:00 | 53.51 | 2025-05-26 11:15:00 | 58.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-17 12:15:00 | 61.22 | 2025-06-19 12:15:00 | 58.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:15:00 | 61.22 | 2025-06-20 09:15:00 | 59.67 | STOP_HIT | 0.50 | 2.53% |
| BUY | retest2 | 2025-06-27 14:00:00 | 64.00 | 2025-07-02 15:15:00 | 63.85 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-07-04 11:30:00 | 62.93 | 2025-07-14 13:15:00 | 62.95 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-09 11:30:00 | 62.75 | 2025-07-14 13:15:00 | 62.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-07-30 14:45:00 | 58.71 | 2025-08-06 09:15:00 | 55.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 14:45:00 | 58.71 | 2025-08-07 14:15:00 | 54.60 | STOP_HIT | 0.50 | 7.00% |
| BUY | retest2 | 2025-09-11 10:30:00 | 54.29 | 2025-09-18 09:15:00 | 59.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-15 09:15:00 | 57.07 | 2025-09-23 09:15:00 | 57.68 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest1 | 2025-09-29 11:30:00 | 53.60 | 2025-09-30 10:15:00 | 54.27 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-30 13:30:00 | 53.72 | 2025-10-01 09:15:00 | 54.55 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-10-08 09:15:00 | 58.17 | 2025-10-13 09:15:00 | 56.83 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-10-16 11:15:00 | 55.53 | 2025-10-20 11:15:00 | 55.93 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-16 15:00:00 | 55.58 | 2025-10-20 11:15:00 | 55.93 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-11-03 14:15:00 | 57.73 | 2025-11-03 14:15:00 | 57.35 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-10 12:15:00 | 54.85 | 2025-11-12 09:15:00 | 55.60 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-10 13:00:00 | 54.82 | 2025-11-12 09:15:00 | 55.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-10 14:45:00 | 54.80 | 2025-11-12 09:15:00 | 55.60 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-14 13:30:00 | 54.63 | 2025-11-17 09:15:00 | 55.80 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-11-20 15:00:00 | 53.42 | 2025-11-24 14:15:00 | 50.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 15:00:00 | 53.42 | 2025-11-26 09:15:00 | 51.45 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-12-01 12:15:00 | 51.01 | 2025-12-01 15:15:00 | 51.55 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-10 11:15:00 | 48.55 | 2025-12-12 14:15:00 | 48.72 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-03-06 14:45:00 | 54.60 | 2026-03-09 09:15:00 | 51.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 54.60 | 2026-03-10 09:15:00 | 52.97 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2026-03-20 14:30:00 | 54.29 | 2026-03-23 09:15:00 | 51.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 54.29 | 2026-03-24 12:15:00 | 52.24 | STOP_HIT | 0.50 | 3.78% |
| BUY | retest2 | 2026-04-13 10:15:00 | 56.01 | 2026-04-17 09:15:00 | 61.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 12:00:00 | 58.84 | 2026-05-06 11:15:00 | 59.12 | STOP_HIT | 1.00 | -0.48% |
