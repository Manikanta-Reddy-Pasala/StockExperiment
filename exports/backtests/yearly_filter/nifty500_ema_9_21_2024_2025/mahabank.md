# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 83.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 105 |
| ALERT2 | 104 |
| ALERT2_SKIP | 55 |
| ALERT3 | 271 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 114 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 39 / 88
- **Target hits / Stop hits / Partials:** 6 / 114 / 7
- **Avg / median % per leg:** 0.16% / -0.98%
- **Sum % (uncompounded):** 19.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 24 | 50.0% | 4 | 43 | 1 | 0.84% | 40.5% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 1 | 5 | 1 | 2.39% | 16.7% |
| BUY @ 3rd Alert (retest2) | 41 | 18 | 43.9% | 3 | 38 | 0 | 0.58% | 23.7% |
| SELL (all) | 79 | 15 | 19.0% | 2 | 71 | 6 | -0.26% | -20.5% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.68% | 8.0% |
| SELL @ 3rd Alert (retest2) | 76 | 13 | 17.1% | 2 | 69 | 5 | -0.38% | -28.6% |
| retest1 (combined) | 10 | 8 | 80.0% | 1 | 7 | 2 | 2.48% | 24.8% |
| retest2 (combined) | 117 | 31 | 26.5% | 5 | 107 | 5 | -0.04% | -4.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 64.30 | 63.77 | 63.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 64.95 | 64.11 | 63.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 64.50 | 64.60 | 64.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 15:15:00 | 64.50 | 64.60 | 64.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 64.50 | 64.60 | 64.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 65.10 | 64.60 | 64.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 64.00 | 64.42 | 64.31 | SL hit (close<static) qty=1.00 sl=64.20 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 15:15:00 | 64.10 | 64.25 | 64.25 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 64.35 | 64.27 | 64.26 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 14:15:00 | 64.10 | 64.24 | 64.26 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 64.50 | 64.30 | 64.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 66.45 | 64.76 | 64.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 12:15:00 | 66.95 | 67.02 | 66.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 12:45:00 | 67.05 | 67.02 | 66.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 68.30 | 68.72 | 68.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:30:00 | 68.40 | 68.72 | 68.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 68.25 | 68.63 | 68.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 69.60 | 68.63 | 68.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 10:15:00 | 68.60 | 69.05 | 69.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 68.60 | 69.05 | 69.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 68.25 | 68.86 | 68.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 67.95 | 67.71 | 68.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 12:15:00 | 67.95 | 67.71 | 68.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 67.95 | 67.71 | 68.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 67.95 | 67.71 | 68.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 69.25 | 68.02 | 68.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 69.30 | 68.02 | 68.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 68.95 | 68.20 | 68.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:15:00 | 69.00 | 68.20 | 68.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 69.00 | 68.36 | 68.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 71.95 | 69.08 | 68.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 69.15 | 71.14 | 70.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 69.15 | 71.14 | 70.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 69.15 | 71.14 | 70.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 68.00 | 71.14 | 70.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 65.00 | 69.91 | 69.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 65.00 | 69.91 | 69.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 61.35 | 68.20 | 68.99 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 66.45 | 65.92 | 65.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 67.05 | 66.46 | 66.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 66.34 | 66.47 | 66.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 66.34 | 66.47 | 66.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 66.26 | 66.47 | 66.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 66.29 | 66.47 | 66.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 66.30 | 66.43 | 66.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 66.04 | 66.43 | 66.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 66.33 | 66.41 | 66.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:45:00 | 67.05 | 66.38 | 66.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 66.81 | 66.38 | 66.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 66.79 | 66.92 | 66.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 13:15:00 | 66.30 | 66.76 | 66.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 66.30 | 66.76 | 66.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 65.74 | 66.43 | 66.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 66.43 | 66.02 | 66.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 66.43 | 66.02 | 66.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 66.43 | 66.02 | 66.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 66.43 | 66.02 | 66.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 65.92 | 66.00 | 66.24 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 66.77 | 66.29 | 66.23 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 65.43 | 66.13 | 66.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 65.21 | 65.95 | 66.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 65.65 | 65.49 | 65.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 65.65 | 65.49 | 65.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 65.65 | 65.49 | 65.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 65.74 | 65.49 | 65.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 66.02 | 65.60 | 65.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 66.02 | 65.60 | 65.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 66.08 | 65.69 | 65.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 66.04 | 65.69 | 65.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 65.88 | 65.77 | 65.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:45:00 | 65.89 | 65.77 | 65.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 65.99 | 65.81 | 65.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 65.99 | 65.81 | 65.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 66.05 | 65.86 | 65.85 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 09:15:00 | 65.78 | 65.84 | 65.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 65.52 | 65.78 | 65.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 65.30 | 65.30 | 65.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 65.30 | 65.30 | 65.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 65.30 | 65.30 | 65.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 64.36 | 64.87 | 65.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:15:00 | 64.43 | 64.87 | 65.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:15:00 | 64.41 | 64.80 | 65.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:15:00 | 64.43 | 64.76 | 65.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 65.17 | 64.70 | 64.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 65.17 | 64.70 | 64.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 65.06 | 64.78 | 64.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 65.41 | 64.78 | 64.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 65.34 | 65.03 | 65.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:45:00 | 65.17 | 65.05 | 65.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 13:15:00 | 65.14 | 65.09 | 65.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 65.37 | 65.14 | 65.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 65.37 | 65.14 | 65.12 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 64.91 | 65.10 | 65.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 15:15:00 | 64.74 | 65.03 | 65.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 64.00 | 63.96 | 64.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 63.91 | 63.93 | 64.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 63.91 | 63.93 | 64.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 63.90 | 63.93 | 64.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 64.04 | 63.92 | 64.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:45:00 | 64.11 | 63.92 | 64.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 63.86 | 63.87 | 64.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 63.69 | 63.87 | 64.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 63.79 | 63.77 | 63.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:00:00 | 63.26 | 63.67 | 63.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 13:15:00 | 63.39 | 63.55 | 63.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 65.25 | 63.79 | 63.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 65.25 | 63.79 | 63.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 66.30 | 64.29 | 64.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 63.75 | 64.67 | 64.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 63.75 | 64.67 | 64.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 63.75 | 64.67 | 64.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 63.75 | 64.67 | 64.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 63.59 | 64.45 | 64.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 63.33 | 64.45 | 64.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 63.70 | 64.19 | 64.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 14:15:00 | 63.42 | 63.69 | 63.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 65.30 | 63.98 | 63.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 65.30 | 63.98 | 63.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 65.30 | 63.98 | 63.99 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 65.00 | 64.18 | 64.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 65.34 | 64.90 | 64.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 67.73 | 68.02 | 66.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 14:00:00 | 67.73 | 68.02 | 66.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 69.23 | 68.20 | 67.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 67.41 | 68.20 | 67.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 67.20 | 68.01 | 67.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 67.20 | 68.01 | 67.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 67.54 | 67.91 | 67.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 67.34 | 67.91 | 67.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 65.93 | 67.20 | 67.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 65.70 | 66.48 | 66.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 66.35 | 66.32 | 66.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:30:00 | 66.34 | 66.32 | 66.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 66.45 | 66.30 | 66.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 66.57 | 66.30 | 66.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 66.72 | 66.40 | 66.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 66.25 | 66.40 | 66.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 66.07 | 66.34 | 66.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 64.29 | 66.31 | 66.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 10:00:00 | 65.64 | 65.67 | 66.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 67.38 | 65.92 | 66.03 | SL hit (close>static) qty=1.00 sl=67.24 alert=retest2 |

### Cycle 21 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 67.20 | 66.18 | 66.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 68.01 | 67.06 | 66.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 68.09 | 68.12 | 67.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:15:00 | 68.20 | 68.12 | 67.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 68.13 | 68.14 | 67.72 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 10:15:00 | 67.19 | 67.70 | 67.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 66.88 | 67.39 | 67.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 62.40 | 62.33 | 63.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 12:00:00 | 62.40 | 62.33 | 63.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 63.04 | 62.56 | 63.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:45:00 | 62.57 | 62.59 | 62.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 63.22 | 62.72 | 63.01 | SL hit (close>static) qty=1.00 sl=63.15 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 62.56 | 61.32 | 61.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 62.73 | 62.21 | 61.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 62.88 | 63.23 | 62.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 14:15:00 | 62.88 | 63.23 | 62.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 62.88 | 63.23 | 62.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 62.88 | 63.23 | 62.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 62.65 | 63.11 | 62.94 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 62.58 | 62.81 | 62.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 62.37 | 62.67 | 62.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 62.62 | 62.61 | 62.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 62.62 | 62.61 | 62.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 62.62 | 62.61 | 62.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 62.40 | 62.61 | 62.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 15:15:00 | 63.25 | 62.59 | 62.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 15:15:00 | 63.25 | 62.59 | 62.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 63.34 | 62.74 | 62.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 13:15:00 | 62.88 | 62.97 | 62.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 14:00:00 | 62.88 | 62.97 | 62.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 62.92 | 62.96 | 62.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:30:00 | 62.56 | 62.96 | 62.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 62.78 | 62.93 | 62.78 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 62.30 | 62.65 | 62.68 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 63.16 | 62.69 | 62.68 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 62.61 | 62.67 | 62.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 15:15:00 | 62.54 | 62.64 | 62.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 59.66 | 59.51 | 60.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 59.66 | 59.51 | 60.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 59.66 | 59.51 | 60.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:15:00 | 59.22 | 59.54 | 59.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 59.90 | 59.37 | 59.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 59.90 | 59.37 | 59.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 61.10 | 59.72 | 59.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 60.53 | 60.73 | 60.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 60.53 | 60.73 | 60.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 60.05 | 60.58 | 60.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 60.05 | 60.58 | 60.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 59.98 | 60.46 | 60.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 59.98 | 60.46 | 60.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 59.99 | 60.37 | 60.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:15:00 | 60.02 | 60.37 | 60.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 14:15:00 | 59.86 | 60.20 | 60.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 15:15:00 | 59.79 | 60.12 | 60.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 58.55 | 58.47 | 58.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:45:00 | 58.77 | 58.47 | 58.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 58.95 | 58.57 | 58.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 58.85 | 58.57 | 58.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 59.00 | 58.66 | 58.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 59.00 | 58.66 | 58.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 58.84 | 58.69 | 58.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 58.92 | 58.69 | 58.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 58.80 | 58.60 | 58.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 59.44 | 58.60 | 58.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 59.74 | 58.83 | 58.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 59.84 | 58.83 | 58.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 60.77 | 59.22 | 59.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 61.24 | 59.62 | 59.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 61.43 | 61.63 | 60.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 15:00:00 | 61.43 | 61.63 | 60.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 60.33 | 61.35 | 60.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 60.33 | 61.35 | 60.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 60.26 | 61.13 | 60.86 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 60.42 | 60.71 | 60.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 59.64 | 60.35 | 60.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 10:15:00 | 60.52 | 60.39 | 60.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 10:15:00 | 60.52 | 60.39 | 60.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 60.52 | 60.39 | 60.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 60.89 | 60.39 | 60.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 60.15 | 60.34 | 60.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 59.97 | 60.34 | 60.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 60.04 | 60.13 | 60.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:30:00 | 60.04 | 60.17 | 60.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:15:00 | 60.09 | 60.17 | 60.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 59.84 | 60.06 | 60.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:45:00 | 60.08 | 60.06 | 60.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 59.82 | 60.01 | 60.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:45:00 | 60.04 | 60.01 | 60.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 60.60 | 60.05 | 60.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 60.60 | 60.05 | 60.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 60.41 | 60.12 | 60.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:15:00 | 60.27 | 60.20 | 60.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 60.63 | 60.28 | 60.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 13:15:00 | 60.63 | 60.28 | 60.25 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 59.88 | 60.20 | 60.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 59.11 | 59.90 | 60.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 56.50 | 56.44 | 57.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 56.50 | 56.44 | 57.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 57.18 | 56.73 | 57.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 55.96 | 56.73 | 57.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 12:15:00 | 54.68 | 54.44 | 54.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 12:15:00 | 54.68 | 54.44 | 54.42 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 53.97 | 54.37 | 54.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 53.13 | 53.93 | 54.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 54.42 | 53.79 | 53.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 54.42 | 53.79 | 53.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 54.42 | 53.79 | 53.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 54.51 | 53.79 | 53.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 54.18 | 53.87 | 53.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 54.30 | 53.87 | 53.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 53.83 | 53.88 | 53.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 53.68 | 53.81 | 53.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 51.00 | 52.69 | 53.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 48.31 | 50.36 | 51.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 50.53 | 49.90 | 49.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 12:15:00 | 51.00 | 50.12 | 49.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 52.08 | 52.12 | 51.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 52.08 | 52.12 | 51.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 53.43 | 54.38 | 53.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 53.36 | 54.38 | 53.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 53.51 | 54.21 | 53.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 53.88 | 54.14 | 53.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 54.13 | 54.27 | 53.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:45:00 | 53.82 | 54.22 | 53.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 54.40 | 55.15 | 55.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 54.40 | 55.15 | 55.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 54.12 | 54.86 | 55.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 53.89 | 53.84 | 54.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 12:30:00 | 53.21 | 53.61 | 54.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 13:15:00 | 50.55 | 51.66 | 52.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 51.04 | 51.01 | 51.90 | SL hit (close>ema200) qty=0.50 sl=51.01 alert=retest1 |

### Cycle 39 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 53.00 | 51.80 | 51.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 53.11 | 52.06 | 51.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 52.26 | 52.34 | 52.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 52.26 | 52.34 | 52.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 51.01 | 52.07 | 51.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 50.85 | 52.07 | 51.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 51.32 | 51.92 | 51.92 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 51.47 | 51.83 | 51.88 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 52.49 | 51.92 | 51.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 52.71 | 52.07 | 51.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 56.40 | 56.40 | 55.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 56.40 | 56.40 | 55.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 55.86 | 56.29 | 55.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:00:00 | 55.86 | 56.29 | 55.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 55.90 | 56.21 | 55.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 55.90 | 56.21 | 55.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 56.44 | 56.26 | 55.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 56.89 | 56.26 | 55.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:15:00 | 56.55 | 56.37 | 56.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 11:15:00 | 57.79 | 57.97 | 57.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 11:15:00 | 57.79 | 57.97 | 57.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 12:15:00 | 57.65 | 57.91 | 57.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 55.47 | 55.41 | 55.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 55.47 | 55.41 | 55.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 55.69 | 55.47 | 55.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 55.72 | 55.47 | 55.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 56.08 | 55.59 | 55.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 56.35 | 55.59 | 55.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 56.08 | 55.69 | 55.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 56.29 | 55.69 | 55.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 56.16 | 56.01 | 55.99 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 55.92 | 55.98 | 55.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 55.51 | 55.89 | 55.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 53.61 | 53.35 | 53.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 53.61 | 53.35 | 53.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 54.05 | 53.49 | 53.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:45:00 | 54.26 | 53.49 | 53.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 53.65 | 53.52 | 53.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:45:00 | 53.57 | 53.53 | 53.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 54.68 | 53.76 | 53.91 | SL hit (close>static) qty=1.00 sl=54.21 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 09:15:00 | 54.46 | 54.07 | 54.04 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 11:15:00 | 53.86 | 54.10 | 54.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 12:15:00 | 53.75 | 54.03 | 54.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 51.88 | 51.71 | 52.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 51.88 | 51.71 | 52.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 52.23 | 51.82 | 52.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 52.23 | 51.82 | 52.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 52.09 | 51.87 | 52.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 52.32 | 51.95 | 52.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 52.58 | 52.08 | 52.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 52.53 | 52.08 | 52.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 52.65 | 52.38 | 52.34 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 52.03 | 52.31 | 52.32 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 52.81 | 52.42 | 52.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 54.78 | 52.98 | 52.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 53.91 | 54.57 | 53.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 53.91 | 54.57 | 53.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 53.91 | 54.57 | 53.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 54.01 | 54.57 | 53.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 53.24 | 54.31 | 53.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 53.24 | 54.31 | 53.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 53.28 | 54.10 | 53.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 52.91 | 54.10 | 53.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 52.57 | 53.37 | 53.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 51.93 | 52.69 | 53.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 48.99 | 48.94 | 49.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:15:00 | 48.36 | 48.94 | 49.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 48.39 | 47.66 | 48.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 48.86 | 47.90 | 48.51 | SL hit (close>ema400) qty=1.00 sl=48.51 alert=retest1 |

### Cycle 51 — BUY (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 14:15:00 | 52.19 | 49.23 | 48.98 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 51.35 | 51.53 | 51.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 49.90 | 51.18 | 51.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 50.79 | 50.37 | 50.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 50.79 | 50.37 | 50.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 50.79 | 50.37 | 50.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 50.92 | 50.37 | 50.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 50.71 | 50.43 | 50.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 50.82 | 50.43 | 50.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 50.53 | 50.45 | 50.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 50.25 | 50.55 | 50.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 13:15:00 | 49.96 | 49.42 | 49.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 50.28 | 49.71 | 49.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 50.28 | 49.71 | 49.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 50.72 | 49.91 | 49.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 50.28 | 50.42 | 50.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 50.28 | 50.42 | 50.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 50.27 | 50.39 | 50.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 50.29 | 50.39 | 50.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 50.32 | 50.37 | 50.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 50.22 | 50.37 | 50.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 50.43 | 50.38 | 50.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:15:00 | 50.98 | 50.44 | 50.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 50.94 | 51.10 | 50.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 50.98 | 50.99 | 50.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:30:00 | 50.80 | 50.97 | 50.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 50.50 | 50.87 | 50.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 50.50 | 50.87 | 50.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 50.45 | 50.79 | 50.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 49.95 | 50.79 | 50.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 49.91 | 50.61 | 50.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 49.91 | 50.61 | 50.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 12:15:00 | 49.40 | 50.10 | 50.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 50.48 | 49.96 | 50.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 50.48 | 49.96 | 50.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 50.48 | 49.96 | 50.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 50.78 | 49.96 | 50.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 49.99 | 49.97 | 50.18 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 51.15 | 50.33 | 50.27 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 50.06 | 50.60 | 50.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 49.70 | 50.42 | 50.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 48.40 | 48.12 | 48.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 48.60 | 48.12 | 48.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 49.11 | 48.32 | 48.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 49.11 | 48.32 | 48.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 48.52 | 48.36 | 48.72 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 49.17 | 48.88 | 48.87 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 48.14 | 48.75 | 48.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 47.62 | 48.41 | 48.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 48.25 | 47.79 | 48.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 13:15:00 | 48.25 | 47.79 | 48.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 48.25 | 47.79 | 48.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 48.25 | 47.79 | 48.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 48.85 | 48.00 | 48.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 48.85 | 48.00 | 48.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 48.13 | 48.16 | 48.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:30:00 | 48.23 | 48.16 | 48.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 48.17 | 48.14 | 48.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 48.17 | 48.14 | 48.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 49.00 | 48.31 | 48.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 49.74 | 48.70 | 48.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 49.75 | 50.01 | 49.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 49.75 | 50.01 | 49.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 49.97 | 50.00 | 49.64 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 49.06 | 49.51 | 49.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 48.88 | 49.31 | 49.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 49.26 | 49.18 | 49.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 49.26 | 49.18 | 49.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 49.26 | 49.18 | 49.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 49.10 | 49.18 | 49.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 46.64 | 47.88 | 48.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 10:15:00 | 44.19 | 46.01 | 47.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 46.66 | 45.99 | 45.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 46.79 | 46.15 | 46.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 47.70 | 47.79 | 47.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 47.70 | 47.79 | 47.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 47.61 | 47.92 | 47.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 47.61 | 47.92 | 47.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 47.40 | 47.82 | 47.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 46.79 | 47.82 | 47.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 46.90 | 47.63 | 47.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 46.83 | 47.63 | 47.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 47.10 | 47.53 | 47.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 46.79 | 47.09 | 47.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 46.15 | 45.18 | 45.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 46.15 | 45.18 | 45.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 46.15 | 45.18 | 45.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 46.15 | 45.18 | 45.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 46.09 | 45.36 | 45.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:15:00 | 45.84 | 45.50 | 45.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 46.78 | 45.93 | 45.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 46.78 | 45.93 | 45.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 47.14 | 46.17 | 45.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 47.23 | 47.26 | 46.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 47.23 | 47.26 | 46.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 47.80 | 47.34 | 46.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 47.92 | 47.34 | 46.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:15:00 | 48.28 | 47.58 | 47.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 46.86 | 47.81 | 47.76 | SL hit (close<static) qty=1.00 sl=46.94 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 46.63 | 47.57 | 47.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 46.24 | 46.81 | 47.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 46.79 | 46.50 | 46.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 46.79 | 46.50 | 46.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 46.79 | 46.50 | 46.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 46.79 | 46.50 | 46.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 46.81 | 46.57 | 46.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 47.40 | 46.57 | 46.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 47.54 | 46.76 | 46.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 47.66 | 46.76 | 46.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 46.64 | 46.61 | 46.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 46.63 | 46.61 | 46.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 46.31 | 46.55 | 46.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 45.58 | 46.41 | 46.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:00:00 | 46.18 | 46.26 | 46.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 46.12 | 46.24 | 46.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 46.88 | 46.41 | 46.43 | SL hit (close>static) qty=1.00 sl=46.70 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 46.66 | 46.46 | 46.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 47.28 | 46.66 | 46.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 46.58 | 46.94 | 46.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 46.58 | 46.94 | 46.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 46.58 | 46.94 | 46.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 46.58 | 46.94 | 46.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 46.68 | 46.89 | 46.72 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 43.29 | 46.00 | 46.35 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 45.37 | 45.17 | 45.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 45.72 | 45.36 | 45.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 15:15:00 | 45.66 | 45.66 | 45.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:15:00 | 46.28 | 45.66 | 45.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 09:15:00 | 48.59 | 47.31 | 46.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-04-21 10:15:00 | 50.91 | 49.04 | 48.00 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 68 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 49.44 | 50.20 | 50.24 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 13:15:00 | 50.59 | 50.30 | 50.28 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 49.65 | 50.15 | 50.21 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 50.81 | 50.28 | 50.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 12:15:00 | 50.90 | 50.51 | 50.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 51.10 | 51.28 | 50.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 51.10 | 51.28 | 50.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 51.10 | 51.28 | 50.99 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 50.35 | 50.81 | 50.86 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 51.03 | 50.89 | 50.89 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 50.54 | 50.82 | 50.86 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 51.59 | 50.87 | 50.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 51.63 | 51.19 | 51.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 51.17 | 51.42 | 51.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 51.17 | 51.42 | 51.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 51.17 | 51.42 | 51.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 51.17 | 51.42 | 51.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 51.21 | 51.38 | 51.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 12:00:00 | 51.45 | 51.40 | 51.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 50.09 | 51.08 | 51.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 50.09 | 51.08 | 51.12 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 51.20 | 50.84 | 50.83 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 50.45 | 50.81 | 50.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 47.97 | 50.18 | 50.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 50.38 | 49.17 | 49.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 50.38 | 49.17 | 49.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 50.38 | 49.17 | 49.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 50.49 | 49.17 | 49.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 50.70 | 49.97 | 49.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 50.90 | 50.28 | 50.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 50.51 | 50.62 | 50.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 50.51 | 50.62 | 50.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 50.53 | 50.60 | 50.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 50.66 | 50.61 | 50.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 50.67 | 50.56 | 50.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 50.71 | 50.57 | 50.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:15:00 | 50.68 | 50.59 | 50.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 50.50 | 50.57 | 50.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 50.50 | 50.57 | 50.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 50.48 | 50.55 | 50.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 50.48 | 50.55 | 50.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 50.50 | 50.54 | 50.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 50.84 | 50.54 | 50.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 51.57 | 51.79 | 51.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 51.57 | 51.79 | 51.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 51.04 | 51.64 | 51.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 50.78 | 50.73 | 51.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 50.78 | 50.73 | 51.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 51.17 | 50.82 | 51.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 51.14 | 50.82 | 51.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 51.14 | 50.88 | 51.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:30:00 | 50.97 | 50.96 | 51.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 51.14 | 51.10 | 51.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 51.14 | 51.10 | 51.10 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 50.98 | 51.08 | 51.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 50.67 | 50.99 | 51.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 51.01 | 50.98 | 51.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 51.01 | 50.98 | 51.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 51.01 | 50.98 | 51.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:00:00 | 50.84 | 50.95 | 51.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 50.88 | 50.96 | 51.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 51.32 | 51.03 | 51.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 51.32 | 51.03 | 51.03 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 50.99 | 51.09 | 51.09 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 51.28 | 51.12 | 51.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 51.45 | 51.18 | 51.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 56.32 | 56.45 | 55.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 12:00:00 | 56.32 | 56.45 | 55.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 56.00 | 56.31 | 56.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 55.84 | 56.31 | 56.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 55.87 | 56.22 | 56.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 55.87 | 56.22 | 56.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 56.04 | 56.19 | 56.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 56.00 | 56.19 | 56.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 56.02 | 56.15 | 56.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 55.36 | 56.15 | 56.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 55.03 | 55.93 | 55.95 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 57.21 | 56.03 | 55.91 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 55.79 | 56.38 | 56.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 55.45 | 55.97 | 56.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 54.48 | 54.19 | 54.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 54.48 | 54.19 | 54.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 54.58 | 54.26 | 54.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 54.44 | 54.26 | 54.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 54.60 | 54.33 | 54.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 54.57 | 54.33 | 54.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 54.72 | 54.40 | 54.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 54.72 | 54.40 | 54.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 54.66 | 54.46 | 54.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:45:00 | 54.54 | 54.71 | 54.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 54.25 | 54.69 | 54.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 54.82 | 54.47 | 54.56 | SL hit (close>static) qty=1.00 sl=54.80 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 54.86 | 54.06 | 53.95 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 53.80 | 54.17 | 54.22 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 54.87 | 54.26 | 54.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 56.50 | 54.86 | 54.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 57.43 | 57.57 | 56.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 57.43 | 57.57 | 56.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 57.33 | 57.27 | 56.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 57.40 | 57.27 | 56.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 11:00:00 | 57.38 | 57.25 | 57.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 57.39 | 57.21 | 57.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 57.53 | 57.27 | 57.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 57.26 | 57.33 | 57.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 57.28 | 57.33 | 57.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 57.22 | 57.31 | 57.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 57.37 | 57.30 | 57.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 56.89 | 57.22 | 57.21 | SL hit (close<static) qty=1.00 sl=57.14 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 56.72 | 57.12 | 57.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 56.53 | 57.00 | 57.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 56.90 | 56.87 | 57.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 56.90 | 56.87 | 57.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 56.70 | 56.83 | 56.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:30:00 | 56.44 | 56.63 | 56.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 57.11 | 56.26 | 56.30 | SL hit (close>static) qty=1.00 sl=57.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 57.12 | 56.43 | 56.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 57.32 | 56.85 | 56.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 57.30 | 57.32 | 57.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:30:00 | 57.35 | 57.32 | 57.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 57.38 | 57.32 | 57.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:30:00 | 57.39 | 57.32 | 57.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 57.23 | 57.30 | 57.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 57.27 | 57.30 | 57.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 56.84 | 57.21 | 57.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 56.84 | 57.21 | 57.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 56.86 | 57.14 | 57.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 56.55 | 57.14 | 57.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 56.74 | 57.06 | 57.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 56.64 | 56.98 | 57.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 57.04 | 56.88 | 56.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 57.04 | 56.88 | 56.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 57.04 | 56.88 | 56.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 56.96 | 56.88 | 56.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 57.06 | 56.92 | 56.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 12:30:00 | 56.89 | 56.94 | 56.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 56.90 | 56.86 | 56.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 56.80 | 56.82 | 56.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 56.85 | 56.82 | 56.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 56.65 | 56.79 | 56.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 56.59 | 56.79 | 56.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 56.60 | 56.75 | 56.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 56.46 | 56.37 | 56.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 57.13 | 56.56 | 56.59 | SL hit (close>static) qty=1.00 sl=56.95 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 57.18 | 56.68 | 56.64 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 56.84 | 56.90 | 56.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 55.92 | 56.49 | 56.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 54.97 | 54.86 | 55.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 54.97 | 54.86 | 55.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 55.15 | 54.99 | 55.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 55.39 | 54.99 | 55.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 55.30 | 55.05 | 55.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 55.09 | 55.13 | 55.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 55.09 | 55.14 | 55.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 54.99 | 55.14 | 55.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 52.34 | 53.39 | 54.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 52.34 | 53.39 | 54.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 53.53 | 53.42 | 54.04 | SL hit (close>ema200) qty=0.50 sl=53.42 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 55.44 | 54.51 | 54.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 14:15:00 | 55.70 | 54.91 | 54.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 55.35 | 55.38 | 55.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:45:00 | 55.32 | 55.38 | 55.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 55.33 | 55.50 | 55.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 55.33 | 55.50 | 55.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 55.07 | 55.42 | 55.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 55.07 | 55.42 | 55.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 55.35 | 55.40 | 55.31 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 54.94 | 55.21 | 55.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 54.40 | 54.86 | 55.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 55.18 | 54.83 | 54.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 55.18 | 54.83 | 54.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 55.18 | 54.83 | 54.99 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 55.22 | 55.02 | 55.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 55.44 | 55.14 | 55.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 55.18 | 55.20 | 55.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:45:00 | 55.15 | 55.20 | 55.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 55.20 | 55.20 | 55.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 55.16 | 55.20 | 55.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 55.18 | 55.20 | 55.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:45:00 | 55.17 | 55.20 | 55.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 55.21 | 55.21 | 55.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 55.34 | 55.21 | 55.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 55.39 | 55.24 | 55.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 55.64 | 55.35 | 55.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 55.61 | 55.34 | 55.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 55.04 | 55.26 | 55.24 | SL hit (close<static) qty=1.00 sl=55.06 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 54.99 | 55.21 | 55.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 54.91 | 55.15 | 55.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 52.28 | 52.04 | 52.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 52.29 | 52.04 | 52.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 52.95 | 52.22 | 52.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 53.01 | 52.22 | 52.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 53.00 | 52.38 | 52.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 53.09 | 52.38 | 52.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 12:15:00 | 52.89 | 52.59 | 52.58 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 52.47 | 52.57 | 52.57 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 52.79 | 52.60 | 52.58 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 52.33 | 52.61 | 52.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 52.06 | 52.41 | 52.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 52.51 | 52.24 | 52.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 12:15:00 | 52.51 | 52.24 | 52.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 52.51 | 52.24 | 52.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 52.51 | 52.24 | 52.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 52.57 | 52.31 | 52.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 52.55 | 52.31 | 52.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 52.75 | 52.45 | 52.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 53.23 | 52.61 | 52.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 52.91 | 52.96 | 52.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 52.91 | 52.96 | 52.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 52.91 | 52.96 | 52.79 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 52.38 | 52.69 | 52.70 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 53.38 | 52.75 | 52.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 53.72 | 52.94 | 52.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 54.18 | 54.42 | 53.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 54.18 | 54.42 | 53.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 54.18 | 54.42 | 53.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:15:00 | 54.80 | 54.29 | 54.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 54.59 | 54.50 | 54.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 54.62 | 54.52 | 54.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 56.63 | 57.03 | 57.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 56.63 | 57.03 | 57.03 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 57.09 | 57.04 | 57.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 58.15 | 57.28 | 57.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 57.13 | 57.25 | 57.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 57.13 | 57.25 | 57.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 57.13 | 57.25 | 57.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 57.13 | 57.25 | 57.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 56.84 | 57.17 | 57.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 56.84 | 57.17 | 57.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 57.08 | 57.15 | 57.11 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 56.33 | 56.95 | 57.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 56.17 | 56.68 | 56.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 54.96 | 54.62 | 55.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 54.96 | 54.62 | 55.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 55.06 | 54.71 | 55.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 54.91 | 54.71 | 55.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 56.06 | 55.19 | 55.28 | SL hit (close>static) qty=1.00 sl=55.54 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 56.46 | 55.57 | 55.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 56.79 | 56.21 | 55.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 55.99 | 56.20 | 56.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 55.99 | 56.20 | 56.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 55.99 | 56.20 | 56.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 55.95 | 56.20 | 56.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 56.47 | 56.25 | 56.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 57.15 | 56.31 | 56.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 56.60 | 56.85 | 56.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 56.63 | 56.81 | 56.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 56.63 | 56.81 | 56.81 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 57.72 | 56.92 | 56.84 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 56.37 | 57.22 | 57.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 54.97 | 56.47 | 56.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 57.03 | 56.44 | 56.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 57.03 | 56.44 | 56.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 57.03 | 56.44 | 56.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 57.03 | 56.44 | 56.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 57.91 | 56.73 | 56.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 57.91 | 56.73 | 56.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 58.94 | 57.18 | 57.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 59.89 | 57.72 | 57.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 57.89 | 58.41 | 57.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 57.89 | 58.41 | 57.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 57.89 | 58.41 | 57.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 57.92 | 58.41 | 57.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 57.90 | 58.31 | 57.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 57.81 | 58.31 | 57.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 57.94 | 58.23 | 57.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 57.56 | 58.23 | 57.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 57.51 | 58.09 | 57.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 57.51 | 58.09 | 57.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 57.75 | 58.02 | 57.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 57.22 | 58.02 | 57.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 57.43 | 57.76 | 57.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 56.84 | 57.57 | 57.69 | Break + close below crossover candle low |

### Cycle 117 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 59.10 | 57.68 | 57.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 59.18 | 58.18 | 57.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 59.57 | 59.62 | 59.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 12:00:00 | 59.57 | 59.62 | 59.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 59.34 | 59.48 | 59.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 59.34 | 59.48 | 59.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 59.03 | 59.39 | 59.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 58.88 | 59.39 | 59.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 58.92 | 59.29 | 59.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 58.92 | 59.29 | 59.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 58.86 | 59.02 | 59.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 58.29 | 58.87 | 58.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 15:15:00 | 59.35 | 58.64 | 58.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 15:15:00 | 59.35 | 58.64 | 58.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 59.35 | 58.64 | 58.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 60.38 | 58.64 | 58.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 59.26 | 58.76 | 58.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 59.71 | 58.76 | 58.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 60.25 | 59.11 | 58.95 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 58.93 | 59.35 | 59.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 58.46 | 59.17 | 59.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 59.13 | 58.94 | 59.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 59.13 | 58.94 | 59.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 59.13 | 58.94 | 59.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 59.13 | 58.94 | 59.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 58.70 | 58.89 | 59.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 59.07 | 58.89 | 59.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 59.99 | 59.11 | 59.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 59.95 | 59.11 | 59.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 59.49 | 59.19 | 59.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:15:00 | 59.12 | 59.19 | 59.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 59.13 | 59.18 | 59.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 59.04 | 59.15 | 59.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 60.41 | 59.39 | 59.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 60.41 | 59.39 | 59.30 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 58.68 | 59.27 | 59.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 58.41 | 58.93 | 59.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 57.99 | 57.90 | 58.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 57.77 | 57.90 | 58.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 58.38 | 57.99 | 58.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 58.36 | 57.99 | 58.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 58.78 | 58.15 | 58.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 58.78 | 58.15 | 58.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 59.35 | 58.39 | 58.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 59.35 | 58.39 | 58.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 59.61 | 58.64 | 58.59 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 57.90 | 58.57 | 58.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 56.93 | 58.10 | 58.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 57.49 | 57.38 | 57.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 57.49 | 57.38 | 57.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 57.49 | 57.38 | 57.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 57.52 | 57.38 | 57.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 57.55 | 57.42 | 57.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:45:00 | 57.72 | 57.42 | 57.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 57.58 | 57.45 | 57.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 57.70 | 57.45 | 57.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 57.65 | 57.48 | 57.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 57.65 | 57.48 | 57.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 57.59 | 57.50 | 57.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 57.45 | 57.50 | 57.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 57.32 | 57.45 | 57.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 57.38 | 57.21 | 57.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 57.94 | 57.40 | 57.46 | SL hit (close>static) qty=1.00 sl=57.68 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 13:15:00 | 57.81 | 57.54 | 57.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 58.05 | 57.64 | 57.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 60.15 | 60.23 | 59.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:00:00 | 60.15 | 60.23 | 59.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 60.28 | 60.51 | 60.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 60.28 | 60.51 | 60.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 60.14 | 60.43 | 60.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 60.14 | 60.43 | 60.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 59.91 | 60.33 | 60.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 59.91 | 60.33 | 60.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 59.83 | 60.23 | 60.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 59.88 | 60.23 | 60.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 59.80 | 60.14 | 60.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:45:00 | 59.86 | 60.14 | 60.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 59.00 | 59.85 | 59.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 58.47 | 59.57 | 59.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 59.37 | 59.17 | 59.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 59.37 | 59.17 | 59.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 59.37 | 59.17 | 59.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 59.37 | 59.17 | 59.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 59.19 | 59.17 | 59.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 58.69 | 59.06 | 59.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 58.80 | 58.76 | 59.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 58.84 | 58.81 | 59.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 58.90 | 58.87 | 59.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 59.11 | 58.91 | 59.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 59.11 | 58.91 | 59.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 59.68 | 59.07 | 59.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 59.68 | 59.07 | 59.14 | SL hit (close>static) qty=1.00 sl=59.55 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 59.60 | 59.26 | 59.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 60.40 | 59.49 | 59.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 59.65 | 59.83 | 59.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 59.65 | 59.83 | 59.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 59.65 | 59.83 | 59.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 59.64 | 59.83 | 59.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 59.67 | 59.80 | 59.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 59.63 | 59.80 | 59.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 59.62 | 59.76 | 59.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 59.61 | 59.76 | 59.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 59.22 | 59.66 | 59.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 59.22 | 59.66 | 59.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 59.12 | 59.55 | 59.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 58.72 | 59.38 | 59.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 59.37 | 58.91 | 59.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 59.37 | 58.91 | 59.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 59.37 | 58.91 | 59.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 59.45 | 58.91 | 59.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 58.83 | 58.90 | 59.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 57.98 | 58.71 | 58.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:30:00 | 57.51 | 58.03 | 58.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 56.95 | 56.79 | 56.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 56.95 | 56.79 | 56.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 57.48 | 56.93 | 56.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 56.91 | 57.03 | 56.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 12:15:00 | 56.91 | 57.03 | 56.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 56.91 | 57.03 | 56.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 56.91 | 57.03 | 56.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 56.79 | 56.99 | 56.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 56.79 | 56.99 | 56.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 56.80 | 56.95 | 56.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:15:00 | 56.60 | 56.95 | 56.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 56.60 | 56.88 | 56.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 56.98 | 56.88 | 56.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 56.85 | 57.14 | 57.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 56.85 | 57.14 | 57.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 56.57 | 57.03 | 57.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 56.76 | 56.68 | 56.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 56.76 | 56.68 | 56.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 56.76 | 56.68 | 56.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 56.76 | 56.68 | 56.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 56.52 | 56.65 | 56.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 56.50 | 56.65 | 56.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 56.33 | 56.58 | 56.74 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 57.04 | 56.70 | 56.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 57.40 | 56.94 | 56.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 57.74 | 57.74 | 57.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:30:00 | 57.62 | 57.74 | 57.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 57.54 | 57.70 | 57.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 57.54 | 57.70 | 57.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 57.63 | 57.68 | 57.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 57.50 | 57.68 | 57.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 57.59 | 57.67 | 57.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 57.59 | 57.67 | 57.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 57.41 | 57.61 | 57.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 57.41 | 57.61 | 57.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 57.31 | 57.55 | 57.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 57.31 | 57.55 | 57.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 57.24 | 57.44 | 57.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 56.98 | 57.32 | 57.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 57.53 | 57.31 | 57.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 57.53 | 57.31 | 57.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 57.53 | 57.31 | 57.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 57.53 | 57.31 | 57.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 57.37 | 57.32 | 57.37 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 57.91 | 57.44 | 57.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 58.42 | 57.84 | 57.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 64.06 | 64.07 | 63.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 64.06 | 64.07 | 63.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 63.79 | 64.29 | 63.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 63.66 | 64.29 | 63.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 63.46 | 64.12 | 63.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 63.37 | 64.12 | 63.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 63.65 | 64.03 | 63.90 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 63.21 | 63.77 | 63.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 62.40 | 63.31 | 63.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 63.15 | 62.62 | 63.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 63.15 | 62.62 | 63.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 63.15 | 62.62 | 63.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 63.15 | 62.62 | 63.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 63.27 | 62.75 | 63.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 63.70 | 62.75 | 63.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 62.69 | 62.76 | 63.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 62.46 | 62.68 | 62.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 62.57 | 62.66 | 62.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 62.25 | 62.67 | 62.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 63.35 | 62.80 | 62.93 | SL hit (close>static) qty=1.00 sl=63.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 63.59 | 63.09 | 63.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 63.97 | 63.48 | 63.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 64.97 | 66.37 | 66.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 64.97 | 66.37 | 66.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 64.97 | 66.37 | 66.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 64.99 | 66.37 | 66.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 64.73 | 66.04 | 65.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 64.73 | 66.04 | 65.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 64.37 | 65.71 | 65.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 64.01 | 65.37 | 65.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 64.85 | 64.21 | 64.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 64.85 | 64.21 | 64.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 64.85 | 64.21 | 64.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 64.99 | 64.21 | 64.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 64.90 | 64.35 | 64.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 64.87 | 64.35 | 64.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 65.14 | 64.51 | 64.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:15:00 | 65.32 | 64.51 | 64.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 65.67 | 64.85 | 64.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 67.09 | 65.56 | 65.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 65.92 | 66.16 | 65.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:45:00 | 65.89 | 66.16 | 65.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 65.64 | 66.05 | 65.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 65.64 | 66.05 | 65.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 65.42 | 65.93 | 65.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 64.75 | 65.93 | 65.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 65.41 | 65.82 | 65.60 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 64.57 | 65.33 | 65.40 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 65.74 | 65.31 | 65.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 65.83 | 65.44 | 65.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 65.64 | 65.64 | 65.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 65.64 | 65.64 | 65.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 65.64 | 65.64 | 65.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 65.64 | 65.64 | 65.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 65.94 | 65.72 | 65.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 64.85 | 65.72 | 65.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 65.38 | 65.65 | 65.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 65.58 | 65.65 | 65.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 65.45 | 65.61 | 65.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:15:00 | 65.43 | 65.61 | 65.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 65.28 | 65.54 | 65.50 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 65.19 | 65.46 | 65.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 09:15:00 | 63.94 | 65.16 | 65.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 61.30 | 61.27 | 62.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 61.30 | 61.27 | 62.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 63.35 | 61.74 | 62.51 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 64.10 | 63.00 | 62.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 64.44 | 63.50 | 63.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 64.88 | 65.06 | 64.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 65.58 | 65.06 | 64.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 65.35 | 65.18 | 64.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:30:00 | 65.28 | 65.21 | 64.77 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 14:45:00 | 65.40 | 65.20 | 64.87 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 66.26 | 66.13 | 65.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:15:00 | 67.01 | 66.13 | 65.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 66.11 | 66.68 | 66.26 | SL hit (close<ema400) qty=1.00 sl=66.26 alert=retest1 |

### Cycle 142 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 65.66 | 66.24 | 66.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 65.50 | 66.10 | 66.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 65.90 | 65.81 | 66.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:30:00 | 65.91 | 65.81 | 66.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 65.48 | 65.12 | 65.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 65.48 | 65.12 | 65.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 65.68 | 65.23 | 65.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 65.68 | 65.23 | 65.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 65.57 | 65.30 | 65.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 65.98 | 65.30 | 65.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 67.15 | 65.78 | 65.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 67.45 | 66.11 | 65.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 68.55 | 68.60 | 67.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 68.55 | 68.60 | 67.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 67.64 | 68.32 | 67.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 68.11 | 68.32 | 67.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 68.77 | 68.41 | 68.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 70.66 | 68.69 | 68.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 69.52 | 69.33 | 68.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 69.48 | 69.34 | 68.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 09:15:00 | 76.47 | 74.91 | 73.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 72.29 | 73.72 | 73.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 14:15:00 | 72.00 | 73.15 | 73.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 70.40 | 69.50 | 70.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 70.31 | 69.50 | 70.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 70.74 | 70.01 | 70.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 14:00:00 | 70.07 | 70.02 | 70.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 71.11 | 70.24 | 70.75 | SL hit (close>static) qty=1.00 sl=71.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 64.19 | 63.72 | 63.68 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 62.83 | 63.54 | 63.60 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 64.93 | 63.43 | 63.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 65.75 | 63.89 | 63.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 62.15 | 64.35 | 64.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 62.15 | 64.35 | 64.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 62.15 | 64.35 | 64.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 62.15 | 64.35 | 64.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 62.22 | 63.92 | 63.94 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 65.87 | 63.87 | 63.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 66.50 | 64.39 | 63.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 63.52 | 65.01 | 64.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 63.52 | 65.01 | 64.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 63.52 | 65.01 | 64.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 63.52 | 65.01 | 64.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 63.27 | 64.67 | 64.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 63.12 | 64.67 | 64.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 64.63 | 64.47 | 64.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:15:00 | 64.35 | 64.47 | 64.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 64.58 | 64.49 | 64.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 64.58 | 64.49 | 64.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 62.58 | 64.10 | 64.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 61.40 | 62.61 | 63.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 63.82 | 62.64 | 63.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 63.82 | 62.64 | 63.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 63.82 | 62.64 | 63.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 63.93 | 62.64 | 63.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 65.42 | 63.62 | 63.57 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 62.06 | 63.41 | 63.56 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 64.15 | 63.59 | 63.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 65.34 | 63.94 | 63.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 69.95 | 69.98 | 68.94 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 70.80 | 69.98 | 68.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 69.89 | 70.02 | 69.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:45:00 | 69.65 | 70.02 | 69.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 68.95 | 70.00 | 69.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 68.95 | 70.00 | 69.55 | SL hit (close<ema400) qty=1.00 sl=69.55 alert=retest1 |

### Cycle 154 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 76.70 | 77.96 | 78.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 76.06 | 77.58 | 77.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 77.64 | 77.26 | 77.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 77.64 | 77.26 | 77.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 77.64 | 77.26 | 77.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 77.64 | 77.26 | 77.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 78.25 | 77.46 | 77.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 80.04 | 77.46 | 77.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 79.30 | 77.83 | 77.81 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 77.95 | 78.45 | 78.46 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 78.71 | 78.50 | 78.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 80.10 | 78.82 | 78.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 77.77 | 79.19 | 79.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 77.77 | 79.19 | 79.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 77.77 | 79.19 | 79.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 77.77 | 79.19 | 79.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 78.24 | 79.00 | 78.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 77.95 | 79.00 | 78.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 78.67 | 78.87 | 78.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 78.32 | 78.76 | 78.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 79.36 | 78.77 | 78.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 79.36 | 78.77 | 78.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 79.36 | 78.77 | 78.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 79.89 | 78.77 | 78.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 79.45 | 78.90 | 78.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 79.83 | 79.09 | 78.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 15:15:00 | 79.11 | 79.26 | 79.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 15:15:00 | 79.11 | 79.26 | 79.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 79.11 | 79.26 | 79.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 78.63 | 79.26 | 79.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 79.69 | 79.34 | 79.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:00:00 | 80.05 | 79.48 | 79.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 80.19 | 79.63 | 79.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 65.10 | 2024-05-16 12:15:00 | 64.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-05-27 09:15:00 | 69.60 | 2024-05-29 10:15:00 | 68.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-06-11 13:45:00 | 67.05 | 2024-06-14 13:15:00 | 66.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-06-12 09:15:00 | 66.81 | 2024-06-14 13:15:00 | 66.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-14 09:45:00 | 66.79 | 2024-06-14 13:15:00 | 66.30 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-06-27 10:30:00 | 64.36 | 2024-06-28 13:15:00 | 65.37 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-06-27 11:15:00 | 64.43 | 2024-06-28 13:15:00 | 65.37 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-06-27 12:15:00 | 64.41 | 2024-06-28 13:15:00 | 65.37 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-06-27 13:15:00 | 64.43 | 2024-06-28 13:15:00 | 65.37 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-06-28 11:45:00 | 65.17 | 2024-06-28 13:15:00 | 65.37 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-06-28 13:15:00 | 65.14 | 2024-06-28 13:15:00 | 65.37 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-07-08 11:00:00 | 63.26 | 2024-07-09 09:15:00 | 65.25 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-07-08 13:15:00 | 63.39 | 2024-07-09 09:15:00 | 65.25 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-07-23 12:15:00 | 64.29 | 2024-07-24 13:15:00 | 67.38 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-07-24 10:00:00 | 65.64 | 2024-07-24 13:15:00 | 67.38 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-08-08 09:45:00 | 62.57 | 2024-08-08 10:15:00 | 63.22 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-08-08 12:45:00 | 62.24 | 2024-08-19 09:15:00 | 62.56 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-09 09:30:00 | 62.40 | 2024-08-19 09:15:00 | 62.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-08-26 10:15:00 | 62.40 | 2024-08-27 15:15:00 | 63.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-09-11 13:15:00 | 59.22 | 2024-09-13 11:15:00 | 59.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-09-26 12:15:00 | 59.97 | 2024-09-30 13:15:00 | 60.63 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-09-27 09:15:00 | 60.04 | 2024-09-30 13:15:00 | 60.63 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-27 11:30:00 | 60.04 | 2024-09-30 13:15:00 | 60.63 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-27 12:15:00 | 60.09 | 2024-09-30 13:15:00 | 60.63 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-30 13:15:00 | 60.27 | 2024-09-30 13:15:00 | 60.63 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-09 09:15:00 | 55.96 | 2024-10-16 12:15:00 | 54.68 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2024-10-21 11:30:00 | 53.68 | 2024-10-22 09:15:00 | 51.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 53.68 | 2024-10-23 09:15:00 | 48.31 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-04 13:00:00 | 53.88 | 2024-11-08 11:15:00 | 54.40 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2024-11-05 09:45:00 | 54.13 | 2024-11-08 11:15:00 | 54.40 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-11-05 11:45:00 | 53.82 | 2024-11-08 11:15:00 | 54.40 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest1 | 2024-11-12 12:30:00 | 53.21 | 2024-11-13 13:15:00 | 50.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-12 12:30:00 | 53.21 | 2024-11-14 11:15:00 | 51.04 | STOP_HIT | 0.50 | 4.08% |
| BUY | retest2 | 2024-11-29 14:15:00 | 56.89 | 2024-12-09 11:15:00 | 57.79 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2024-12-02 13:15:00 | 56.55 | 2024-12-09 11:15:00 | 57.79 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2024-12-20 13:45:00 | 53.57 | 2024-12-20 14:15:00 | 54.68 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest1 | 2025-01-13 09:15:00 | 48.36 | 2025-01-14 10:15:00 | 48.86 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-23 15:15:00 | 50.25 | 2025-01-29 10:15:00 | 50.28 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-01-27 13:15:00 | 49.96 | 2025-01-29 10:15:00 | 50.28 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-01-31 11:15:00 | 50.98 | 2025-02-03 09:15:00 | 49.91 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-02-01 12:00:00 | 50.94 | 2025-02-03 09:15:00 | 49.91 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-02-01 12:45:00 | 50.98 | 2025-02-03 09:15:00 | 49.91 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-02-01 13:30:00 | 50.80 | 2025-02-03 09:15:00 | 49.91 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-02-25 10:15:00 | 49.10 | 2025-02-28 09:15:00 | 46.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 10:15:00 | 49.10 | 2025-03-03 10:15:00 | 44.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-18 12:15:00 | 45.84 | 2025-03-19 09:15:00 | 46.78 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-03-21 10:15:00 | 47.92 | 2025-03-25 11:15:00 | 46.86 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-03-21 14:15:00 | 48.28 | 2025-03-25 11:15:00 | 46.86 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-04-02 09:15:00 | 45.58 | 2025-04-03 10:15:00 | 46.88 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-04-02 11:00:00 | 46.18 | 2025-04-03 10:15:00 | 46.88 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-04-02 12:45:00 | 46.12 | 2025-04-03 10:15:00 | 46.88 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest1 | 2025-04-15 09:15:00 | 46.28 | 2025-04-17 09:15:00 | 48.59 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-15 09:15:00 | 46.28 | 2025-04-21 10:15:00 | 50.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-06 12:00:00 | 51.45 | 2025-05-06 14:15:00 | 50.09 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-05-14 09:15:00 | 50.66 | 2025-05-21 10:15:00 | 51.57 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2025-05-15 09:15:00 | 50.67 | 2025-05-21 10:15:00 | 51.57 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-05-15 10:30:00 | 50.71 | 2025-05-21 10:15:00 | 51.57 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2025-05-15 13:15:00 | 50.68 | 2025-05-21 10:15:00 | 51.57 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-05-16 09:15:00 | 50.84 | 2025-05-21 10:15:00 | 51.57 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2025-05-23 14:30:00 | 50.97 | 2025-05-26 12:15:00 | 51.14 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-05-27 14:00:00 | 50.84 | 2025-05-28 09:15:00 | 51.32 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-27 14:30:00 | 50.88 | 2025-05-28 09:15:00 | 51.32 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-17 12:45:00 | 54.54 | 2025-06-18 12:15:00 | 54.82 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-06-17 15:15:00 | 54.25 | 2025-06-18 12:15:00 | 54.82 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-06-18 13:30:00 | 54.45 | 2025-06-24 09:15:00 | 54.99 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-18 15:00:00 | 54.51 | 2025-06-24 09:15:00 | 54.99 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-06-19 10:45:00 | 53.60 | 2025-06-24 09:15:00 | 54.99 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-06-23 09:30:00 | 53.40 | 2025-06-24 09:15:00 | 54.99 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-06-23 13:00:00 | 53.56 | 2025-06-24 09:15:00 | 54.99 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-06-23 14:15:00 | 53.55 | 2025-06-24 09:15:00 | 54.99 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-07-03 10:15:00 | 57.40 | 2025-07-08 09:15:00 | 56.89 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-04 11:00:00 | 57.38 | 2025-07-08 10:15:00 | 56.72 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-04 14:00:00 | 57.39 | 2025-07-08 10:15:00 | 56.72 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-04 14:30:00 | 57.53 | 2025-07-08 10:15:00 | 56.72 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-08 09:15:00 | 57.37 | 2025-07-08 10:15:00 | 56.72 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-11 09:30:00 | 56.44 | 2025-07-15 09:15:00 | 57.11 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-21 12:30:00 | 56.89 | 2025-07-24 09:15:00 | 57.13 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-21 13:30:00 | 56.90 | 2025-07-24 09:15:00 | 57.13 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-22 09:30:00 | 56.80 | 2025-07-24 09:15:00 | 57.13 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-22 10:15:00 | 56.85 | 2025-07-24 10:15:00 | 57.18 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-22 11:15:00 | 56.59 | 2025-07-24 10:15:00 | 57.18 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-22 12:00:00 | 56.60 | 2025-07-24 10:15:00 | 57.18 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-23 15:00:00 | 56.46 | 2025-07-24 10:15:00 | 57.18 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-05 14:00:00 | 55.09 | 2025-08-07 13:15:00 | 52.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 14:30:00 | 55.09 | 2025-08-07 13:15:00 | 52.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 14:00:00 | 55.09 | 2025-08-07 14:15:00 | 53.53 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-08-05 14:30:00 | 55.09 | 2025-08-07 14:15:00 | 53.53 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-08-06 09:15:00 | 54.99 | 2025-08-08 12:15:00 | 55.44 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-08 09:45:00 | 54.80 | 2025-08-08 12:15:00 | 55.44 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-21 10:30:00 | 55.64 | 2025-08-22 09:15:00 | 55.04 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-08-21 14:30:00 | 55.61 | 2025-08-22 09:15:00 | 55.04 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-09-15 14:15:00 | 54.80 | 2025-09-23 11:15:00 | 56.63 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest2 | 2025-09-16 09:45:00 | 54.59 | 2025-09-23 11:15:00 | 56.63 | STOP_HIT | 1.00 | 3.74% |
| BUY | retest2 | 2025-09-16 11:00:00 | 54.62 | 2025-09-23 11:15:00 | 56.63 | STOP_HIT | 1.00 | 3.68% |
| SELL | retest2 | 2025-09-29 11:15:00 | 54.91 | 2025-09-30 09:15:00 | 56.06 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-10-06 09:15:00 | 57.15 | 2025-10-08 14:15:00 | 56.63 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-08 12:45:00 | 56.60 | 2025-10-08 14:15:00 | 56.63 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-10-31 13:15:00 | 59.12 | 2025-11-03 09:15:00 | 60.41 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-31 14:00:00 | 59.13 | 2025-11-03 09:15:00 | 60.41 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-10-31 15:00:00 | 59.04 | 2025-11-03 09:15:00 | 60.41 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-11-13 09:15:00 | 57.45 | 2025-11-14 11:15:00 | 57.94 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-13 10:45:00 | 57.32 | 2025-11-14 11:15:00 | 57.94 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-11-14 10:00:00 | 57.38 | 2025-11-14 11:15:00 | 57.94 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-24 13:45:00 | 58.69 | 2025-11-25 13:15:00 | 59.68 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-25 10:00:00 | 58.80 | 2025-11-25 13:15:00 | 59.68 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-11-25 10:45:00 | 58.84 | 2025-11-25 13:15:00 | 59.68 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-25 12:15:00 | 58.90 | 2025-11-25 13:15:00 | 59.68 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-12-01 12:00:00 | 57.98 | 2025-12-09 15:15:00 | 56.95 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2025-12-02 11:30:00 | 57.51 | 2025-12-09 15:15:00 | 56.95 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-12-11 09:15:00 | 56.98 | 2025-12-16 10:15:00 | 56.85 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-01-09 13:45:00 | 62.46 | 2026-01-12 09:15:00 | 63.35 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-01-09 15:00:00 | 62.57 | 2026-01-12 09:15:00 | 63.35 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-01-12 09:15:00 | 62.25 | 2026-01-12 09:15:00 | 63.35 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest1 | 2026-02-06 09:15:00 | 65.58 | 2026-02-11 09:15:00 | 66.11 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest1 | 2026-02-06 11:00:00 | 65.35 | 2026-02-11 09:15:00 | 66.11 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest1 | 2026-02-06 11:30:00 | 65.28 | 2026-02-11 09:15:00 | 66.11 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest1 | 2026-02-06 14:45:00 | 65.40 | 2026-02-11 09:15:00 | 66.11 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2026-02-10 10:15:00 | 67.01 | 2026-02-12 14:15:00 | 65.66 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-12 09:45:00 | 66.68 | 2026-02-12 14:15:00 | 65.66 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-23 09:15:00 | 70.66 | 2026-02-27 09:15:00 | 76.47 | TARGET_HIT | 1.00 | 8.23% |
| BUY | retest2 | 2026-02-23 12:30:00 | 69.52 | 2026-02-27 09:15:00 | 76.43 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2026-02-23 14:15:00 | 69.48 | 2026-03-02 12:15:00 | 72.29 | STOP_HIT | 1.00 | 4.04% |
| SELL | retest2 | 2026-03-05 14:00:00 | 70.07 | 2026-03-05 14:15:00 | 71.11 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-03-06 15:15:00 | 69.99 | 2026-03-09 09:15:00 | 66.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 69.99 | 2026-03-10 09:15:00 | 67.06 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest1 | 2026-04-10 09:15:00 | 70.80 | 2026-04-13 09:15:00 | 68.95 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-04-13 11:45:00 | 69.56 | 2026-04-20 14:15:00 | 76.52 | TARGET_HIT | 1.00 | 10.00% |
