# Samvardhana Motherson International Ltd. (MOTHERSON)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 131.57
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 54 |
| ALERT2 | 53 |
| ALERT2_SKIP | 25 |
| ALERT3 | 144 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 86 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 86 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 74
- **Target hits / Stop hits / Partials:** 3 / 86 / 4
- **Avg / median % per leg:** -0.30% / -0.93%
- **Sum % (uncompounded):** -27.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 10 | 16.7% | 2 | 58 | 0 | -0.65% | -38.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.49% | -3.0% |
| BUY @ 3rd Alert (retest2) | 58 | 10 | 17.2% | 2 | 56 | 0 | -0.62% | -35.8% |
| SELL (all) | 33 | 9 | 27.3% | 1 | 28 | 4 | 0.34% | 11.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.62% | -0.6% |
| SELL @ 3rd Alert (retest2) | 32 | 9 | 28.1% | 1 | 27 | 4 | 0.37% | 11.8% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.20% | -3.6% |
| retest2 (combined) | 90 | 19 | 21.1% | 3 | 83 | 4 | -0.27% | -24.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 95.24 | 93.51 | 93.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 96.43 | 94.34 | 93.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 94.77 | 95.37 | 94.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 94.77 | 95.37 | 94.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 95.34 | 95.36 | 94.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:30:00 | 95.51 | 95.20 | 94.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 10:00:00 | 95.57 | 95.20 | 94.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 94.55 | 95.05 | 94.86 | SL hit (close<static) qty=1.00 sl=94.73 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 94.28 | 94.73 | 94.74 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 95.13 | 94.75 | 94.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 95.81 | 94.96 | 94.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 98.13 | 98.30 | 97.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 97.81 | 98.19 | 97.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 97.81 | 98.19 | 97.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 97.81 | 98.19 | 97.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 98.16 | 98.18 | 97.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 98.16 | 98.18 | 97.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 97.31 | 97.96 | 97.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 97.31 | 97.96 | 97.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 97.56 | 97.88 | 97.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 98.16 | 98.00 | 97.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 98.49 | 98.00 | 97.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 98.05 | 98.03 | 97.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:45:00 | 98.37 | 98.29 | 98.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 98.57 | 98.34 | 98.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 99.95 | 98.69 | 98.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 14:15:00 | 99.04 | 99.21 | 99.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 99.04 | 99.21 | 99.21 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 100.66 | 99.33 | 99.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 10:15:00 | 101.19 | 99.70 | 99.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 98.82 | 99.75 | 99.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 13:15:00 | 98.82 | 99.75 | 99.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 98.82 | 99.75 | 99.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 98.82 | 99.75 | 99.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 101.33 | 100.07 | 99.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 105.43 | 100.33 | 99.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 102.90 | 101.25 | 100.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 102.67 | 101.48 | 100.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:30:00 | 102.60 | 101.76 | 100.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 102.17 | 101.85 | 101.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 100.57 | 101.00 | 101.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 100.57 | 101.00 | 101.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 100.19 | 100.84 | 100.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 101.13 | 100.67 | 100.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 101.13 | 100.67 | 100.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 101.13 | 100.67 | 100.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 101.13 | 100.67 | 100.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 101.60 | 100.85 | 100.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 101.68 | 100.85 | 100.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 101.85 | 101.05 | 100.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 102.10 | 101.46 | 101.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 106.27 | 106.44 | 105.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:45:00 | 106.19 | 106.44 | 105.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 106.03 | 106.16 | 105.70 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 104.70 | 105.50 | 105.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 104.37 | 105.27 | 105.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 102.82 | 102.50 | 103.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 102.82 | 102.50 | 103.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 100.61 | 100.27 | 100.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 101.05 | 100.27 | 100.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 99.61 | 100.14 | 100.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 100.23 | 100.14 | 100.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 99.99 | 99.47 | 99.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 100.00 | 99.47 | 99.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 99.93 | 99.56 | 99.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 99.93 | 99.56 | 99.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 102.80 | 99.49 | 99.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 102.92 | 102.26 | 101.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 102.82 | 102.85 | 102.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 102.82 | 102.85 | 102.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 102.95 | 102.89 | 102.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 102.95 | 102.89 | 102.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 102.18 | 102.82 | 102.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 102.18 | 102.82 | 102.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 102.72 | 102.80 | 102.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:15:00 | 102.93 | 102.80 | 102.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:15:00 | 102.97 | 102.80 | 102.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 102.89 | 102.90 | 102.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 101.65 | 102.48 | 102.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 101.65 | 102.48 | 102.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 101.11 | 101.77 | 102.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 102.06 | 101.78 | 102.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 102.06 | 101.78 | 102.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 102.06 | 101.78 | 102.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 102.06 | 101.78 | 102.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 101.79 | 101.78 | 102.00 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 103.41 | 102.15 | 102.12 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 102.28 | 102.47 | 102.47 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 103.37 | 102.65 | 102.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 103.79 | 102.99 | 102.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 102.60 | 103.09 | 102.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 102.60 | 103.09 | 102.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 102.60 | 103.09 | 102.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 102.60 | 103.09 | 102.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 101.75 | 102.83 | 102.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 101.75 | 102.83 | 102.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 102.30 | 102.72 | 102.74 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 102.94 | 102.72 | 102.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 103.61 | 102.95 | 102.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 102.93 | 103.13 | 102.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 15:15:00 | 102.93 | 103.13 | 102.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 102.93 | 103.13 | 102.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 103.57 | 103.13 | 102.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 102.58 | 103.02 | 102.93 | SL hit (close<static) qty=1.00 sl=102.87 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 102.22 | 102.86 | 102.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 102.05 | 102.70 | 102.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 102.55 | 102.50 | 102.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 102.55 | 102.50 | 102.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 102.55 | 102.50 | 102.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 102.07 | 102.47 | 102.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 101.79 | 101.59 | 101.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 103.03 | 101.91 | 101.95 | SL hit (close>static) qty=1.00 sl=102.99 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 103.42 | 102.22 | 102.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 105.27 | 102.83 | 102.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 102.41 | 103.43 | 102.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 102.41 | 103.43 | 102.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 102.41 | 103.43 | 102.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 102.55 | 103.43 | 102.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 103.03 | 103.35 | 102.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:00:00 | 103.40 | 103.36 | 102.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:45:00 | 103.40 | 103.37 | 103.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 102.67 | 102.94 | 102.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 102.67 | 102.94 | 102.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 102.37 | 102.71 | 102.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 12:15:00 | 98.49 | 98.46 | 99.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 13:00:00 | 98.49 | 98.46 | 99.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 101.95 | 99.29 | 99.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 101.95 | 99.29 | 99.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 102.26 | 99.88 | 99.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 12:15:00 | 103.06 | 100.95 | 100.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 102.00 | 102.06 | 101.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 11:15:00 | 101.69 | 102.06 | 101.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 101.02 | 101.85 | 101.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 101.24 | 101.85 | 101.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 100.95 | 101.67 | 101.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 101.94 | 101.07 | 100.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 101.33 | 101.10 | 101.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 13:45:00 | 101.35 | 101.35 | 101.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:15:00 | 101.43 | 101.53 | 101.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 101.99 | 101.62 | 101.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 101.30 | 101.62 | 101.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 100.52 | 101.77 | 101.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 100.52 | 101.77 | 101.60 | SL hit (close<static) qty=1.00 sl=100.66 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 100.30 | 101.30 | 101.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 12:15:00 | 99.87 | 101.02 | 101.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 96.89 | 96.48 | 97.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:30:00 | 96.80 | 96.48 | 97.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 97.44 | 96.83 | 97.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 97.44 | 96.83 | 97.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 97.91 | 97.05 | 97.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 97.91 | 97.05 | 97.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 98.31 | 97.30 | 97.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 98.30 | 97.30 | 97.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 98.49 | 97.80 | 97.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 98.89 | 98.02 | 97.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 97.83 | 98.02 | 97.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 97.83 | 98.02 | 97.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 97.83 | 98.02 | 97.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 97.83 | 98.02 | 97.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 97.51 | 97.92 | 97.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 96.95 | 97.92 | 97.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 97.49 | 97.83 | 97.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 96.17 | 97.83 | 97.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 96.00 | 97.47 | 97.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 95.42 | 96.29 | 96.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 94.50 | 94.26 | 95.28 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 09:30:00 | 93.72 | 94.09 | 95.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 91.45 | 90.96 | 91.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 91.78 | 90.96 | 91.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 91.47 | 91.06 | 91.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 90.88 | 91.06 | 91.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 90.99 | 91.06 | 91.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 94.30 | 91.75 | 91.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 23 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 94.30 | 91.75 | 91.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 95.06 | 93.15 | 92.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 97.95 | 97.99 | 97.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 13:45:00 | 98.14 | 98.01 | 97.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 14:45:00 | 98.05 | 98.01 | 97.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 96.63 | 97.70 | 97.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 96.63 | 97.70 | 97.38 | SL hit (close<ema400) qty=1.00 sl=97.38 alert=retest1 |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 95.81 | 97.10 | 97.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 95.68 | 96.81 | 97.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 95.88 | 95.87 | 96.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 95.88 | 95.87 | 96.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 93.74 | 92.74 | 93.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 93.74 | 92.74 | 93.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 93.89 | 92.97 | 93.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 94.49 | 92.97 | 93.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 94.51 | 93.35 | 93.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:15:00 | 94.15 | 93.35 | 93.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 12:15:00 | 94.73 | 93.72 | 93.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 94.73 | 93.72 | 93.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 95.42 | 94.06 | 93.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 96.03 | 96.22 | 95.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 96.03 | 96.22 | 95.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 96.03 | 96.22 | 95.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 95.38 | 96.22 | 95.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 95.96 | 96.31 | 95.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 95.96 | 96.31 | 95.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 95.56 | 96.16 | 95.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 95.56 | 96.16 | 95.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 94.80 | 95.89 | 95.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 94.80 | 95.89 | 95.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 94.21 | 95.55 | 95.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 93.68 | 94.87 | 95.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 94.63 | 94.62 | 95.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 94.63 | 94.62 | 95.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 94.63 | 94.62 | 95.05 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 98.35 | 95.72 | 95.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 13:15:00 | 98.92 | 97.17 | 96.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 100.55 | 100.74 | 99.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 100.55 | 100.74 | 99.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 108.96 | 109.37 | 108.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 110.05 | 109.39 | 108.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 110.00 | 109.14 | 108.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 110.01 | 109.24 | 109.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 14:45:00 | 110.03 | 109.39 | 109.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 111.20 | 109.81 | 109.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:45:00 | 112.04 | 110.25 | 109.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 109.21 | 110.71 | 110.39 | SL hit (close<static) qty=1.00 sl=109.37 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 108.83 | 110.09 | 110.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 107.98 | 109.67 | 109.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 105.59 | 105.56 | 106.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 15:00:00 | 105.59 | 105.56 | 106.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 106.68 | 105.84 | 106.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 106.68 | 105.84 | 106.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 106.87 | 106.04 | 106.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 106.87 | 106.04 | 106.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 105.91 | 106.02 | 106.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 105.49 | 106.04 | 106.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 105.55 | 106.30 | 106.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:00:00 | 105.57 | 105.66 | 106.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 105.59 | 105.72 | 106.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 105.89 | 105.79 | 106.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 105.91 | 105.79 | 106.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 106.11 | 105.86 | 106.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 106.11 | 105.86 | 106.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 105.99 | 105.88 | 106.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 106.26 | 105.88 | 106.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 106.24 | 105.96 | 106.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 106.72 | 106.18 | 106.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 106.72 | 106.18 | 106.12 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 105.40 | 106.09 | 106.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 104.86 | 105.85 | 106.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 102.92 | 102.62 | 103.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 102.92 | 102.62 | 103.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 103.51 | 102.86 | 103.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 103.51 | 102.86 | 103.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 103.31 | 102.95 | 103.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 103.05 | 102.95 | 103.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 104.22 | 103.20 | 103.52 | SL hit (close>static) qty=1.00 sl=103.75 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 104.73 | 103.79 | 103.74 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 102.65 | 103.58 | 103.68 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 104.35 | 103.75 | 103.72 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 103.39 | 103.70 | 103.71 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 12:15:00 | 103.94 | 103.75 | 103.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 13:15:00 | 104.27 | 103.85 | 103.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 11:15:00 | 104.35 | 104.55 | 104.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 12:00:00 | 104.35 | 104.55 | 104.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 104.17 | 104.47 | 104.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 104.17 | 104.47 | 104.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 104.25 | 104.43 | 104.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 104.25 | 104.43 | 104.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 104.75 | 104.49 | 104.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 105.82 | 104.58 | 104.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 104.73 | 105.56 | 105.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 104.73 | 105.56 | 105.59 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 105.88 | 105.52 | 105.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 108.49 | 106.10 | 105.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 106.94 | 107.58 | 106.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 106.94 | 107.58 | 106.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 106.45 | 107.32 | 106.80 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 106.30 | 106.64 | 106.64 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 107.05 | 106.68 | 106.65 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 106.25 | 106.64 | 106.65 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 106.93 | 106.70 | 106.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 107.28 | 106.86 | 106.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 106.53 | 106.85 | 106.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 106.53 | 106.85 | 106.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 106.53 | 106.85 | 106.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 106.53 | 106.85 | 106.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 106.81 | 106.84 | 106.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 106.65 | 106.84 | 106.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 106.34 | 106.74 | 106.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 106.34 | 106.74 | 106.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 107.09 | 106.81 | 106.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:30:00 | 107.66 | 106.89 | 106.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:00:00 | 107.19 | 107.00 | 106.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 107.20 | 107.02 | 106.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 107.45 | 107.11 | 106.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 107.09 | 107.25 | 107.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 106.72 | 107.25 | 107.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 106.64 | 107.13 | 107.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 106.64 | 107.13 | 107.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 106.60 | 107.02 | 106.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 106.49 | 107.02 | 106.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 107.10 | 107.10 | 107.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 107.09 | 107.10 | 107.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 106.86 | 107.05 | 107.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 106.57 | 107.05 | 107.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 106.90 | 107.02 | 107.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 107.53 | 107.02 | 107.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 107.07 | 107.03 | 107.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 106.63 | 106.95 | 106.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 106.63 | 106.95 | 106.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 106.07 | 106.68 | 106.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 105.31 | 105.19 | 105.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:45:00 | 105.41 | 105.19 | 105.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 106.08 | 105.37 | 105.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 105.34 | 105.37 | 105.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 105.20 | 105.33 | 105.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 104.99 | 105.26 | 105.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:15:00 | 104.88 | 105.08 | 105.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 105.30 | 103.63 | 103.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 105.30 | 103.63 | 103.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 105.80 | 104.06 | 103.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 104.07 | 105.43 | 105.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 104.07 | 105.43 | 105.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 104.07 | 105.43 | 105.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 104.07 | 105.43 | 105.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 109.26 | 106.20 | 105.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 111.60 | 106.20 | 105.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 111.44 | 108.56 | 106.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 110.04 | 109.31 | 107.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 110.24 | 109.39 | 107.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 108.90 | 109.65 | 108.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 108.90 | 109.65 | 108.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 108.59 | 109.44 | 108.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 108.62 | 109.44 | 108.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 109.51 | 109.45 | 108.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 13:45:00 | 109.70 | 109.51 | 109.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 109.80 | 109.33 | 109.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 109.76 | 110.91 | 111.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 109.76 | 110.91 | 111.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 109.45 | 110.41 | 110.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 110.47 | 109.89 | 110.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 110.47 | 109.89 | 110.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 110.47 | 109.89 | 110.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 110.47 | 109.89 | 110.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 110.19 | 109.95 | 110.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 110.18 | 109.95 | 110.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 110.28 | 110.02 | 110.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:30:00 | 110.23 | 110.02 | 110.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 110.40 | 110.09 | 110.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:00:00 | 110.40 | 110.09 | 110.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 110.82 | 110.24 | 110.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 110.82 | 110.24 | 110.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 110.33 | 110.26 | 110.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 110.08 | 110.26 | 110.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 111.61 | 110.50 | 110.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 111.61 | 110.50 | 110.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 112.24 | 111.50 | 111.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 115.86 | 116.15 | 114.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 115.74 | 116.15 | 114.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 118.02 | 116.67 | 115.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 11:15:00 | 118.66 | 117.00 | 116.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 118.80 | 117.36 | 116.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 118.65 | 117.69 | 116.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 116.78 | 117.18 | 117.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 116.78 | 117.18 | 117.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 116.59 | 117.06 | 117.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 116.25 | 115.84 | 116.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 116.25 | 115.84 | 116.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 116.25 | 115.84 | 116.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 115.80 | 115.84 | 116.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 117.01 | 116.08 | 116.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 117.01 | 116.08 | 116.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 117.12 | 116.28 | 116.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 117.12 | 116.28 | 116.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 116.88 | 116.40 | 116.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:30:00 | 116.64 | 116.53 | 116.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 116.90 | 116.60 | 116.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 116.90 | 116.60 | 116.57 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 116.00 | 116.55 | 116.59 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 118.46 | 116.93 | 116.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 119.21 | 117.39 | 116.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 119.83 | 120.46 | 119.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 119.83 | 120.46 | 119.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 119.83 | 120.46 | 119.51 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 14:15:00 | 119.67 | 119.75 | 119.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 116.58 | 119.06 | 119.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 118.81 | 117.86 | 118.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 118.81 | 117.86 | 118.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 118.81 | 117.86 | 118.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 119.48 | 117.86 | 118.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 119.65 | 118.21 | 118.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 119.96 | 118.21 | 118.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 119.70 | 118.93 | 118.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 120.90 | 119.33 | 119.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 120.21 | 120.66 | 120.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 120.21 | 120.66 | 120.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 120.21 | 120.66 | 120.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 119.87 | 120.66 | 120.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 120.05 | 120.54 | 120.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 120.55 | 120.54 | 120.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:45:00 | 120.40 | 120.50 | 120.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 120.68 | 120.43 | 120.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 119.43 | 120.17 | 120.14 | SL hit (close<static) qty=1.00 sl=119.53 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 119.81 | 120.10 | 120.11 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 120.82 | 120.14 | 120.09 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 119.18 | 119.94 | 120.01 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 120.04 | 119.79 | 119.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 121.40 | 120.23 | 120.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 122.10 | 122.34 | 121.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 122.10 | 122.34 | 121.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 120.87 | 121.96 | 121.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 120.60 | 121.96 | 121.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 121.39 | 121.85 | 121.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 121.17 | 121.85 | 121.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 122.20 | 121.92 | 121.61 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 120.78 | 121.55 | 121.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 120.50 | 121.21 | 121.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 115.24 | 115.10 | 116.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 115.68 | 115.10 | 116.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 115.55 | 115.19 | 116.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 114.68 | 115.15 | 115.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 114.38 | 115.11 | 115.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 114.68 | 114.87 | 115.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 114.55 | 114.88 | 115.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 114.95 | 114.50 | 115.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 114.98 | 114.50 | 115.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 115.34 | 114.66 | 115.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 115.54 | 114.66 | 115.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 116.33 | 115.00 | 115.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 116.33 | 115.00 | 115.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 108.95 | 110.57 | 112.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 108.66 | 110.57 | 112.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 108.95 | 110.57 | 112.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 108.82 | 110.57 | 112.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 110.96 | 109.45 | 110.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 110.96 | 109.45 | 110.74 | SL hit (close>ema200) qty=0.50 sl=109.45 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 111.41 | 110.89 | 110.89 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 109.79 | 110.82 | 110.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 109.49 | 110.43 | 110.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 109.15 | 109.15 | 109.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:30:00 | 109.48 | 109.15 | 109.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 109.62 | 109.03 | 109.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 109.62 | 109.03 | 109.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 109.80 | 109.18 | 109.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 110.17 | 109.18 | 109.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 109.85 | 109.21 | 109.46 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 111.50 | 109.89 | 109.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 112.44 | 111.11 | 110.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 112.16 | 112.54 | 111.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:00:00 | 112.16 | 112.54 | 111.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 111.92 | 112.42 | 111.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 111.79 | 112.42 | 111.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 111.81 | 112.30 | 111.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 111.81 | 112.30 | 111.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 112.73 | 112.38 | 111.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 112.95 | 112.49 | 111.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:30:00 | 113.20 | 112.47 | 111.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 113.61 | 112.70 | 112.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 113.60 | 113.12 | 112.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 110.58 | 112.63 | 112.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 110.58 | 112.63 | 112.35 | SL hit (close<static) qty=1.00 sl=111.68 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 111.24 | 112.06 | 112.12 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 113.96 | 112.35 | 112.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 114.63 | 112.80 | 112.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 120.86 | 121.63 | 119.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 120.86 | 121.63 | 119.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 116.42 | 120.16 | 119.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 116.42 | 120.16 | 119.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 116.92 | 119.51 | 119.55 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 121.45 | 119.14 | 119.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 122.79 | 120.23 | 119.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 14:15:00 | 130.43 | 131.19 | 128.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:45:00 | 130.06 | 131.19 | 128.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 131.34 | 132.36 | 131.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 131.34 | 132.36 | 131.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 130.99 | 132.09 | 131.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 131.89 | 132.09 | 131.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 130.87 | 131.85 | 131.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 13:15:00 | 132.84 | 131.98 | 131.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 15:00:00 | 132.37 | 132.10 | 131.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 132.33 | 132.10 | 131.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 133.22 | 132.03 | 131.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 131.94 | 132.11 | 131.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 131.94 | 132.11 | 131.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 133.30 | 132.48 | 132.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:30:00 | 133.69 | 132.69 | 132.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:30:00 | 133.80 | 132.98 | 132.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 131.80 | 132.91 | 132.76 | SL hit (close<static) qty=1.00 sl=131.89 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 131.42 | 132.62 | 132.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 130.92 | 132.28 | 132.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 131.54 | 131.46 | 132.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 131.54 | 131.46 | 132.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 131.54 | 131.46 | 132.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 131.80 | 131.46 | 132.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 131.70 | 131.50 | 131.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:30:00 | 130.75 | 131.25 | 131.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 131.78 | 130.92 | 130.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 14:15:00 | 131.78 | 130.92 | 130.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 132.30 | 131.20 | 130.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 134.10 | 134.27 | 133.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 134.10 | 134.27 | 133.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 133.02 | 134.19 | 133.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 133.02 | 134.19 | 133.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 133.10 | 133.98 | 133.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 130.55 | 133.98 | 133.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 130.68 | 133.32 | 133.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 124.00 | 128.36 | 130.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 125.95 | 124.62 | 127.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 125.95 | 124.62 | 127.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 125.95 | 124.62 | 127.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 126.58 | 124.62 | 127.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 126.03 | 125.08 | 126.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 126.03 | 125.08 | 126.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 126.70 | 125.40 | 126.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 125.80 | 125.40 | 126.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 113.22 | 122.57 | 124.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 123.65 | 121.87 | 121.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 123.90 | 122.28 | 122.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 122.30 | 122.32 | 122.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 122.30 | 122.32 | 122.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 122.30 | 122.32 | 122.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 122.30 | 122.32 | 122.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 121.73 | 122.20 | 122.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 121.73 | 122.20 | 122.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 121.01 | 121.96 | 121.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 117.98 | 121.05 | 121.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 14:15:00 | 120.30 | 120.30 | 120.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 14:30:00 | 120.04 | 120.30 | 120.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 115.44 | 113.64 | 115.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 115.44 | 113.64 | 115.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 115.81 | 114.07 | 115.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 115.81 | 114.07 | 115.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 115.13 | 114.28 | 115.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 113.88 | 114.28 | 115.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 116.01 | 114.94 | 115.38 | SL hit (close>static) qty=1.00 sl=115.90 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 117.04 | 115.78 | 115.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 117.71 | 116.17 | 115.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 113.85 | 116.50 | 116.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 113.85 | 116.50 | 116.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 113.85 | 116.50 | 116.28 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 112.60 | 115.72 | 115.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 112.18 | 114.52 | 115.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 113.30 | 113.25 | 114.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:45:00 | 113.21 | 113.25 | 114.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 106.94 | 107.91 | 110.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:45:00 | 106.75 | 107.69 | 109.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 111.48 | 109.05 | 109.93 | SL hit (close>static) qty=1.00 sl=110.90 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 114.50 | 110.78 | 110.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 115.35 | 111.70 | 111.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 109.85 | 112.47 | 111.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 109.85 | 112.47 | 111.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 109.85 | 112.47 | 111.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 109.69 | 112.47 | 111.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 110.00 | 111.98 | 111.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:30:00 | 110.27 | 111.60 | 111.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 110.45 | 111.60 | 111.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 12:15:00 | 110.71 | 111.42 | 111.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 110.71 | 111.42 | 111.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 109.26 | 110.85 | 111.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 108.99 | 107.17 | 108.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 108.99 | 107.17 | 108.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 108.99 | 107.17 | 108.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 107.89 | 107.34 | 108.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 107.75 | 108.09 | 108.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 107.61 | 107.93 | 108.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 107.39 | 106.76 | 107.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 107.43 | 106.90 | 107.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:30:00 | 108.55 | 106.90 | 107.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 108.32 | 107.54 | 107.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 108.32 | 107.54 | 107.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 108.44 | 108.01 | 107.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 116.82 | 116.91 | 114.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 116.82 | 116.91 | 114.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 118.71 | 119.98 | 117.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 119.77 | 119.85 | 118.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 120.12 | 119.90 | 118.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 122.24 | 119.59 | 118.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 131.75 | 127.89 | 126.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 125.72 | 127.21 | 127.22 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 127.66 | 126.88 | 126.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 128.30 | 127.29 | 127.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 127.27 | 127.29 | 127.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:00:00 | 127.27 | 127.29 | 127.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 126.57 | 127.17 | 127.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 126.57 | 127.17 | 127.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 126.06 | 126.94 | 126.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 125.87 | 126.73 | 126.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 122.50 | 121.76 | 123.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 122.50 | 121.76 | 123.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 122.50 | 121.76 | 123.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 121.09 | 121.66 | 122.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 121.48 | 121.59 | 122.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 123.23 | 121.81 | 121.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 123.23 | 121.81 | 121.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 124.12 | 122.47 | 122.06 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:30:00 | 95.51 | 2025-05-14 11:15:00 | 94.55 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-14 10:00:00 | 95.57 | 2025-05-14 11:15:00 | 94.55 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-21 09:30:00 | 98.16 | 2025-05-27 14:15:00 | 99.04 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2025-05-21 10:00:00 | 98.49 | 2025-05-27 14:15:00 | 99.04 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-05-21 12:15:00 | 98.05 | 2025-05-27 14:15:00 | 99.04 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-05-22 10:45:00 | 98.37 | 2025-05-27 14:15:00 | 99.04 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2025-05-26 09:15:00 | 99.95 | 2025-05-27 14:15:00 | 99.04 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-05-30 09:15:00 | 105.43 | 2025-06-03 11:15:00 | 100.57 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2025-05-30 11:00:00 | 102.90 | 2025-06-03 11:15:00 | 100.57 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-05-30 12:15:00 | 102.67 | 2025-06-03 11:15:00 | 100.57 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-05-30 14:30:00 | 102.60 | 2025-06-03 11:15:00 | 100.57 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-30 11:15:00 | 102.93 | 2025-07-01 11:15:00 | 101.65 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-30 14:15:00 | 102.97 | 2025-07-01 11:15:00 | 101.65 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-01 09:45:00 | 102.89 | 2025-07-01 11:15:00 | 101.65 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-10 09:15:00 | 103.57 | 2025-07-10 09:15:00 | 102.58 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-11 11:15:00 | 102.07 | 2025-07-15 09:15:00 | 103.03 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-14 13:30:00 | 101.79 | 2025-07-15 09:15:00 | 103.03 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-16 12:00:00 | 103.40 | 2025-07-17 12:15:00 | 102.67 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-16 12:45:00 | 103.40 | 2025-07-17 12:15:00 | 102.67 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-28 09:15:00 | 101.94 | 2025-07-30 09:15:00 | 100.52 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-28 10:15:00 | 101.33 | 2025-07-30 09:15:00 | 100.52 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-28 13:45:00 | 101.35 | 2025-07-30 09:15:00 | 100.52 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-29 11:15:00 | 101.43 | 2025-07-30 09:15:00 | 100.52 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest1 | 2025-08-08 09:30:00 | 93.72 | 2025-08-13 13:15:00 | 94.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-08-13 11:45:00 | 90.88 | 2025-08-13 13:15:00 | 94.30 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-08-13 12:15:00 | 90.99 | 2025-08-13 13:15:00 | 94.30 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest1 | 2025-08-21 13:45:00 | 98.14 | 2025-08-22 09:15:00 | 96.63 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest1 | 2025-08-21 14:45:00 | 98.05 | 2025-08-22 09:15:00 | 96.63 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-01 10:15:00 | 94.15 | 2025-09-01 12:15:00 | 94.73 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-18 12:15:00 | 110.05 | 2025-09-24 11:15:00 | 109.21 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-09-19 09:15:00 | 110.00 | 2025-09-24 13:15:00 | 108.83 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-09-22 13:30:00 | 110.01 | 2025-09-24 13:15:00 | 108.83 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-22 14:45:00 | 110.03 | 2025-09-24 13:15:00 | 108.83 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-09-23 10:45:00 | 112.04 | 2025-09-24 13:15:00 | 108.83 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-09-29 12:30:00 | 105.49 | 2025-10-03 14:15:00 | 106.72 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-30 11:15:00 | 105.55 | 2025-10-03 14:15:00 | 106.72 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-30 15:00:00 | 105.57 | 2025-10-03 14:15:00 | 106.72 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-01 10:30:00 | 105.59 | 2025-10-03 14:15:00 | 106.72 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-10 09:15:00 | 103.05 | 2025-10-10 09:15:00 | 104.22 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-16 09:15:00 | 105.82 | 2025-10-17 14:15:00 | 104.73 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-29 10:30:00 | 107.66 | 2025-10-31 10:15:00 | 106.63 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-29 13:00:00 | 107.19 | 2025-10-31 10:15:00 | 106.63 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-10-29 14:15:00 | 107.20 | 2025-10-31 10:15:00 | 106.63 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-10-29 15:00:00 | 107.45 | 2025-10-31 10:15:00 | 106.63 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-04 11:00:00 | 104.99 | 2025-11-11 14:15:00 | 105.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-11-04 13:15:00 | 104.88 | 2025-11-11 14:15:00 | 105.30 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-11-13 15:15:00 | 111.60 | 2025-11-21 14:15:00 | 109.76 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-14 10:45:00 | 111.44 | 2025-11-21 14:15:00 | 109.76 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-14 15:00:00 | 110.04 | 2025-11-21 14:15:00 | 109.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-17 09:15:00 | 110.24 | 2025-11-21 14:15:00 | 109.76 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-11-18 13:45:00 | 109.70 | 2025-11-21 14:15:00 | 109.76 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-19 09:15:00 | 109.80 | 2025-11-21 14:15:00 | 109.76 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-25 15:15:00 | 110.08 | 2025-11-26 09:15:00 | 111.61 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-12-02 11:15:00 | 118.66 | 2025-12-05 15:15:00 | 116.78 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-12-02 12:00:00 | 118.80 | 2025-12-05 15:15:00 | 116.78 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-02 14:15:00 | 118.65 | 2025-12-05 15:15:00 | 116.78 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-09 14:30:00 | 116.64 | 2025-12-09 15:15:00 | 116.90 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-12-23 11:15:00 | 120.55 | 2025-12-24 10:15:00 | 119.43 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-23 11:45:00 | 120.40 | 2025-12-24 10:15:00 | 119.43 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-24 09:15:00 | 120.68 | 2025-12-24 10:15:00 | 119.43 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-01-13 14:00:00 | 114.68 | 2026-01-20 13:15:00 | 108.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 114.38 | 2026-01-20 13:15:00 | 108.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 114.68 | 2026-01-20 13:15:00 | 108.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 114.55 | 2026-01-20 13:15:00 | 108.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 114.68 | 2026-01-21 12:15:00 | 110.96 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2026-01-13 15:15:00 | 114.38 | 2026-01-21 12:15:00 | 110.96 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2026-01-14 09:30:00 | 114.68 | 2026-01-21 12:15:00 | 110.96 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2026-01-14 13:30:00 | 114.55 | 2026-01-21 12:15:00 | 110.96 | STOP_HIT | 0.50 | 3.13% |
| BUY | retest2 | 2026-01-30 14:45:00 | 112.95 | 2026-02-01 15:15:00 | 110.58 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-01 09:30:00 | 113.20 | 2026-02-01 15:15:00 | 110.58 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-02-01 11:00:00 | 113.61 | 2026-02-01 15:15:00 | 110.58 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-02-01 12:30:00 | 113.60 | 2026-02-01 15:15:00 | 110.58 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-02-16 13:15:00 | 132.84 | 2026-02-19 11:15:00 | 131.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-02-16 15:00:00 | 132.37 | 2026-02-19 11:15:00 | 131.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-02-17 10:15:00 | 132.33 | 2026-02-19 12:15:00 | 131.42 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-02-17 12:15:00 | 133.22 | 2026-02-19 12:15:00 | 131.42 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-18 10:30:00 | 133.69 | 2026-02-19 12:15:00 | 131.42 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-02-18 11:30:00 | 133.80 | 2026-02-19 12:15:00 | 131.42 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-02-20 13:30:00 | 130.75 | 2026-02-24 14:15:00 | 131.78 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-03-06 09:15:00 | 125.80 | 2026-03-09 09:15:00 | 113.22 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-17 12:15:00 | 113.88 | 2026-03-17 14:15:00 | 116.01 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-24 10:45:00 | 106.75 | 2026-03-24 13:15:00 | 111.48 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2026-03-27 11:30:00 | 110.27 | 2026-03-27 12:15:00 | 110.71 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2026-03-27 12:15:00 | 110.45 | 2026-03-27 12:15:00 | 110.71 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-04-01 11:15:00 | 107.89 | 2026-04-06 13:15:00 | 108.32 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-04-01 13:30:00 | 107.75 | 2026-04-06 13:15:00 | 108.32 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-04-01 14:30:00 | 107.61 | 2026-04-06 13:15:00 | 108.32 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-06 09:45:00 | 107.39 | 2026-04-06 13:15:00 | 108.32 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-04-13 10:45:00 | 119.77 | 2026-04-22 12:15:00 | 131.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:30:00 | 120.12 | 2026-04-22 12:15:00 | 132.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 122.24 | 2026-04-24 10:15:00 | 125.72 | STOP_HIT | 1.00 | 2.85% |
| SELL | retest2 | 2026-05-04 12:00:00 | 121.09 | 2026-05-06 11:15:00 | 123.23 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-05-04 12:30:00 | 121.48 | 2026-05-06 11:15:00 | 123.23 | STOP_HIT | 1.00 | -1.44% |
