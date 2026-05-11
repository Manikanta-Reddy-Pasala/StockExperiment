# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 522.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 214 |
| ALERT1 | 145 |
| ALERT2 | 140 |
| ALERT2_SKIP | 79 |
| ALERT3 | 374 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 163 |
| PARTIAL | 15 |
| TARGET_HIT | 8 |
| STOP_HIT | 159 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 180 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 114
- **Target hits / Stop hits / Partials:** 8 / 157 / 15
- **Avg / median % per leg:** 0.46% / -0.59%
- **Sum % (uncompounded):** 83.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 23 | 34.3% | 8 | 59 | 0 | 0.83% | 55.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.54% | -1.5% |
| BUY @ 3rd Alert (retest2) | 66 | 23 | 34.8% | 8 | 58 | 0 | 0.87% | 57.4% |
| SELL (all) | 113 | 43 | 38.1% | 0 | 98 | 15 | 0.24% | 27.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 6.40% | 12.8% |
| SELL @ 3rd Alert (retest2) | 111 | 41 | 36.9% | 0 | 97 | 14 | 0.13% | 14.8% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.75% | 11.3% |
| retest2 (combined) | 177 | 64 | 36.2% | 8 | 155 | 14 | 0.41% | 72.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 11:15:00 | 387.05 | 384.49 | 384.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 390.60 | 387.53 | 386.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 10:15:00 | 390.25 | 391.23 | 389.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 10:15:00 | 390.25 | 391.23 | 389.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 390.25 | 391.23 | 389.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:00:00 | 390.25 | 391.23 | 389.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 389.45 | 390.88 | 389.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:45:00 | 389.35 | 390.88 | 389.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 389.65 | 390.41 | 389.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 09:45:00 | 391.75 | 390.08 | 389.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 12:15:00 | 390.55 | 390.37 | 389.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 12:45:00 | 391.25 | 390.81 | 389.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 15:00:00 | 391.40 | 391.28 | 390.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 390.00 | 391.02 | 390.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:15:00 | 388.35 | 391.02 | 390.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 387.50 | 390.32 | 389.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-05-25 09:15:00 | 387.50 | 390.32 | 389.98 | SL hit (close<static) qty=1.00 sl=388.80 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 386.75 | 389.35 | 389.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 12:15:00 | 385.00 | 388.48 | 389.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 392.50 | 389.22 | 389.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 392.50 | 389.22 | 389.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 392.50 | 389.22 | 389.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 392.50 | 389.22 | 389.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-05-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 15:15:00 | 391.10 | 389.60 | 389.54 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 12:15:00 | 387.50 | 389.34 | 389.49 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 15:15:00 | 391.00 | 389.73 | 389.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 393.40 | 390.47 | 389.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 391.75 | 392.80 | 391.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 391.75 | 392.80 | 391.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 391.75 | 392.80 | 391.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:30:00 | 391.80 | 392.80 | 391.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 390.85 | 392.41 | 391.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 390.85 | 392.41 | 391.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 391.55 | 392.24 | 391.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:45:00 | 391.40 | 392.24 | 391.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 393.40 | 392.47 | 391.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 09:30:00 | 393.80 | 392.40 | 391.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 11:15:00 | 391.50 | 392.20 | 391.93 | SL hit (close<static) qty=1.00 sl=391.55 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-01 11:15:00 | 390.55 | 391.74 | 391.82 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 392.70 | 391.81 | 391.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 11:15:00 | 394.10 | 392.27 | 391.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 12:15:00 | 392.00 | 392.21 | 391.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 12:15:00 | 392.00 | 392.21 | 391.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 392.00 | 392.21 | 391.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:00:00 | 392.00 | 392.21 | 391.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 392.20 | 392.21 | 392.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 14:15:00 | 391.60 | 392.21 | 392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 392.35 | 392.24 | 392.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 14:30:00 | 392.30 | 392.24 | 392.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 392.00 | 392.19 | 392.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:30:00 | 390.60 | 392.18 | 392.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 393.90 | 392.53 | 392.21 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 14:15:00 | 391.15 | 392.10 | 392.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 15:15:00 | 390.50 | 391.78 | 391.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 14:15:00 | 392.60 | 390.48 | 390.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 14:15:00 | 392.60 | 390.48 | 390.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 392.60 | 390.48 | 390.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 15:00:00 | 392.60 | 390.48 | 390.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 392.90 | 390.96 | 391.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 394.40 | 390.96 | 391.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 400.75 | 392.92 | 392.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 403.00 | 394.94 | 393.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 400.50 | 401.57 | 398.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 12:00:00 | 400.50 | 401.57 | 398.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 398.30 | 400.75 | 398.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:30:00 | 399.25 | 400.75 | 398.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 395.30 | 399.66 | 398.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 395.30 | 399.66 | 398.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 396.00 | 398.93 | 398.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 411.50 | 398.93 | 398.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 15:15:00 | 401.20 | 402.18 | 402.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 15:15:00 | 401.20 | 402.18 | 402.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 10:15:00 | 398.25 | 401.16 | 401.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 397.25 | 392.32 | 394.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 397.25 | 392.32 | 394.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 397.25 | 392.32 | 394.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:00:00 | 397.25 | 392.32 | 394.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 396.10 | 393.07 | 394.63 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 13:15:00 | 398.90 | 395.58 | 395.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 400.15 | 396.77 | 396.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 396.65 | 397.75 | 396.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 12:15:00 | 396.65 | 397.75 | 396.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 12:15:00 | 396.65 | 397.75 | 396.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:00:00 | 396.65 | 397.75 | 396.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 395.35 | 397.27 | 396.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:30:00 | 395.75 | 397.27 | 396.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 394.65 | 396.75 | 396.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 14:30:00 | 395.80 | 396.75 | 396.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 391.60 | 395.50 | 395.95 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 397.95 | 395.88 | 395.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 12:15:00 | 401.20 | 398.28 | 397.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 396.95 | 404.39 | 402.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 396.95 | 404.39 | 402.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 396.95 | 404.39 | 402.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:00:00 | 396.95 | 404.39 | 402.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 397.10 | 402.93 | 401.88 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 13:15:00 | 395.00 | 400.01 | 400.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 14:15:00 | 390.60 | 398.13 | 399.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 385.00 | 382.83 | 385.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 385.00 | 382.83 | 385.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 385.00 | 382.83 | 385.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:30:00 | 384.75 | 382.83 | 385.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 386.35 | 383.53 | 385.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:45:00 | 387.10 | 383.53 | 385.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 386.50 | 384.13 | 385.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 11:30:00 | 386.90 | 384.13 | 385.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 384.85 | 384.82 | 385.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 14:30:00 | 385.40 | 384.82 | 385.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 384.65 | 384.79 | 385.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 12:30:00 | 383.40 | 384.68 | 385.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 15:15:00 | 383.55 | 384.60 | 385.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 09:45:00 | 382.30 | 384.21 | 384.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 12:45:00 | 382.35 | 383.94 | 384.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 386.95 | 383.80 | 384.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:45:00 | 386.85 | 383.80 | 384.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 383.95 | 383.83 | 384.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 12:30:00 | 382.90 | 383.79 | 384.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 10:45:00 | 383.50 | 383.69 | 383.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 11:15:00 | 383.35 | 383.69 | 383.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 13:15:00 | 382.85 | 383.66 | 383.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 382.20 | 383.37 | 383.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-07 09:15:00 | 386.65 | 384.19 | 384.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 386.65 | 384.19 | 384.00 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 382.30 | 383.75 | 383.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 381.15 | 383.23 | 383.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 383.05 | 382.77 | 383.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 383.05 | 382.77 | 383.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 383.05 | 382.77 | 383.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:15:00 | 384.65 | 382.77 | 383.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 385.00 | 383.21 | 383.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:45:00 | 385.75 | 383.21 | 383.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 382.30 | 383.03 | 383.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-10 15:15:00 | 382.00 | 382.97 | 383.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 11:00:00 | 382.00 | 382.68 | 382.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:00:00 | 381.90 | 382.53 | 382.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 14:15:00 | 381.80 | 382.55 | 382.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 382.45 | 382.53 | 382.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 382.45 | 382.53 | 382.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 382.05 | 382.43 | 382.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 385.65 | 382.43 | 382.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-12 09:15:00 | 390.45 | 384.04 | 383.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 390.45 | 384.04 | 383.44 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 12:15:00 | 382.45 | 383.79 | 383.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 375.65 | 382.16 | 383.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 12:15:00 | 379.25 | 378.95 | 380.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 13:00:00 | 379.25 | 378.95 | 380.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 380.05 | 379.14 | 380.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:45:00 | 380.65 | 379.14 | 380.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 380.50 | 379.41 | 380.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 381.35 | 379.41 | 380.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 384.95 | 380.52 | 380.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:45:00 | 385.30 | 380.52 | 380.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 383.25 | 381.07 | 381.09 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 387.70 | 382.39 | 381.69 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 380.75 | 382.42 | 382.55 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 389.35 | 382.31 | 382.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 10:15:00 | 392.80 | 384.41 | 383.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 09:15:00 | 385.60 | 387.87 | 385.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 385.60 | 387.87 | 385.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 385.60 | 387.87 | 385.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:30:00 | 384.15 | 387.87 | 385.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 386.05 | 387.51 | 385.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 11:00:00 | 386.05 | 387.51 | 385.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 387.00 | 387.41 | 385.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 11:30:00 | 385.65 | 387.41 | 385.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 12:15:00 | 385.55 | 387.03 | 385.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 12:30:00 | 385.55 | 387.03 | 385.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 13:15:00 | 390.50 | 387.73 | 386.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 14:15:00 | 392.20 | 387.73 | 386.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-28 09:15:00 | 393.20 | 397.96 | 398.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 09:15:00 | 393.20 | 397.96 | 398.53 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 10:15:00 | 401.10 | 397.50 | 397.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 15:15:00 | 408.45 | 404.69 | 402.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 400.80 | 403.99 | 402.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 400.80 | 403.99 | 402.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 400.80 | 403.99 | 402.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 400.80 | 403.99 | 402.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 399.20 | 403.03 | 402.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:30:00 | 399.10 | 403.03 | 402.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 395.25 | 400.58 | 401.04 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 10:15:00 | 406.00 | 402.03 | 401.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 11:15:00 | 407.30 | 403.09 | 402.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 10:15:00 | 409.05 | 410.61 | 408.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-07 10:45:00 | 410.00 | 410.61 | 408.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 408.80 | 410.33 | 408.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:00:00 | 408.80 | 410.33 | 408.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 408.00 | 409.87 | 408.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:45:00 | 409.30 | 409.87 | 408.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 404.35 | 408.76 | 407.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 15:00:00 | 404.35 | 408.76 | 407.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 403.60 | 407.73 | 407.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 09:15:00 | 396.75 | 407.73 | 407.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 09:15:00 | 400.85 | 406.35 | 406.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 10:15:00 | 392.95 | 403.67 | 405.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 14:15:00 | 395.00 | 394.90 | 398.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-09 14:45:00 | 396.15 | 394.90 | 398.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 395.35 | 394.91 | 397.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 09:30:00 | 397.50 | 394.91 | 397.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 396.15 | 395.14 | 397.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 12:45:00 | 396.20 | 395.14 | 397.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 395.10 | 395.12 | 396.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:45:00 | 395.70 | 395.12 | 396.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 393.70 | 394.78 | 396.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 11:00:00 | 392.30 | 394.28 | 395.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 12:45:00 | 392.00 | 393.52 | 395.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 389.85 | 391.27 | 392.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 13:15:00 | 394.20 | 392.84 | 392.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 13:15:00 | 394.20 | 392.84 | 392.83 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 12:15:00 | 392.45 | 392.92 | 392.92 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 13:15:00 | 393.20 | 392.98 | 392.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 14:15:00 | 394.00 | 393.18 | 393.04 | Break + close above crossover candle high |

### Cycle 30 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 386.75 | 392.11 | 392.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 384.40 | 389.16 | 390.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 14:15:00 | 389.30 | 385.25 | 387.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 14:15:00 | 389.30 | 385.25 | 387.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 389.30 | 385.25 | 387.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 389.30 | 385.25 | 387.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 389.00 | 386.00 | 387.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 395.50 | 386.00 | 387.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 395.70 | 389.14 | 388.62 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 387.80 | 391.58 | 391.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 12:15:00 | 387.05 | 390.68 | 391.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 390.75 | 388.37 | 389.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 390.75 | 388.37 | 389.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 390.75 | 388.37 | 389.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:45:00 | 389.40 | 388.37 | 389.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 391.65 | 389.02 | 390.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 391.65 | 389.02 | 390.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 391.50 | 389.52 | 390.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:30:00 | 392.50 | 389.52 | 390.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 392.20 | 390.07 | 390.25 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 391.80 | 390.41 | 390.39 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 388.70 | 390.28 | 390.34 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 09:15:00 | 394.30 | 390.43 | 390.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 10:15:00 | 397.20 | 391.79 | 390.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 393.45 | 394.22 | 392.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 393.45 | 394.22 | 392.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 393.45 | 394.22 | 392.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:30:00 | 394.20 | 394.22 | 392.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 390.75 | 393.53 | 392.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:00:00 | 390.75 | 393.53 | 392.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 390.75 | 392.97 | 392.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 12:00:00 | 390.75 | 392.97 | 392.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 389.15 | 391.70 | 391.95 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 399.20 | 393.32 | 392.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 401.40 | 396.38 | 394.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 09:15:00 | 402.90 | 403.00 | 399.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 10:00:00 | 402.90 | 403.00 | 399.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 407.00 | 403.80 | 400.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 411.80 | 403.92 | 401.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 14:00:00 | 407.30 | 410.58 | 410.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 14:45:00 | 407.30 | 409.86 | 409.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 15:15:00 | 406.90 | 409.27 | 409.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 15:15:00 | 406.90 | 409.27 | 409.48 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 09:15:00 | 418.95 | 411.20 | 410.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 10:15:00 | 424.45 | 413.85 | 411.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 404.45 | 415.56 | 414.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 404.45 | 415.56 | 414.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 404.45 | 415.56 | 414.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 407.05 | 415.56 | 414.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 406.90 | 413.83 | 413.44 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 404.95 | 412.05 | 412.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 399.30 | 407.21 | 410.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 405.85 | 405.70 | 408.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 10:30:00 | 402.10 | 405.70 | 408.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 409.90 | 406.54 | 408.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 409.90 | 406.54 | 408.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 410.10 | 407.25 | 408.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:00:00 | 410.10 | 407.25 | 408.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 415.35 | 408.87 | 409.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 14:00:00 | 415.35 | 408.87 | 409.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 416.00 | 410.30 | 410.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 437.75 | 416.41 | 412.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 11:15:00 | 427.70 | 431.83 | 425.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 11:15:00 | 427.70 | 431.83 | 425.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 11:15:00 | 427.70 | 431.83 | 425.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 11:30:00 | 426.65 | 431.83 | 425.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 438.85 | 438.00 | 434.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 12:30:00 | 442.30 | 439.41 | 435.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 12:15:00 | 430.50 | 435.84 | 435.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 430.50 | 435.84 | 435.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 420.85 | 431.85 | 433.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 434.65 | 420.37 | 424.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 434.65 | 420.37 | 424.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 434.65 | 420.37 | 424.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:00:00 | 434.65 | 420.37 | 424.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 439.90 | 424.28 | 425.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 11:00:00 | 439.90 | 424.28 | 425.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 439.85 | 427.39 | 426.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 444.00 | 436.75 | 433.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 442.80 | 445.66 | 441.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 442.80 | 445.66 | 441.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 442.80 | 445.66 | 441.13 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 15:15:00 | 437.00 | 439.73 | 439.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 09:15:00 | 432.65 | 438.32 | 439.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 09:15:00 | 434.00 | 433.05 | 435.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 09:15:00 | 434.00 | 433.05 | 435.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 434.00 | 433.05 | 435.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:45:00 | 434.90 | 433.05 | 435.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 427.00 | 427.38 | 430.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 423.45 | 427.38 | 430.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 14:45:00 | 424.55 | 425.41 | 428.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 15:15:00 | 430.55 | 429.27 | 429.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 15:15:00 | 430.55 | 429.27 | 429.14 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 427.50 | 428.93 | 429.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 424.05 | 427.96 | 428.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 427.25 | 426.61 | 427.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 427.25 | 426.61 | 427.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 427.25 | 426.61 | 427.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:30:00 | 428.90 | 426.61 | 427.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 427.10 | 426.71 | 427.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 13:30:00 | 425.00 | 426.28 | 427.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 10:15:00 | 425.00 | 425.73 | 426.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 11:15:00 | 424.75 | 425.78 | 426.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 10:15:00 | 424.60 | 419.96 | 420.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 425.05 | 420.98 | 421.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-16 11:15:00 | 424.20 | 421.62 | 421.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 11:15:00 | 424.20 | 421.62 | 421.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 09:15:00 | 426.20 | 423.70 | 423.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 421.35 | 423.23 | 422.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 421.35 | 423.23 | 422.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 421.35 | 423.23 | 422.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 421.35 | 423.23 | 422.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 422.05 | 422.99 | 422.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:45:00 | 420.55 | 422.99 | 422.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 425.45 | 423.48 | 423.03 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-10-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 10:15:00 | 420.85 | 422.74 | 422.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 418.50 | 421.53 | 422.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 13:15:00 | 424.00 | 422.02 | 422.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 13:15:00 | 424.00 | 422.02 | 422.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 13:15:00 | 424.00 | 422.02 | 422.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 14:00:00 | 424.00 | 422.02 | 422.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 423.40 | 422.30 | 422.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 14:30:00 | 424.05 | 422.30 | 422.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 422.70 | 422.38 | 422.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:15:00 | 418.75 | 422.38 | 422.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 411.45 | 420.19 | 421.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 11:00:00 | 409.05 | 417.97 | 420.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:15:00 | 409.80 | 410.57 | 413.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 14:15:00 | 413.50 | 408.26 | 407.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 413.50 | 408.26 | 407.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 09:15:00 | 415.60 | 413.60 | 412.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 13:15:00 | 422.55 | 422.67 | 419.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 14:00:00 | 422.55 | 422.67 | 419.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 428.80 | 430.79 | 428.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:45:00 | 428.50 | 430.79 | 428.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 427.25 | 430.08 | 428.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:00:00 | 427.25 | 430.08 | 428.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 430.00 | 430.06 | 428.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 15:15:00 | 431.50 | 430.06 | 428.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 12:15:00 | 427.75 | 432.85 | 433.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 12:15:00 | 427.75 | 432.85 | 433.15 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 435.00 | 432.39 | 432.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 13:15:00 | 436.45 | 433.20 | 432.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 09:15:00 | 433.20 | 434.11 | 433.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 433.20 | 434.11 | 433.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 433.20 | 434.11 | 433.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:00:00 | 433.20 | 434.11 | 433.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 433.65 | 434.02 | 433.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:30:00 | 432.25 | 434.02 | 433.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 11:15:00 | 432.20 | 433.66 | 433.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 12:00:00 | 432.20 | 433.66 | 433.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 432.35 | 433.40 | 433.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 13:00:00 | 432.35 | 433.40 | 433.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 434.65 | 433.86 | 433.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:30:00 | 434.70 | 433.86 | 433.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 435.30 | 434.33 | 433.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 13:00:00 | 437.15 | 435.00 | 434.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 13:45:00 | 437.20 | 435.61 | 434.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-30 09:15:00 | 480.87 | 471.98 | 468.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 14:15:00 | 468.30 | 471.04 | 471.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 10:15:00 | 464.85 | 469.39 | 470.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 11:15:00 | 385.55 | 382.99 | 388.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 11:15:00 | 385.55 | 382.99 | 388.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 385.55 | 382.99 | 388.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:30:00 | 387.40 | 382.99 | 388.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 386.70 | 383.74 | 387.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 14:45:00 | 386.20 | 383.74 | 387.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 388.40 | 385.12 | 387.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 12:30:00 | 386.80 | 386.19 | 387.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 09:15:00 | 404.60 | 389.28 | 388.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 09:15:00 | 404.60 | 389.28 | 388.41 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 389.20 | 394.79 | 395.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 385.55 | 392.94 | 394.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 390.65 | 387.42 | 389.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 390.65 | 387.42 | 389.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 390.65 | 387.42 | 389.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 390.65 | 387.42 | 389.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 391.15 | 388.17 | 389.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 391.15 | 388.17 | 389.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 389.55 | 388.44 | 389.71 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 392.20 | 390.07 | 389.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 11:15:00 | 393.40 | 390.73 | 390.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 414.30 | 414.55 | 409.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 09:15:00 | 405.50 | 412.64 | 409.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 405.50 | 412.64 | 409.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 405.50 | 412.64 | 409.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 404.25 | 410.96 | 408.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 403.50 | 410.96 | 408.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 14:15:00 | 405.25 | 407.52 | 407.72 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 12:15:00 | 407.85 | 407.22 | 407.13 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 10:15:00 | 405.45 | 406.89 | 407.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 14:15:00 | 403.95 | 405.68 | 406.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 11:15:00 | 399.15 | 398.34 | 400.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 12:00:00 | 399.15 | 398.34 | 400.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 403.25 | 399.32 | 401.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 12:45:00 | 403.45 | 399.32 | 401.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 399.80 | 399.42 | 400.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-09 14:15:00 | 398.00 | 399.42 | 400.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 09:15:00 | 392.60 | 399.09 | 400.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 378.10 | 386.74 | 388.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-18 11:15:00 | 387.35 | 386.82 | 388.54 | SL hit (close>ema200) qty=0.50 sl=386.82 alert=retest2 |

### Cycle 59 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 384.90 | 382.48 | 382.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 389.55 | 383.90 | 383.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 11:15:00 | 383.20 | 383.98 | 383.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 11:15:00 | 383.20 | 383.98 | 383.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 383.20 | 383.98 | 383.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:45:00 | 383.65 | 383.98 | 383.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 384.90 | 384.17 | 383.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 14:00:00 | 387.35 | 384.80 | 383.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 15:00:00 | 387.95 | 385.43 | 384.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 11:30:00 | 386.80 | 387.19 | 385.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 13:15:00 | 389.80 | 392.18 | 392.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 389.80 | 392.18 | 392.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 14:15:00 | 388.65 | 391.47 | 391.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 394.15 | 391.76 | 392.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 394.15 | 391.76 | 392.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 394.15 | 391.76 | 392.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 394.15 | 391.76 | 392.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 396.30 | 392.67 | 392.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 13:15:00 | 403.10 | 396.32 | 394.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 13:15:00 | 397.65 | 397.86 | 396.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-06 13:45:00 | 397.40 | 397.86 | 396.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 397.00 | 397.51 | 396.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 09:15:00 | 399.70 | 397.51 | 396.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 403.05 | 397.45 | 396.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 09:15:00 | 395.60 | 400.14 | 398.89 | SL hit (close<static) qty=1.00 sl=395.80 alert=retest2 |

### Cycle 62 — SELL (started 2024-02-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 11:15:00 | 393.40 | 397.37 | 397.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 15:15:00 | 390.80 | 394.04 | 395.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 13:15:00 | 376.10 | 375.41 | 379.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 14:00:00 | 376.10 | 375.41 | 379.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 374.15 | 373.15 | 375.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:15:00 | 377.35 | 373.15 | 375.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 376.75 | 373.87 | 375.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:30:00 | 378.00 | 373.87 | 375.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 376.15 | 374.32 | 375.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 12:15:00 | 375.30 | 374.32 | 375.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 09:30:00 | 375.00 | 373.97 | 374.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 09:15:00 | 388.90 | 376.09 | 375.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 09:15:00 | 388.90 | 376.09 | 375.33 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 377.40 | 380.26 | 380.49 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 382.75 | 380.29 | 380.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 383.30 | 381.61 | 380.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 14:15:00 | 381.10 | 381.67 | 381.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 14:15:00 | 381.10 | 381.67 | 381.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 381.10 | 381.67 | 381.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 15:00:00 | 381.10 | 381.67 | 381.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 378.25 | 380.98 | 380.86 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 379.75 | 380.74 | 380.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 375.85 | 379.10 | 379.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 11:15:00 | 381.65 | 378.73 | 379.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 11:15:00 | 381.65 | 378.73 | 379.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 381.65 | 378.73 | 379.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:00:00 | 381.65 | 378.73 | 379.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 382.55 | 379.50 | 379.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:45:00 | 383.20 | 379.50 | 379.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 13:15:00 | 383.00 | 380.20 | 379.96 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 375.60 | 379.86 | 379.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 373.80 | 378.65 | 379.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 373.95 | 373.11 | 375.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 373.95 | 373.11 | 375.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 374.35 | 373.40 | 374.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:30:00 | 376.20 | 373.40 | 374.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 375.20 | 373.76 | 374.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:15:00 | 375.00 | 373.76 | 374.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 376.20 | 374.25 | 375.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:45:00 | 376.80 | 374.25 | 375.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 376.20 | 374.64 | 375.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:30:00 | 376.55 | 374.64 | 375.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 374.60 | 373.99 | 374.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:45:00 | 374.60 | 373.99 | 374.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 375.35 | 374.26 | 374.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 12:00:00 | 375.35 | 374.26 | 374.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 375.50 | 374.51 | 374.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 375.55 | 374.51 | 374.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 09:15:00 | 381.70 | 375.95 | 375.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 10:15:00 | 384.20 | 377.60 | 376.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 379.65 | 379.99 | 378.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 09:30:00 | 379.55 | 379.99 | 378.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 379.40 | 379.87 | 378.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:00:00 | 379.40 | 379.87 | 378.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 378.30 | 379.56 | 378.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 378.30 | 379.56 | 378.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 379.75 | 379.60 | 378.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 13:15:00 | 380.60 | 379.60 | 378.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 14:45:00 | 380.60 | 379.72 | 378.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 15:15:00 | 381.20 | 379.72 | 378.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 373.20 | 378.65 | 378.46 | SL hit (close<static) qty=1.00 sl=377.40 alert=retest2 |

### Cycle 70 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 373.20 | 377.56 | 377.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 371.40 | 376.33 | 377.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 377.35 | 375.02 | 376.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 377.35 | 375.02 | 376.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 377.35 | 375.02 | 376.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 377.35 | 375.02 | 376.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 379.40 | 375.89 | 376.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:00:00 | 379.40 | 375.89 | 376.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 380.00 | 377.16 | 376.94 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 10:15:00 | 375.25 | 376.88 | 376.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 15:15:00 | 373.00 | 375.24 | 376.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 356.30 | 354.79 | 360.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 356.30 | 354.79 | 360.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 351.25 | 354.64 | 358.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 350.00 | 354.64 | 358.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 09:15:00 | 363.00 | 357.84 | 358.01 | SL hit (close>static) qty=1.00 sl=359.75 alert=retest2 |

### Cycle 73 — BUY (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 10:15:00 | 360.40 | 358.35 | 358.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 12:15:00 | 363.70 | 360.94 | 359.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 358.95 | 361.65 | 360.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 358.95 | 361.65 | 360.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 358.95 | 361.65 | 360.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 358.95 | 361.65 | 360.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 360.50 | 361.42 | 360.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 360.30 | 361.42 | 360.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 359.70 | 361.07 | 360.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:30:00 | 360.70 | 361.07 | 360.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 360.10 | 360.88 | 360.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 12:45:00 | 360.05 | 360.88 | 360.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 357.95 | 360.06 | 360.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 15:15:00 | 357.45 | 359.54 | 359.95 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 363.00 | 360.23 | 360.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 364.80 | 361.15 | 360.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 364.80 | 367.36 | 365.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 14:15:00 | 364.80 | 367.36 | 365.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 364.80 | 367.36 | 365.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 15:00:00 | 364.80 | 367.36 | 365.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 365.95 | 367.08 | 365.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 370.00 | 367.08 | 365.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 367.90 | 367.24 | 365.56 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 11:15:00 | 363.20 | 365.65 | 365.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 362.80 | 365.08 | 365.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 14:15:00 | 362.00 | 361.98 | 363.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 14:15:00 | 362.00 | 361.98 | 363.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 362.00 | 361.98 | 363.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 15:00:00 | 362.00 | 361.98 | 363.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 368.40 | 363.11 | 363.59 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 370.90 | 364.67 | 364.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 372.05 | 366.14 | 364.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 383.70 | 384.95 | 381.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 10:00:00 | 383.70 | 384.95 | 381.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 382.25 | 383.52 | 381.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 15:15:00 | 384.50 | 383.01 | 381.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 09:45:00 | 384.35 | 383.59 | 381.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 14:15:00 | 384.90 | 386.39 | 386.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 12:15:00 | 380.70 | 387.24 | 387.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 380.70 | 387.24 | 387.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 379.85 | 385.76 | 387.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 11:15:00 | 372.00 | 371.65 | 376.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 11:45:00 | 372.20 | 371.65 | 376.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 374.35 | 373.17 | 375.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:45:00 | 372.45 | 373.93 | 375.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 11:15:00 | 378.00 | 371.45 | 371.46 | SL hit (close>static) qty=1.00 sl=376.55 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 378.55 | 372.87 | 372.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 10:15:00 | 380.70 | 378.56 | 376.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 378.75 | 378.87 | 376.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 13:00:00 | 378.75 | 378.87 | 376.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 395.25 | 399.61 | 397.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 395.25 | 399.61 | 397.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 396.50 | 398.99 | 397.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:30:00 | 394.40 | 398.33 | 396.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 396.20 | 397.90 | 396.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:30:00 | 395.05 | 397.90 | 396.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 395.30 | 397.38 | 396.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:30:00 | 395.25 | 397.38 | 396.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 396.40 | 397.19 | 396.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 13:45:00 | 397.20 | 397.28 | 396.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 14:45:00 | 396.90 | 396.93 | 396.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 397.95 | 396.75 | 396.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 10:15:00 | 393.75 | 396.17 | 396.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 393.75 | 396.17 | 396.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 388.50 | 394.64 | 395.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 380.30 | 380.29 | 384.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 380.30 | 380.29 | 384.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 380.45 | 380.10 | 382.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:45:00 | 381.60 | 380.10 | 382.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 373.90 | 372.71 | 374.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:15:00 | 375.45 | 372.71 | 374.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 377.05 | 373.58 | 374.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 377.95 | 373.58 | 374.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 377.50 | 374.36 | 374.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:30:00 | 377.35 | 374.36 | 374.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 379.50 | 375.39 | 375.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 380.10 | 378.44 | 377.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 378.75 | 378.99 | 378.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:15:00 | 378.85 | 378.99 | 378.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 378.40 | 378.89 | 378.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 379.65 | 378.89 | 378.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 09:15:00 | 376.90 | 380.77 | 380.19 | SL hit (close<static) qty=1.00 sl=377.05 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 378.40 | 379.55 | 379.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 377.75 | 379.00 | 379.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 378.75 | 378.74 | 379.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 378.75 | 378.74 | 379.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 378.75 | 378.74 | 379.21 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 381.35 | 379.04 | 378.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 384.25 | 380.40 | 379.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 383.00 | 383.89 | 381.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 383.00 | 383.89 | 381.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 383.00 | 383.89 | 381.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 383.00 | 383.89 | 381.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 381.90 | 383.49 | 381.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:30:00 | 381.75 | 383.49 | 381.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 382.25 | 383.24 | 381.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:30:00 | 384.50 | 381.87 | 381.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 380.30 | 381.56 | 381.43 | SL hit (close<static) qty=1.00 sl=381.40 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 378.80 | 381.01 | 381.19 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 382.15 | 381.24 | 381.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 13:15:00 | 383.40 | 381.68 | 381.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 382.15 | 382.44 | 381.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 382.15 | 382.44 | 381.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 382.15 | 382.44 | 381.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 382.15 | 382.44 | 381.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 379.65 | 381.88 | 381.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 379.65 | 381.88 | 381.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 380.40 | 381.59 | 381.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:15:00 | 380.80 | 381.59 | 381.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 12:15:00 | 380.90 | 381.45 | 381.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 380.90 | 381.45 | 381.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 377.90 | 380.74 | 381.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 382.35 | 380.34 | 380.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 382.35 | 380.34 | 380.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 382.35 | 380.34 | 380.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 382.35 | 380.34 | 380.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 381.45 | 380.56 | 380.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:30:00 | 380.25 | 380.52 | 380.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 380.85 | 380.86 | 380.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 379.00 | 380.49 | 380.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 387.10 | 381.63 | 381.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 387.10 | 381.63 | 381.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 391.35 | 386.50 | 383.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 383.20 | 386.67 | 384.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 383.20 | 386.67 | 384.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 383.20 | 386.67 | 384.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 379.10 | 386.67 | 384.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 363.15 | 381.96 | 382.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 353.10 | 376.19 | 379.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 376.75 | 371.61 | 375.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 376.75 | 371.61 | 375.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 376.75 | 371.61 | 375.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 376.75 | 371.61 | 375.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 387.85 | 374.86 | 376.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 387.85 | 374.86 | 376.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 386.35 | 377.16 | 377.68 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 385.00 | 378.73 | 378.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 390.80 | 382.38 | 380.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 13:15:00 | 386.75 | 386.97 | 383.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:00:00 | 386.75 | 386.97 | 383.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 426.00 | 430.26 | 426.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 426.00 | 430.26 | 426.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 427.50 | 429.71 | 426.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 432.05 | 429.08 | 427.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 430.10 | 439.99 | 441.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 09:15:00 | 430.10 | 439.99 | 441.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 425.80 | 430.27 | 433.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 432.15 | 430.25 | 432.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 432.15 | 430.25 | 432.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 432.15 | 430.25 | 432.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 432.50 | 430.25 | 432.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 433.55 | 430.91 | 432.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:45:00 | 435.10 | 430.91 | 432.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 432.00 | 431.13 | 432.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:00:00 | 429.15 | 430.55 | 432.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 434.50 | 431.28 | 432.18 | SL hit (close>static) qty=1.00 sl=433.90 alert=retest2 |

### Cycle 91 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 435.10 | 433.11 | 432.88 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 430.40 | 432.92 | 432.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 429.85 | 432.31 | 432.67 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 13:15:00 | 437.05 | 433.26 | 433.07 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 428.80 | 432.45 | 432.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 14:15:00 | 427.70 | 431.50 | 432.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 10:15:00 | 430.75 | 430.72 | 431.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:45:00 | 431.10 | 430.72 | 431.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 433.20 | 431.21 | 431.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:45:00 | 433.10 | 431.21 | 431.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 429.70 | 430.91 | 431.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 428.95 | 430.91 | 431.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:45:00 | 428.50 | 430.52 | 431.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 13:15:00 | 429.10 | 429.79 | 430.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 429.35 | 427.95 | 429.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 425.70 | 425.14 | 426.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:15:00 | 430.65 | 425.14 | 426.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 440.00 | 428.11 | 428.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 440.00 | 428.11 | 428.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 440.70 | 434.92 | 432.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 13:15:00 | 441.75 | 443.87 | 440.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 14:00:00 | 441.75 | 443.87 | 440.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 441.45 | 443.39 | 440.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:45:00 | 439.70 | 443.39 | 440.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 441.80 | 443.07 | 440.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 443.90 | 443.07 | 440.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:45:00 | 443.30 | 442.99 | 440.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:00:00 | 444.00 | 443.19 | 441.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 439.00 | 446.17 | 446.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 439.00 | 446.17 | 446.75 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 13:15:00 | 447.45 | 445.87 | 445.69 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 14:15:00 | 444.00 | 445.49 | 445.53 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 447.00 | 445.79 | 445.67 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 442.45 | 445.13 | 445.37 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 447.90 | 445.68 | 445.60 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 444.65 | 445.47 | 445.52 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 450.30 | 446.04 | 445.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 457.00 | 451.61 | 449.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 450.55 | 451.66 | 449.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 12:00:00 | 450.55 | 451.66 | 449.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 449.55 | 451.23 | 449.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 450.00 | 451.23 | 449.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 448.55 | 450.70 | 449.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 450.25 | 449.44 | 449.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 470.35 | 480.32 | 480.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 470.35 | 480.32 | 480.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 468.20 | 474.06 | 476.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 476.35 | 473.98 | 475.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 476.35 | 473.98 | 475.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 476.35 | 473.98 | 475.92 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 486.35 | 478.54 | 477.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 491.65 | 482.91 | 479.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 489.35 | 490.09 | 485.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 489.35 | 490.09 | 485.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 491.00 | 491.06 | 488.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 498.25 | 491.03 | 488.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-20 10:15:00 | 548.08 | 537.33 | 528.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 587.15 | 589.98 | 590.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 15:15:00 | 586.00 | 588.04 | 588.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 563.65 | 561.65 | 568.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 10:45:00 | 562.60 | 561.65 | 568.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 562.45 | 557.49 | 561.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 560.30 | 557.49 | 561.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 560.85 | 558.16 | 561.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 560.85 | 558.16 | 561.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 566.50 | 559.83 | 561.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 566.50 | 559.83 | 561.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 571.45 | 562.15 | 562.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 571.45 | 562.15 | 562.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 573.15 | 564.35 | 563.56 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 13:15:00 | 557.90 | 563.65 | 564.02 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 574.05 | 564.83 | 564.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 575.90 | 570.00 | 567.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 14:15:00 | 573.10 | 573.45 | 570.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 14:45:00 | 573.75 | 573.45 | 570.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 573.00 | 573.36 | 570.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 577.65 | 573.36 | 570.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:15:00 | 577.00 | 577.35 | 574.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-27 09:15:00 | 635.42 | 618.46 | 610.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 648.50 | 662.85 | 663.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 11:15:00 | 642.10 | 655.97 | 659.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 640.55 | 639.26 | 647.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 640.55 | 639.26 | 647.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 648.95 | 642.18 | 647.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 648.85 | 642.18 | 647.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 652.85 | 644.32 | 647.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 652.85 | 644.32 | 647.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 656.00 | 646.65 | 648.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 652.95 | 646.65 | 648.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 658.90 | 650.69 | 650.07 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 647.40 | 653.03 | 653.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 12:15:00 | 645.80 | 650.73 | 651.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 11:15:00 | 650.75 | 648.72 | 650.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 11:15:00 | 650.75 | 648.72 | 650.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 650.75 | 648.72 | 650.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:45:00 | 651.40 | 648.72 | 650.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 648.45 | 648.67 | 649.99 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 658.00 | 652.05 | 651.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 663.40 | 654.32 | 652.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 658.90 | 660.07 | 656.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 658.90 | 660.07 | 656.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 658.90 | 660.07 | 656.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 658.90 | 660.07 | 656.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 637.50 | 655.56 | 654.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 637.50 | 655.56 | 654.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 638.65 | 652.17 | 653.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 10:15:00 | 634.55 | 638.87 | 642.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 616.35 | 613.65 | 621.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:45:00 | 614.90 | 613.65 | 621.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 619.20 | 615.18 | 619.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 619.20 | 615.18 | 619.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 616.60 | 615.47 | 618.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:45:00 | 614.80 | 615.39 | 618.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 15:00:00 | 614.20 | 616.56 | 618.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 627.50 | 614.41 | 614.61 | SL hit (close>static) qty=1.00 sl=621.55 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 627.35 | 617.00 | 615.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 629.55 | 621.05 | 617.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 647.00 | 654.25 | 644.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:00:00 | 647.00 | 654.25 | 644.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 623.00 | 648.00 | 642.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 623.00 | 648.00 | 642.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 617.70 | 641.94 | 640.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 617.70 | 641.94 | 640.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 627.35 | 639.02 | 639.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 607.70 | 626.96 | 632.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 11:15:00 | 613.90 | 613.83 | 620.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 11:15:00 | 613.90 | 613.83 | 620.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 613.90 | 613.83 | 620.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 613.90 | 613.83 | 620.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 605.50 | 608.58 | 612.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 603.30 | 607.77 | 611.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 603.00 | 606.49 | 610.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:30:00 | 603.80 | 605.23 | 609.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 14:15:00 | 573.13 | 583.07 | 590.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 14:15:00 | 572.85 | 583.07 | 590.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 14:15:00 | 573.61 | 583.07 | 590.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-11 15:15:00 | 585.80 | 583.61 | 589.69 | SL hit (close>ema200) qty=0.50 sl=583.61 alert=retest2 |

### Cycle 117 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 537.85 | 523.10 | 523.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 13:15:00 | 540.00 | 528.81 | 525.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 15:15:00 | 565.10 | 565.73 | 555.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:15:00 | 568.20 | 565.73 | 555.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 552.40 | 562.14 | 556.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 552.40 | 562.14 | 556.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 552.50 | 560.21 | 555.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 554.30 | 560.21 | 555.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 557.55 | 559.68 | 556.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 559.55 | 559.68 | 556.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 13:15:00 | 586.45 | 589.86 | 589.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 13:15:00 | 586.45 | 589.86 | 589.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 15:15:00 | 585.00 | 588.24 | 589.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 15:15:00 | 585.00 | 584.78 | 586.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 15:15:00 | 585.00 | 584.78 | 586.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 585.00 | 584.78 | 586.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 585.00 | 584.78 | 586.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 580.35 | 583.89 | 586.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 11:00:00 | 577.80 | 582.67 | 585.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:45:00 | 578.50 | 577.85 | 581.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 584.00 | 582.55 | 582.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 584.00 | 582.55 | 582.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 587.25 | 584.02 | 583.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 580.10 | 583.39 | 583.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 580.10 | 583.39 | 583.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 580.10 | 583.39 | 583.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 580.10 | 583.39 | 583.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 581.40 | 582.99 | 582.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 583.95 | 582.99 | 582.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:30:00 | 583.30 | 584.43 | 584.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 583.00 | 584.25 | 584.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:30:00 | 583.60 | 584.00 | 583.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 586.90 | 584.58 | 584.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:30:00 | 590.95 | 585.20 | 584.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 579.85 | 586.99 | 587.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 579.85 | 586.99 | 587.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 577.10 | 585.01 | 586.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 15:15:00 | 581.60 | 580.62 | 583.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-19 09:15:00 | 567.00 | 580.62 | 583.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 577.80 | 575.32 | 578.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:30:00 | 583.85 | 575.32 | 578.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 570.45 | 574.34 | 578.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:45:00 | 568.00 | 572.41 | 576.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 538.65 | 557.15 | 567.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 539.60 | 557.15 | 567.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 13:15:00 | 522.80 | 521.24 | 528.98 | SL hit (close>ema200) qty=0.50 sl=521.24 alert=retest1 |

### Cycle 121 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 529.55 | 522.12 | 521.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 539.70 | 526.88 | 524.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 536.65 | 539.58 | 534.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 11:00:00 | 536.65 | 539.58 | 534.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 535.20 | 538.70 | 534.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 534.75 | 538.70 | 534.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 532.25 | 537.41 | 533.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:00:00 | 532.25 | 537.41 | 533.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 525.20 | 534.97 | 533.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:45:00 | 524.95 | 534.97 | 533.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 15:15:00 | 526.70 | 532.12 | 532.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 12:15:00 | 521.95 | 528.39 | 530.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 507.95 | 507.25 | 513.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:30:00 | 507.40 | 507.25 | 513.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 504.00 | 507.12 | 511.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:45:00 | 500.95 | 504.56 | 509.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:30:00 | 501.00 | 503.20 | 506.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 475.90 | 480.82 | 488.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 475.95 | 480.82 | 488.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 481.55 | 478.03 | 484.99 | SL hit (close>ema200) qty=0.50 sl=478.03 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 489.35 | 486.59 | 486.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 502.00 | 492.34 | 489.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 13:15:00 | 494.60 | 495.17 | 492.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 13:15:00 | 494.60 | 495.17 | 492.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 494.60 | 495.17 | 492.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:45:00 | 495.00 | 495.17 | 492.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 490.00 | 494.14 | 491.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 490.00 | 494.14 | 491.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 489.00 | 493.11 | 491.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 487.35 | 493.11 | 491.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 488.40 | 492.17 | 491.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 491.65 | 492.17 | 491.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 489.85 | 491.45 | 491.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 11:15:00 | 485.30 | 498.03 | 499.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 485.30 | 498.03 | 499.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 484.90 | 493.57 | 497.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 12:15:00 | 487.75 | 483.40 | 487.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 12:15:00 | 487.75 | 483.40 | 487.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 487.75 | 483.40 | 487.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 487.75 | 483.40 | 487.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 475.75 | 481.87 | 486.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 474.50 | 480.78 | 485.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 470.60 | 479.97 | 484.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 466.20 | 473.29 | 478.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 12:30:00 | 474.10 | 472.10 | 475.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 480.85 | 473.85 | 476.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:15:00 | 481.05 | 473.85 | 476.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 492.30 | 478.67 | 477.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 492.30 | 478.67 | 477.79 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 475.50 | 480.43 | 481.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 472.15 | 478.78 | 480.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 482.60 | 476.56 | 478.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 11:15:00 | 482.60 | 476.56 | 478.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 482.60 | 476.56 | 478.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 482.60 | 476.56 | 478.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 480.40 | 477.33 | 478.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:30:00 | 479.55 | 478.23 | 478.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 14:15:00 | 487.25 | 480.03 | 479.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 487.25 | 480.03 | 479.46 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 469.05 | 478.97 | 479.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 465.95 | 475.16 | 477.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 469.50 | 466.95 | 470.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 14:15:00 | 469.50 | 466.95 | 470.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 469.50 | 466.95 | 470.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 469.50 | 466.95 | 470.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 466.20 | 466.80 | 470.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 468.65 | 466.80 | 470.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 466.60 | 466.76 | 469.71 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 473.25 | 470.43 | 470.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 12:15:00 | 474.45 | 471.71 | 470.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 471.05 | 472.73 | 471.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 471.05 | 472.73 | 471.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 471.05 | 472.73 | 471.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 471.05 | 472.73 | 471.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 467.00 | 471.58 | 471.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:45:00 | 466.70 | 471.58 | 471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 465.00 | 470.27 | 470.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 461.65 | 467.67 | 469.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 444.60 | 442.75 | 449.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 444.60 | 442.75 | 449.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 435.70 | 429.27 | 432.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 435.70 | 429.27 | 432.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 443.15 | 432.05 | 433.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 442.60 | 432.05 | 433.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 430.05 | 432.90 | 433.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 430.05 | 432.90 | 433.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 432.05 | 429.99 | 431.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 436.90 | 429.99 | 431.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 438.50 | 431.69 | 432.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 440.80 | 431.69 | 432.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 441.50 | 433.65 | 433.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 442.15 | 435.35 | 433.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 13:15:00 | 474.15 | 475.57 | 467.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 13:30:00 | 474.35 | 475.57 | 467.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 467.55 | 473.13 | 468.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 465.30 | 473.13 | 468.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 464.85 | 471.48 | 467.82 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 457.35 | 465.54 | 465.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 455.15 | 463.46 | 464.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 447.05 | 445.24 | 452.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 447.05 | 445.24 | 452.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 431.85 | 440.65 | 445.51 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 446.15 | 443.80 | 443.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 456.65 | 448.27 | 445.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 12:15:00 | 461.90 | 463.83 | 460.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 13:00:00 | 461.90 | 463.83 | 460.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 461.25 | 463.32 | 460.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 460.50 | 463.32 | 460.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 452.50 | 461.15 | 459.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 452.50 | 461.15 | 459.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 454.40 | 459.80 | 459.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 447.90 | 459.80 | 459.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 449.80 | 457.80 | 458.22 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 10:15:00 | 464.10 | 456.50 | 456.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 11:15:00 | 470.75 | 459.35 | 457.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 15:15:00 | 471.95 | 472.37 | 468.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-17 09:15:00 | 471.10 | 472.37 | 468.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 466.00 | 471.10 | 468.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 466.60 | 471.10 | 468.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 469.00 | 470.68 | 468.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:30:00 | 473.00 | 471.42 | 469.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-19 09:15:00 | 520.30 | 500.03 | 486.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 524.70 | 530.41 | 530.48 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 13:15:00 | 534.80 | 530.77 | 530.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-25 14:15:00 | 535.10 | 531.64 | 531.01 | Break + close above crossover candle high |

### Cycle 138 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 516.80 | 528.99 | 529.94 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 536.40 | 529.10 | 528.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 541.15 | 532.62 | 530.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 14:15:00 | 556.90 | 558.45 | 552.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 15:00:00 | 556.90 | 558.45 | 552.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 557.65 | 558.29 | 553.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 556.45 | 558.29 | 553.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 556.00 | 557.83 | 553.28 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 553.40 | 555.30 | 555.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 527.00 | 548.66 | 552.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 536.80 | 534.60 | 542.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 536.80 | 534.60 | 542.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 541.80 | 536.04 | 542.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 544.25 | 536.04 | 542.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 538.10 | 536.45 | 542.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 12:00:00 | 536.55 | 536.93 | 541.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:15:00 | 536.15 | 537.39 | 541.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:45:00 | 536.50 | 529.68 | 529.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 536.80 | 529.68 | 529.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 537.70 | 531.28 | 530.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 537.70 | 531.28 | 530.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 540.00 | 533.03 | 531.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 557.00 | 558.88 | 550.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 557.00 | 558.88 | 550.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 554.25 | 557.93 | 553.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 554.25 | 557.93 | 553.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 553.00 | 556.94 | 553.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 551.65 | 556.94 | 553.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 553.90 | 556.33 | 553.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 558.00 | 557.02 | 553.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 573.00 | 584.38 | 584.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 573.00 | 584.38 | 584.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 570.00 | 581.50 | 583.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 15:15:00 | 552.40 | 551.25 | 557.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:15:00 | 546.70 | 551.25 | 557.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 554.15 | 552.31 | 556.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:45:00 | 556.25 | 552.31 | 556.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 555.85 | 553.02 | 556.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 555.85 | 553.02 | 556.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 551.00 | 552.61 | 555.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 546.00 | 551.23 | 554.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 547.80 | 550.78 | 554.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:30:00 | 547.20 | 549.99 | 553.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 10:15:00 | 562.15 | 552.81 | 553.28 | SL hit (close>static) qty=1.00 sl=556.35 alert=retest2 |

### Cycle 143 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 568.00 | 555.85 | 554.62 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 542.35 | 554.65 | 555.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 533.05 | 550.33 | 553.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 551.10 | 535.90 | 541.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 551.10 | 535.90 | 541.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 551.10 | 535.90 | 541.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 552.00 | 535.90 | 541.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 554.80 | 539.68 | 543.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 554.80 | 539.68 | 543.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 556.55 | 545.19 | 545.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 560.75 | 552.82 | 549.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 554.00 | 554.31 | 551.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 15:15:00 | 554.00 | 554.31 | 551.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 554.00 | 554.31 | 551.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 558.10 | 554.31 | 551.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 555.20 | 558.94 | 558.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 12:15:00 | 555.00 | 558.15 | 558.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 555.00 | 558.15 | 558.32 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 562.85 | 558.92 | 558.53 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 556.55 | 559.19 | 559.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 553.70 | 558.09 | 558.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 553.05 | 552.16 | 554.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 553.05 | 552.16 | 554.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 551.20 | 551.81 | 553.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:30:00 | 547.65 | 551.72 | 552.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 562.35 | 554.06 | 553.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 562.35 | 554.06 | 553.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 565.60 | 557.76 | 555.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 554.50 | 561.30 | 558.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 554.50 | 561.30 | 558.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 554.50 | 561.30 | 558.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 554.50 | 561.30 | 558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 558.15 | 560.67 | 558.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 562.80 | 558.92 | 558.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-09 13:15:00 | 619.08 | 613.63 | 608.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 609.20 | 615.72 | 616.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 606.85 | 613.94 | 615.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 610.80 | 610.68 | 612.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 606.05 | 610.68 | 612.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 600.20 | 608.58 | 611.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 593.85 | 604.39 | 606.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 11:30:00 | 596.00 | 592.77 | 592.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 596.50 | 593.52 | 593.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 596.50 | 593.52 | 593.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 608.00 | 597.08 | 594.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 597.70 | 599.61 | 596.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 597.70 | 599.61 | 596.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 597.15 | 599.12 | 596.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:15:00 | 598.00 | 599.12 | 596.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 597.40 | 598.78 | 596.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 596.60 | 598.78 | 596.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 596.60 | 598.34 | 596.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 594.00 | 597.08 | 596.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 588.50 | 595.37 | 595.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 12:15:00 | 586.65 | 592.54 | 594.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 595.40 | 591.69 | 593.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 595.40 | 591.69 | 593.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 595.40 | 591.69 | 593.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 595.40 | 591.69 | 593.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 593.35 | 592.02 | 593.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 592.50 | 592.02 | 593.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:00:00 | 591.00 | 592.13 | 593.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 593.00 | 590.99 | 591.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 592.50 | 590.78 | 591.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 590.65 | 590.76 | 591.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 588.70 | 590.76 | 591.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:30:00 | 588.15 | 587.18 | 588.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 10:00:00 | 589.30 | 587.64 | 587.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 590.50 | 588.38 | 588.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 590.50 | 588.38 | 588.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 590.90 | 588.88 | 588.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 596.40 | 598.29 | 595.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 11:00:00 | 596.40 | 598.29 | 595.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 597.50 | 598.13 | 595.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 599.00 | 598.13 | 595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 595.70 | 597.64 | 595.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 595.40 | 597.64 | 595.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 596.00 | 597.31 | 595.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 595.10 | 597.31 | 595.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 598.15 | 597.48 | 595.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:15:00 | 599.80 | 597.48 | 595.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 599.80 | 597.95 | 596.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 593.40 | 597.95 | 596.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 596.65 | 597.69 | 596.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 592.40 | 597.69 | 596.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 596.80 | 597.51 | 596.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 595.80 | 597.51 | 596.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 599.65 | 597.94 | 596.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 602.00 | 597.94 | 596.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:45:00 | 600.55 | 598.95 | 597.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 15:00:00 | 601.20 | 599.40 | 597.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 615.55 | 616.55 | 616.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 615.55 | 616.55 | 616.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 613.00 | 615.84 | 616.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 614.50 | 614.48 | 615.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 614.50 | 614.48 | 615.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 614.50 | 614.48 | 615.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 614.50 | 614.48 | 615.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 617.45 | 615.08 | 615.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 617.45 | 615.08 | 615.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 616.55 | 615.37 | 615.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:45:00 | 615.70 | 615.87 | 615.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 616.70 | 616.03 | 615.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 616.70 | 616.03 | 615.99 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 613.65 | 615.59 | 615.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 611.20 | 613.71 | 614.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 612.95 | 612.85 | 613.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 612.95 | 612.85 | 613.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 612.95 | 612.85 | 613.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 609.20 | 612.85 | 613.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 615.65 | 610.51 | 611.96 | SL hit (close>static) qty=1.00 sl=614.45 alert=retest2 |

### Cycle 157 — BUY (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 11:15:00 | 617.25 | 612.88 | 612.85 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 612.10 | 612.73 | 612.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 605.95 | 611.37 | 612.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 590.10 | 587.50 | 592.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 590.30 | 587.50 | 592.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 594.00 | 589.92 | 591.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 592.40 | 589.92 | 591.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 590.45 | 590.03 | 591.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 592.50 | 590.03 | 591.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 591.00 | 589.11 | 590.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 587.20 | 589.11 | 590.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 587.05 | 588.05 | 589.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 557.84 | 564.78 | 568.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 557.70 | 564.78 | 568.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 11:15:00 | 551.60 | 550.45 | 556.69 | SL hit (close>ema200) qty=0.50 sl=550.45 alert=retest2 |

### Cycle 159 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 553.60 | 550.50 | 550.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 563.15 | 553.03 | 551.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 547.20 | 552.17 | 551.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 547.20 | 552.17 | 551.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 547.20 | 552.17 | 551.28 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 545.75 | 550.17 | 550.48 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 553.95 | 550.20 | 550.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 556.00 | 551.36 | 550.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 555.00 | 557.84 | 554.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 555.00 | 557.84 | 554.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 555.00 | 557.84 | 554.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 553.55 | 557.84 | 554.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 567.00 | 562.44 | 559.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 561.95 | 562.44 | 559.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 581.90 | 585.51 | 581.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 588.95 | 585.51 | 581.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 572.00 | 585.44 | 584.42 | SL hit (close<static) qty=1.00 sl=581.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 570.25 | 582.40 | 583.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 558.55 | 572.59 | 577.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 545.25 | 543.30 | 551.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 545.25 | 543.30 | 551.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 577.70 | 550.56 | 551.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 578.55 | 550.56 | 551.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 578.85 | 556.22 | 553.58 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 550.95 | 558.68 | 558.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 549.85 | 552.21 | 554.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 551.95 | 551.35 | 553.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 551.95 | 551.35 | 553.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 551.95 | 551.35 | 553.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 544.50 | 549.80 | 552.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 15:15:00 | 517.27 | 529.41 | 537.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 525.00 | 524.90 | 531.47 | SL hit (close>ema200) qty=0.50 sl=524.90 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 521.30 | 518.76 | 518.70 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 517.30 | 518.43 | 518.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 515.85 | 517.91 | 518.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 455.30 | 453.37 | 458.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 15:00:00 | 455.30 | 453.37 | 458.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 454.45 | 454.03 | 457.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 451.50 | 453.52 | 456.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 460.95 | 456.16 | 456.79 | SL hit (close>static) qty=1.00 sl=458.70 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 462.55 | 458.03 | 457.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 464.10 | 459.25 | 458.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 476.65 | 477.66 | 473.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 479.30 | 478.56 | 475.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 479.30 | 478.56 | 475.90 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 469.90 | 476.46 | 476.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 468.50 | 472.00 | 473.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 468.25 | 466.98 | 470.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 468.25 | 466.98 | 470.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 468.25 | 466.98 | 470.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 468.20 | 466.98 | 470.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 471.00 | 468.33 | 469.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 470.50 | 468.33 | 469.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 469.70 | 468.60 | 469.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 469.70 | 468.60 | 469.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 468.90 | 468.66 | 469.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 471.55 | 468.66 | 469.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 470.35 | 469.00 | 469.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 471.90 | 469.00 | 469.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 469.40 | 469.08 | 469.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 467.10 | 468.30 | 469.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 469.00 | 468.32 | 469.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 15:00:00 | 467.60 | 468.13 | 468.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 468.45 | 468.12 | 468.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 468.00 | 468.09 | 468.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 469.05 | 468.09 | 468.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 467.25 | 467.92 | 468.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 468.70 | 467.92 | 468.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 461.00 | 464.80 | 466.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:30:00 | 460.40 | 463.92 | 466.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:00:00 | 460.40 | 463.92 | 466.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 467.05 | 462.27 | 463.95 | SL hit (close>static) qty=1.00 sl=466.75 alert=retest2 |

### Cycle 169 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 469.75 | 465.00 | 464.97 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 462.10 | 465.60 | 466.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 461.55 | 464.79 | 465.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 464.85 | 464.80 | 465.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 464.85 | 464.80 | 465.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 465.00 | 464.59 | 465.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 462.90 | 464.25 | 465.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 462.20 | 464.12 | 464.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 458.25 | 462.38 | 463.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 468.65 | 463.98 | 464.02 | SL hit (close>static) qty=1.00 sl=466.90 alert=retest2 |

### Cycle 171 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 469.80 | 465.14 | 464.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 477.10 | 468.42 | 466.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 15:15:00 | 470.00 | 470.86 | 468.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:15:00 | 477.10 | 470.86 | 468.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 473.20 | 472.97 | 470.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 471.75 | 472.97 | 470.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 469.75 | 472.32 | 470.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 469.75 | 472.32 | 470.35 | SL hit (close<ema400) qty=1.00 sl=470.35 alert=retest1 |

### Cycle 172 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 462.50 | 468.63 | 469.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 461.70 | 465.61 | 467.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 461.10 | 460.41 | 463.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 461.10 | 460.41 | 463.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 461.10 | 460.41 | 463.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 461.10 | 460.41 | 463.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 460.35 | 460.34 | 462.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 456.75 | 459.25 | 461.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 433.91 | 442.93 | 449.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 445.70 | 438.37 | 443.62 | SL hit (close>ema200) qty=0.50 sl=438.37 alert=retest2 |

### Cycle 173 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 458.00 | 446.48 | 446.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 461.15 | 451.28 | 448.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 450.75 | 452.55 | 449.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:30:00 | 451.05 | 452.55 | 449.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 450.35 | 452.11 | 449.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 450.30 | 452.11 | 449.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 447.70 | 451.23 | 449.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 447.70 | 451.23 | 449.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 449.60 | 450.90 | 449.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 450.05 | 450.90 | 449.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:00:00 | 450.25 | 451.67 | 450.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:30:00 | 450.10 | 451.06 | 450.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 453.30 | 450.65 | 450.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 475.30 | 466.24 | 462.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 464.00 | 466.24 | 462.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 461.90 | 467.37 | 464.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 461.90 | 467.37 | 464.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 461.15 | 466.12 | 464.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 460.65 | 466.12 | 464.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 466.75 | 465.92 | 464.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:15:00 | 464.40 | 465.92 | 464.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 463.40 | 465.41 | 464.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 463.75 | 465.41 | 464.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 461.70 | 464.67 | 463.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:30:00 | 463.10 | 464.67 | 463.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 462.60 | 463.82 | 463.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 458.95 | 463.82 | 463.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 460.85 | 463.23 | 463.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 460.85 | 463.23 | 463.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 458.10 | 460.31 | 461.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 455.00 | 450.31 | 452.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 455.00 | 450.31 | 452.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 455.00 | 450.31 | 452.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 455.00 | 450.31 | 452.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 446.55 | 449.56 | 452.29 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 452.00 | 451.41 | 451.34 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 450.15 | 451.16 | 451.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 447.50 | 449.62 | 450.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 12:15:00 | 447.65 | 447.58 | 448.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:00:00 | 447.65 | 447.58 | 448.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 447.80 | 447.62 | 448.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 448.85 | 447.62 | 448.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 447.75 | 447.49 | 448.10 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 450.00 | 448.45 | 448.31 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 447.25 | 448.13 | 448.19 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 449.45 | 448.39 | 448.29 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 444.15 | 447.55 | 447.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 442.70 | 445.04 | 446.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 443.25 | 442.99 | 444.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 443.25 | 442.99 | 444.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 444.80 | 443.35 | 444.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 444.80 | 443.35 | 444.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 442.20 | 443.12 | 444.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 444.75 | 443.12 | 444.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 445.30 | 443.56 | 444.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:00:00 | 440.65 | 442.88 | 443.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 418.62 | 430.07 | 436.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 431.35 | 423.01 | 428.51 | SL hit (close>ema200) qty=0.50 sl=423.01 alert=retest2 |

### Cycle 181 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 444.75 | 432.83 | 431.80 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 441.00 | 442.39 | 442.49 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 445.45 | 442.68 | 442.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 447.00 | 443.54 | 442.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 442.10 | 445.30 | 444.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 442.10 | 445.30 | 444.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 442.10 | 445.30 | 444.12 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 11:15:00 | 440.00 | 443.41 | 443.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 436.75 | 442.08 | 442.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 440.30 | 440.02 | 441.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 10:15:00 | 440.90 | 440.02 | 441.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 441.55 | 440.33 | 441.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 441.55 | 440.33 | 441.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 441.55 | 440.57 | 441.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:30:00 | 441.10 | 440.57 | 441.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 440.40 | 440.54 | 441.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:30:00 | 441.80 | 440.54 | 441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 440.75 | 440.53 | 441.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:30:00 | 440.50 | 440.53 | 441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 441.05 | 440.63 | 441.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 442.55 | 440.63 | 441.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 443.45 | 441.20 | 441.39 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 444.10 | 441.98 | 441.73 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 440.80 | 441.82 | 441.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 439.20 | 440.92 | 441.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 435.05 | 434.77 | 436.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:45:00 | 434.80 | 434.77 | 436.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 436.45 | 435.19 | 436.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 436.45 | 435.19 | 436.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 436.60 | 435.47 | 436.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 436.50 | 435.47 | 436.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 438.50 | 436.08 | 436.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 439.00 | 436.08 | 436.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 439.75 | 436.81 | 436.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 438.60 | 436.81 | 436.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 443.40 | 438.13 | 437.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 445.60 | 439.62 | 438.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 441.75 | 442.58 | 440.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 441.75 | 442.58 | 440.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 439.80 | 442.02 | 440.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 438.35 | 442.02 | 440.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 438.80 | 441.38 | 440.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 438.80 | 441.38 | 440.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 438.75 | 439.71 | 439.78 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 441.10 | 440.00 | 439.90 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 438.90 | 439.86 | 439.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 437.90 | 439.47 | 439.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 11:15:00 | 440.25 | 439.06 | 439.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 11:15:00 | 440.25 | 439.06 | 439.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 440.25 | 439.06 | 439.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 440.25 | 439.06 | 439.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 440.10 | 439.27 | 439.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 440.10 | 439.27 | 439.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 440.25 | 439.29 | 439.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 441.00 | 439.29 | 439.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 439.00 | 439.23 | 439.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 437.10 | 439.23 | 439.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 437.10 | 435.93 | 436.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 425.00 | 424.48 | 424.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 425.00 | 424.48 | 424.43 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 421.50 | 423.88 | 424.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 420.30 | 423.16 | 423.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 14:15:00 | 421.65 | 421.26 | 422.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 15:00:00 | 421.65 | 421.26 | 422.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 421.20 | 421.25 | 422.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 418.50 | 421.25 | 422.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 412.10 | 416.43 | 418.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 397.57 | 408.00 | 412.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 406.50 | 402.20 | 407.02 | SL hit (close>ema200) qty=0.50 sl=402.20 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 13:15:00 | 410.65 | 407.23 | 406.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 415.80 | 410.41 | 408.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 11:15:00 | 404.20 | 409.85 | 408.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 11:15:00 | 404.20 | 409.85 | 408.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 404.20 | 409.85 | 408.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 404.20 | 409.85 | 408.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 403.45 | 408.57 | 408.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:15:00 | 402.55 | 408.57 | 408.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 402.00 | 407.26 | 407.59 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 412.80 | 407.92 | 407.77 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 406.10 | 408.26 | 408.45 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 410.00 | 408.54 | 408.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 417.50 | 410.33 | 409.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 419.95 | 424.93 | 419.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 419.95 | 424.93 | 419.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 419.95 | 424.93 | 419.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 419.95 | 424.93 | 419.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 422.90 | 424.52 | 419.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 424.05 | 424.52 | 419.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 418.40 | 423.30 | 419.58 | SL hit (close<static) qty=1.00 sl=419.55 alert=retest2 |

### Cycle 198 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 463.50 | 466.04 | 466.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 461.45 | 464.18 | 465.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 461.90 | 457.77 | 459.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 11:15:00 | 461.90 | 457.77 | 459.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 461.90 | 457.77 | 459.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 461.90 | 457.77 | 459.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 461.25 | 458.47 | 459.69 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 464.35 | 461.14 | 460.73 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 457.05 | 460.13 | 460.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 455.35 | 459.18 | 459.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 460.50 | 458.62 | 459.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 460.50 | 458.62 | 459.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 460.50 | 458.62 | 459.29 | EMA400 retest candle locked (from downside) |

### Cycle 201 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 462.50 | 460.02 | 459.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 467.15 | 461.44 | 460.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 461.30 | 462.32 | 461.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 461.30 | 462.32 | 461.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 461.30 | 462.32 | 461.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 461.30 | 462.32 | 461.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 460.20 | 461.89 | 461.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 460.00 | 461.89 | 461.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 462.45 | 462.01 | 461.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 460.60 | 462.01 | 461.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 461.00 | 461.80 | 461.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 461.00 | 461.80 | 461.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 461.95 | 461.83 | 461.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:15:00 | 460.50 | 461.83 | 461.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 460.50 | 461.57 | 461.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 462.15 | 461.57 | 461.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 458.25 | 464.49 | 464.38 | SL hit (close<static) qty=1.00 sl=458.95 alert=retest2 |

### Cycle 202 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 461.05 | 463.80 | 464.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 454.95 | 460.42 | 462.34 | Break + close below crossover candle low |

### Cycle 203 — BUY (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 10:15:00 | 485.00 | 465.34 | 464.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 505.65 | 493.02 | 488.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 508.50 | 511.83 | 503.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 10:15:00 | 506.55 | 511.83 | 503.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 503.00 | 508.24 | 505.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 498.10 | 506.21 | 504.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 496.50 | 504.27 | 504.01 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 496.95 | 502.81 | 503.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 488.00 | 499.85 | 501.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 482.85 | 475.92 | 481.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 482.85 | 475.92 | 481.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 482.85 | 475.92 | 481.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 482.15 | 475.92 | 481.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 485.90 | 477.91 | 482.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 485.55 | 477.91 | 482.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 481.70 | 479.15 | 481.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 481.70 | 479.15 | 481.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 477.00 | 478.72 | 481.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:30:00 | 475.90 | 478.03 | 480.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 475.25 | 478.03 | 480.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 484.00 | 480.70 | 481.08 | SL hit (close>static) qty=1.00 sl=482.15 alert=retest2 |

### Cycle 205 — BUY (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 15:15:00 | 485.50 | 481.99 | 481.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 490.00 | 483.59 | 482.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 15:15:00 | 482.00 | 484.34 | 483.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 482.00 | 484.34 | 483.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 482.00 | 484.34 | 483.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 481.40 | 484.34 | 483.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 473.10 | 482.09 | 482.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 468.85 | 477.92 | 480.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 470.60 | 465.18 | 471.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 470.60 | 465.18 | 471.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 470.60 | 465.18 | 471.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 470.60 | 465.18 | 471.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 474.85 | 467.12 | 471.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 474.85 | 467.12 | 471.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 476.55 | 469.00 | 471.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 476.55 | 469.00 | 471.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 494.05 | 475.29 | 474.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 504.20 | 481.07 | 477.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 497.65 | 499.97 | 491.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:45:00 | 497.75 | 499.97 | 491.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 490.60 | 498.10 | 491.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 490.60 | 498.10 | 491.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 490.70 | 496.62 | 491.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 490.70 | 496.62 | 491.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 501.55 | 497.61 | 492.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 503.95 | 497.43 | 492.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 504.70 | 495.72 | 494.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 485.00 | 493.75 | 494.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 485.00 | 493.75 | 494.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 09:15:00 | 477.50 | 483.72 | 486.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 15:15:00 | 481.50 | 481.41 | 483.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 484.60 | 481.41 | 483.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 485.70 | 482.27 | 483.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 484.70 | 482.27 | 483.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 484.00 | 482.62 | 483.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:00:00 | 482.75 | 482.80 | 483.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 09:45:00 | 482.95 | 477.48 | 478.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 484.00 | 478.78 | 478.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 484.00 | 478.78 | 478.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 489.50 | 485.99 | 484.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 509.10 | 510.42 | 502.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 509.10 | 510.42 | 502.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 523.00 | 535.09 | 527.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 522.50 | 535.09 | 527.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 507.65 | 529.60 | 526.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 507.65 | 529.60 | 526.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 518.30 | 523.23 | 523.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 515.00 | 521.58 | 522.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 531.00 | 522.73 | 523.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 531.00 | 522.73 | 523.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 531.00 | 522.73 | 523.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 531.00 | 522.73 | 523.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 531.00 | 524.39 | 523.88 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 520.90 | 523.37 | 523.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 514.05 | 521.05 | 522.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 14:15:00 | 510.10 | 507.77 | 511.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 510.10 | 507.77 | 511.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 510.10 | 507.77 | 511.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 510.10 | 507.77 | 511.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 518.70 | 510.16 | 512.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 521.00 | 510.16 | 512.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 533.75 | 514.88 | 514.21 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 521.75 | 524.62 | 524.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 519.45 | 522.91 | 524.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 14:15:00 | 523.35 | 521.94 | 523.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 523.35 | 521.94 | 523.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 523.35 | 521.94 | 523.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 523.35 | 521.94 | 523.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 522.00 | 521.95 | 523.09 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-24 09:45:00 | 391.75 | 2023-05-25 09:15:00 | 387.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-05-24 12:15:00 | 390.55 | 2023-05-25 09:15:00 | 387.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-05-24 12:45:00 | 391.25 | 2023-05-25 09:15:00 | 387.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-05-24 15:00:00 | 391.40 | 2023-05-25 09:15:00 | 387.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-05-31 09:30:00 | 393.80 | 2023-05-31 11:15:00 | 391.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-06-01 09:15:00 | 394.95 | 2023-06-01 10:15:00 | 391.45 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-06-09 09:15:00 | 411.50 | 2023-06-13 15:15:00 | 401.20 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2023-07-03 12:30:00 | 383.40 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-07-03 15:15:00 | 383.55 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-07-04 09:45:00 | 382.30 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-07-04 12:45:00 | 382.35 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-07-05 12:30:00 | 382.90 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-07-06 10:45:00 | 383.50 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-07-06 11:15:00 | 383.35 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-07-06 13:15:00 | 382.85 | 2023-07-07 09:15:00 | 386.65 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-07-10 15:15:00 | 382.00 | 2023-07-12 09:15:00 | 390.45 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2023-07-11 11:00:00 | 382.00 | 2023-07-12 09:15:00 | 390.45 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2023-07-11 12:00:00 | 381.90 | 2023-07-12 09:15:00 | 390.45 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2023-07-11 14:15:00 | 381.80 | 2023-07-12 09:15:00 | 390.45 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2023-07-21 14:15:00 | 392.20 | 2023-07-28 09:15:00 | 393.20 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2023-08-11 11:00:00 | 392.30 | 2023-08-16 13:15:00 | 394.20 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-08-11 12:45:00 | 392.00 | 2023-08-16 13:15:00 | 394.20 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-08-16 09:15:00 | 389.85 | 2023-08-16 13:15:00 | 394.20 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-09-06 09:15:00 | 411.80 | 2023-09-08 15:15:00 | 406.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2023-09-08 14:00:00 | 407.30 | 2023-09-08 15:15:00 | 406.90 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2023-09-08 14:45:00 | 407.30 | 2023-09-08 15:15:00 | 406.90 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2023-09-20 12:30:00 | 442.30 | 2023-09-21 12:15:00 | 430.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2023-10-05 10:15:00 | 423.45 | 2023-10-06 15:15:00 | 430.55 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-10-05 14:45:00 | 424.55 | 2023-10-06 15:15:00 | 430.55 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-10-10 13:30:00 | 425.00 | 2023-10-16 11:15:00 | 424.20 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2023-10-11 10:15:00 | 425.00 | 2023-10-16 11:15:00 | 424.20 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2023-10-11 11:15:00 | 424.75 | 2023-10-16 11:15:00 | 424.20 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2023-10-16 10:15:00 | 424.60 | 2023-10-16 11:15:00 | 424.20 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2023-10-23 11:00:00 | 409.05 | 2023-10-27 14:15:00 | 413.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-10-25 12:15:00 | 409.80 | 2023-10-27 14:15:00 | 413.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-11-08 15:15:00 | 431.50 | 2023-11-13 12:15:00 | 427.75 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-11-17 13:00:00 | 437.15 | 2023-11-30 09:15:00 | 480.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-17 13:45:00 | 437.20 | 2023-11-30 09:15:00 | 480.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-15 12:30:00 | 386.80 | 2023-12-18 09:15:00 | 404.60 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2024-01-09 14:15:00 | 398.00 | 2024-01-18 09:15:00 | 378.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-09 14:15:00 | 398.00 | 2024-01-18 11:15:00 | 387.35 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2024-01-10 09:15:00 | 392.60 | 2024-01-18 12:15:00 | 372.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-10 09:15:00 | 392.60 | 2024-01-18 14:15:00 | 385.00 | STOP_HIT | 0.50 | 1.94% |
| BUY | retest2 | 2024-01-25 14:00:00 | 387.35 | 2024-02-01 13:15:00 | 389.80 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-01-25 15:00:00 | 387.95 | 2024-02-01 13:15:00 | 389.80 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2024-01-29 11:30:00 | 386.80 | 2024-02-01 13:15:00 | 389.80 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2024-02-07 09:15:00 | 399.70 | 2024-02-08 09:15:00 | 395.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-02-07 10:15:00 | 403.05 | 2024-02-08 09:15:00 | 395.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-02-15 12:15:00 | 375.30 | 2024-02-19 09:15:00 | 388.90 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-02-16 09:30:00 | 375.00 | 2024-02-19 09:15:00 | 388.90 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-03-05 13:15:00 | 380.60 | 2024-03-06 09:15:00 | 373.20 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-03-05 14:45:00 | 380.60 | 2024-03-06 09:15:00 | 373.20 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-03-05 15:15:00 | 381.20 | 2024-03-06 09:15:00 | 373.20 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-03-15 10:15:00 | 350.00 | 2024-03-18 09:15:00 | 363.00 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-04-04 15:15:00 | 384.50 | 2024-04-12 12:15:00 | 380.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-04-05 09:45:00 | 384.35 | 2024-04-12 12:15:00 | 380.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-04-09 14:15:00 | 384.90 | 2024-04-12 12:15:00 | 380.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-04-18 13:45:00 | 372.45 | 2024-04-22 11:15:00 | 378.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-02 13:45:00 | 397.20 | 2024-05-03 10:15:00 | 393.75 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-05-02 14:45:00 | 396.90 | 2024-05-03 10:15:00 | 393.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-03 09:15:00 | 397.95 | 2024-05-03 10:15:00 | 393.75 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-05-17 09:15:00 | 379.65 | 2024-05-21 09:15:00 | 376.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-05-28 09:30:00 | 384.50 | 2024-05-28 10:15:00 | 380.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-05-30 12:15:00 | 380.80 | 2024-05-30 12:15:00 | 380.90 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-05-31 12:30:00 | 380.25 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-05-31 14:15:00 | 380.85 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-05-31 15:00:00 | 379.00 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-06-19 10:45:00 | 432.05 | 2024-06-26 09:15:00 | 430.10 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-06-28 15:00:00 | 429.15 | 2024-07-01 09:15:00 | 434.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-07-04 13:15:00 | 428.95 | 2024-07-09 10:15:00 | 440.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-07-04 13:45:00 | 428.50 | 2024-07-09 10:15:00 | 440.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-07-05 13:15:00 | 429.10 | 2024-07-09 10:15:00 | 440.00 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-07-08 10:30:00 | 429.35 | 2024-07-09 10:15:00 | 440.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-07-15 09:15:00 | 443.90 | 2024-07-19 10:15:00 | 439.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-15 10:45:00 | 443.30 | 2024-07-19 10:15:00 | 439.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-15 12:00:00 | 444.00 | 2024-07-19 10:15:00 | 439.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-07-26 10:15:00 | 450.25 | 2024-08-05 10:15:00 | 470.35 | STOP_HIT | 1.00 | 4.46% |
| BUY | retest2 | 2024-08-12 09:15:00 | 498.25 | 2024-08-20 10:15:00 | 548.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 09:15:00 | 577.65 | 2024-09-27 09:15:00 | 635.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 14:15:00 | 577.00 | 2024-09-27 09:15:00 | 634.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-24 11:45:00 | 614.80 | 2024-10-28 10:15:00 | 627.50 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-10-24 15:00:00 | 614.20 | 2024-10-28 10:15:00 | 627.50 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-11-07 10:30:00 | 603.30 | 2024-11-11 14:15:00 | 573.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 603.00 | 2024-11-11 14:15:00 | 572.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 14:30:00 | 603.80 | 2024-11-11 14:15:00 | 573.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 10:30:00 | 603.30 | 2024-11-11 15:15:00 | 585.80 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2024-11-07 13:15:00 | 603.00 | 2024-11-11 15:15:00 | 585.80 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2024-11-07 14:30:00 | 603.80 | 2024-11-11 15:15:00 | 585.80 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2024-11-28 14:15:00 | 559.55 | 2024-12-05 13:15:00 | 586.45 | STOP_HIT | 1.00 | 4.81% |
| SELL | retest2 | 2024-12-09 11:00:00 | 577.80 | 2024-12-10 15:15:00 | 584.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-12-10 09:45:00 | 578.50 | 2024-12-10 15:15:00 | 584.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-12 11:15:00 | 583.95 | 2024-12-18 09:15:00 | 579.85 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-12-13 11:30:00 | 583.30 | 2024-12-18 09:15:00 | 579.85 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-12-13 12:45:00 | 583.00 | 2024-12-18 09:15:00 | 579.85 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-12-13 13:30:00 | 583.60 | 2024-12-18 09:15:00 | 579.85 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-12-16 09:30:00 | 590.95 | 2024-12-18 09:15:00 | 579.85 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2024-12-19 09:15:00 | 567.00 | 2024-12-20 14:15:00 | 538.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 09:45:00 | 568.00 | 2024-12-20 14:15:00 | 539.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-19 09:15:00 | 567.00 | 2024-12-26 13:15:00 | 522.80 | STOP_HIT | 0.50 | 7.80% |
| SELL | retest2 | 2024-12-20 09:45:00 | 568.00 | 2024-12-26 13:15:00 | 522.80 | STOP_HIT | 0.50 | 7.96% |
| SELL | retest2 | 2025-01-08 13:45:00 | 500.95 | 2025-01-13 14:15:00 | 475.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 501.00 | 2025-01-13 14:15:00 | 475.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 13:45:00 | 500.95 | 2025-01-14 10:15:00 | 481.55 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2025-01-09 10:30:00 | 501.00 | 2025-01-14 10:15:00 | 481.55 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-01-17 10:15:00 | 491.65 | 2025-01-22 11:15:00 | 485.30 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-01-17 11:15:00 | 489.85 | 2025-01-22 11:15:00 | 485.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-01-24 14:30:00 | 474.50 | 2025-01-29 09:15:00 | 492.30 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-01-27 09:15:00 | 470.60 | 2025-01-29 09:15:00 | 492.30 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-01-28 09:15:00 | 466.20 | 2025-01-29 09:15:00 | 492.30 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2025-01-28 12:30:00 | 474.10 | 2025-01-29 09:15:00 | 492.30 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-01-31 13:30:00 | 479.55 | 2025-01-31 14:15:00 | 487.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-03-17 14:30:00 | 473.00 | 2025-03-19 09:15:00 | 520.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 12:00:00 | 536.55 | 2025-04-15 10:15:00 | 537.70 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-04-08 13:15:00 | 536.15 | 2025-04-15 10:15:00 | 537.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-04-15 09:45:00 | 536.50 | 2025-04-15 10:15:00 | 537.70 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-04-15 10:15:00 | 536.80 | 2025-04-15 10:15:00 | 537.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-04-21 10:30:00 | 558.00 | 2025-04-25 14:15:00 | 573.00 | STOP_HIT | 1.00 | 2.69% |
| SELL | retest2 | 2025-05-06 09:45:00 | 546.00 | 2025-05-07 10:15:00 | 562.15 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-05-06 11:15:00 | 547.80 | 2025-05-07 10:15:00 | 562.15 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-05-06 12:30:00 | 547.20 | 2025-05-07 10:15:00 | 562.15 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-05-14 09:15:00 | 558.10 | 2025-05-16 12:15:00 | 555.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-16 11:30:00 | 555.20 | 2025-05-16 12:15:00 | 555.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-05-23 10:30:00 | 547.65 | 2025-05-26 09:15:00 | 562.35 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-05-28 09:15:00 | 562.80 | 2025-06-09 13:15:00 | 619.08 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-18 13:15:00 | 593.85 | 2025-06-24 12:15:00 | 596.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-06-24 11:30:00 | 596.00 | 2025-06-24 12:15:00 | 596.50 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-06-27 11:15:00 | 592.50 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-06-27 13:00:00 | 591.00 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-06-30 15:15:00 | 593.00 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-07-01 11:30:00 | 592.50 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-07-01 13:15:00 | 588.70 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-07-02 12:30:00 | 588.15 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-04 10:00:00 | 589.30 | 2025-07-04 12:15:00 | 590.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-07-09 12:15:00 | 602.00 | 2025-07-17 14:15:00 | 615.55 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2025-07-09 13:45:00 | 600.55 | 2025-07-17 14:15:00 | 615.55 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-07-09 15:00:00 | 601.20 | 2025-07-17 14:15:00 | 615.55 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2025-07-18 14:45:00 | 615.70 | 2025-07-18 15:15:00 | 616.70 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-07-23 11:15:00 | 609.20 | 2025-07-24 09:15:00 | 615.65 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-07-31 09:15:00 | 587.20 | 2025-08-06 09:15:00 | 557.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:00:00 | 587.05 | 2025-08-06 09:15:00 | 557.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 587.20 | 2025-08-07 11:15:00 | 551.60 | STOP_HIT | 0.50 | 6.06% |
| SELL | retest2 | 2025-07-31 13:00:00 | 587.05 | 2025-08-07 11:15:00 | 551.60 | STOP_HIT | 0.50 | 6.04% |
| BUY | retest2 | 2025-08-25 09:15:00 | 588.95 | 2025-08-26 09:15:00 | 572.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-09-08 10:30:00 | 544.50 | 2025-09-09 15:15:00 | 517.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 10:30:00 | 544.50 | 2025-09-10 14:15:00 | 525.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-10-01 11:00:00 | 451.50 | 2025-10-03 09:15:00 | 460.95 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-10-16 11:30:00 | 467.10 | 2025-10-21 13:15:00 | 467.05 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-10-16 12:30:00 | 469.00 | 2025-10-21 13:15:00 | 467.05 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-16 15:00:00 | 467.60 | 2025-10-23 09:15:00 | 469.75 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-17 10:00:00 | 468.45 | 2025-10-23 09:15:00 | 469.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-20 10:30:00 | 460.40 | 2025-10-23 09:15:00 | 469.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-10-20 11:00:00 | 460.40 | 2025-10-23 09:15:00 | 469.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-10-27 11:00:00 | 462.90 | 2025-10-28 14:15:00 | 468.65 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-27 13:15:00 | 462.20 | 2025-10-28 14:15:00 | 468.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-28 10:45:00 | 458.25 | 2025-10-28 14:15:00 | 468.65 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest1 | 2025-10-30 09:15:00 | 477.10 | 2025-10-30 13:15:00 | 469.75 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-11-04 11:30:00 | 456.75 | 2025-11-07 10:15:00 | 433.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 456.75 | 2025-11-10 09:15:00 | 445.70 | STOP_HIT | 0.50 | 2.42% |
| SELL | retest2 | 2025-11-10 12:00:00 | 455.15 | 2025-11-10 12:15:00 | 458.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-11-11 13:15:00 | 450.05 | 2025-11-19 10:15:00 | 460.85 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2025-11-12 14:00:00 | 450.25 | 2025-11-19 10:15:00 | 460.85 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-11-12 14:30:00 | 450.10 | 2025-11-19 10:15:00 | 460.85 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2025-11-13 09:15:00 | 453.30 | 2025-11-19 10:15:00 | 460.85 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2025-12-08 11:00:00 | 440.65 | 2025-12-09 09:15:00 | 418.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 11:00:00 | 440.65 | 2025-12-10 09:15:00 | 431.35 | STOP_HIT | 0.50 | 2.11% |
| SELL | retest2 | 2025-12-10 13:00:00 | 441.15 | 2025-12-10 13:15:00 | 444.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-06 09:15:00 | 437.10 | 2026-01-16 09:15:00 | 425.00 | STOP_HIT | 1.00 | 2.77% |
| SELL | retest2 | 2026-01-07 15:15:00 | 437.10 | 2026-01-16 09:15:00 | 425.00 | STOP_HIT | 1.00 | 2.77% |
| SELL | retest2 | 2026-01-19 09:15:00 | 418.50 | 2026-01-21 10:15:00 | 397.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 418.50 | 2026-01-22 09:15:00 | 406.50 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2026-01-20 09:15:00 | 412.10 | 2026-01-23 13:15:00 | 410.65 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2026-02-01 14:15:00 | 424.05 | 2026-02-01 14:15:00 | 418.40 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-03 09:15:00 | 429.40 | 2026-02-10 14:15:00 | 472.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-27 09:15:00 | 462.15 | 2026-03-02 12:15:00 | 458.25 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-03-18 14:30:00 | 475.90 | 2026-03-19 13:15:00 | 484.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-18 15:00:00 | 475.25 | 2026-03-19 13:15:00 | 484.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-03-30 09:15:00 | 503.95 | 2026-04-02 09:15:00 | 485.00 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2026-04-01 09:15:00 | 504.70 | 2026-04-02 09:15:00 | 485.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-04-09 13:00:00 | 482.75 | 2026-04-15 10:15:00 | 484.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-04-15 09:45:00 | 482.95 | 2026-04-15 10:15:00 | 484.00 | STOP_HIT | 1.00 | -0.22% |
