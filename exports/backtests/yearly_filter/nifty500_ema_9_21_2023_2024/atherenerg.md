# Ather Energy Ltd. (ATHERENERG)

## Backtest Summary

- **Window:** 2025-05-06 09:15:00 → 2026-05-11 15:15:00 (1759 bars)
- **Last close:** 963.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 84 |
| ALERT1 | 48 |
| ALERT2 | 46 |
| ALERT2_SKIP | 24 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 51 |
| PARTIAL | 6 |
| TARGET_HIT | 11 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 40
- **Target hits / Stop hits / Partials:** 11 / 44 / 6
- **Avg / median % per leg:** 1.19% / -0.93%
- **Sum % (uncompounded):** 72.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 9 | 27.3% | 8 | 24 | 1 | 1.19% | 39.3% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.75% | 8.7% |
| BUY @ 3rd Alert (retest2) | 28 | 7 | 25.0% | 7 | 21 | 0 | 1.09% | 30.6% |
| SELL (all) | 28 | 12 | 42.9% | 3 | 20 | 5 | 1.19% | 33.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 12 | 42.9% | 3 | 20 | 5 | 1.19% | 33.3% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.75% | 8.7% |
| retest2 (combined) | 56 | 19 | 33.9% | 10 | 41 | 5 | 1.14% | 63.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 298.60 | 308.68 | 309.82 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 311.25 | 307.00 | 306.87 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 304.40 | 306.94 | 307.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 13:15:00 | 303.60 | 306.27 | 306.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 13:15:00 | 302.95 | 302.69 | 304.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 13:15:00 | 302.95 | 302.69 | 304.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 302.95 | 302.69 | 304.36 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 306.05 | 304.56 | 304.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 316.60 | 306.96 | 305.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 314.25 | 315.02 | 311.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 314.25 | 315.02 | 311.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 314.85 | 319.46 | 317.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 314.85 | 319.46 | 317.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 314.00 | 318.37 | 317.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 318.70 | 318.37 | 317.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 312.40 | 316.35 | 316.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 312.40 | 316.35 | 316.60 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 317.10 | 316.41 | 316.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 318.15 | 316.76 | 316.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 317.50 | 319.00 | 318.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 15:15:00 | 317.50 | 319.00 | 318.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 317.50 | 319.00 | 318.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 315.30 | 319.00 | 318.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 315.25 | 318.25 | 317.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 315.50 | 318.25 | 317.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 315.95 | 317.79 | 317.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 315.65 | 317.79 | 317.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 316.50 | 317.53 | 317.60 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 319.70 | 317.83 | 317.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 321.60 | 318.59 | 318.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 318.70 | 319.11 | 318.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 318.70 | 319.11 | 318.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 318.70 | 319.11 | 318.42 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 316.60 | 317.94 | 318.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 15:15:00 | 314.95 | 317.34 | 317.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 313.80 | 313.50 | 314.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 313.80 | 313.50 | 314.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 313.80 | 313.50 | 314.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 313.80 | 313.50 | 314.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 314.75 | 313.75 | 314.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 314.25 | 313.75 | 314.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 315.00 | 314.00 | 314.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 315.20 | 314.00 | 314.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 315.00 | 314.20 | 314.72 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 319.00 | 315.41 | 315.20 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 314.25 | 315.18 | 315.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 313.90 | 314.85 | 315.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 12:15:00 | 309.10 | 309.06 | 310.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-06 13:00:00 | 309.10 | 309.06 | 310.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 309.25 | 309.20 | 310.11 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 313.90 | 310.71 | 310.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 322.45 | 314.27 | 312.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 316.80 | 317.93 | 315.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 316.80 | 317.93 | 315.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 316.80 | 317.93 | 315.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 315.45 | 317.93 | 315.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 315.25 | 317.40 | 315.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 314.50 | 317.40 | 315.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 315.25 | 316.97 | 315.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 315.15 | 316.97 | 315.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 314.00 | 316.37 | 315.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 314.00 | 316.37 | 315.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 311.75 | 314.73 | 314.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 311.40 | 313.14 | 313.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 312.50 | 312.06 | 312.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 312.50 | 312.06 | 312.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 312.50 | 312.06 | 312.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 312.50 | 312.06 | 312.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 313.00 | 312.25 | 312.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 314.65 | 312.25 | 312.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 314.40 | 312.68 | 313.04 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 316.70 | 313.49 | 313.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 11:15:00 | 319.45 | 314.68 | 313.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 322.30 | 323.15 | 320.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:00:00 | 322.30 | 323.15 | 320.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 320.40 | 322.71 | 321.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 319.05 | 322.71 | 321.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 319.70 | 322.11 | 320.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 319.90 | 322.11 | 320.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 319.05 | 320.28 | 320.31 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 321.00 | 319.85 | 319.84 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 319.40 | 319.76 | 319.80 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 15:15:00 | 320.00 | 319.30 | 319.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 326.70 | 320.78 | 319.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 335.75 | 336.32 | 331.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 14:45:00 | 335.90 | 336.32 | 331.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 332.85 | 336.46 | 334.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 332.30 | 336.46 | 334.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 332.50 | 335.67 | 334.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 336.35 | 335.67 | 334.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 331.20 | 334.78 | 333.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 331.20 | 334.78 | 333.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 332.10 | 334.24 | 333.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 330.70 | 334.24 | 333.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 329.70 | 332.73 | 333.13 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 338.55 | 333.91 | 333.51 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 332.40 | 334.82 | 335.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 330.45 | 333.95 | 334.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 332.35 | 332.33 | 333.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 332.35 | 332.33 | 333.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 332.35 | 332.33 | 333.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 332.65 | 332.33 | 333.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 330.00 | 329.73 | 331.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:00:00 | 326.10 | 328.94 | 330.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 325.25 | 327.62 | 328.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 333.75 | 329.05 | 328.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 333.75 | 329.05 | 328.88 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 329.00 | 329.29 | 329.30 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 329.85 | 329.40 | 329.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 12:15:00 | 331.05 | 329.73 | 329.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 343.65 | 343.70 | 340.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:45:00 | 343.05 | 343.70 | 340.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 343.40 | 343.30 | 341.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 340.80 | 343.30 | 341.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 341.95 | 342.91 | 341.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 342.05 | 342.91 | 341.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 340.25 | 342.38 | 341.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 340.25 | 342.38 | 341.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 339.00 | 341.70 | 341.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 347.50 | 341.70 | 341.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 341.50 | 344.88 | 345.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 341.50 | 344.88 | 345.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 338.95 | 343.29 | 344.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 339.75 | 338.57 | 340.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 339.75 | 338.57 | 340.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 339.75 | 338.57 | 340.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 335.05 | 337.60 | 339.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 335.70 | 337.28 | 339.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:45:00 | 335.95 | 335.46 | 337.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 333.80 | 335.71 | 337.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 335.00 | 335.57 | 336.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 332.65 | 334.30 | 335.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 338.55 | 335.05 | 335.50 | SL hit (close>static) qty=1.00 sl=337.70 alert=retest2 |

### Cycle 26 — BUY (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 12:15:00 | 340.15 | 336.07 | 335.92 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 330.65 | 335.28 | 335.64 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 12:15:00 | 338.25 | 336.00 | 335.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 341.75 | 338.08 | 336.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 338.50 | 341.89 | 340.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 338.50 | 341.89 | 340.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 338.50 | 341.89 | 340.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 338.40 | 341.89 | 340.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 340.50 | 341.61 | 340.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 346.30 | 342.55 | 340.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 344.95 | 343.77 | 341.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 346.35 | 344.31 | 342.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-04 14:15:00 | 380.93 | 364.54 | 356.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 400.65 | 408.66 | 409.14 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 415.00 | 410.14 | 409.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 419.10 | 415.19 | 412.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 412.85 | 415.65 | 414.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 11:15:00 | 412.85 | 415.65 | 414.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 412.85 | 415.65 | 414.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 412.85 | 415.65 | 414.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 413.30 | 415.18 | 414.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 414.35 | 415.18 | 414.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 416.00 | 414.85 | 414.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 417.95 | 415.70 | 414.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 413.25 | 417.40 | 417.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 413.25 | 417.40 | 417.72 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 422.40 | 417.60 | 417.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 429.35 | 419.95 | 418.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 13:15:00 | 419.70 | 419.90 | 418.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 14:00:00 | 419.70 | 419.90 | 418.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 417.85 | 419.49 | 418.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 417.85 | 419.49 | 418.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 417.50 | 419.09 | 418.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 427.45 | 419.09 | 418.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 419.00 | 418.96 | 418.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 425.75 | 421.21 | 419.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-01 09:15:00 | 470.20 | 449.25 | 440.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 575.80 | 578.45 | 578.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 569.95 | 576.75 | 577.87 | Break + close below crossover candle low |

### Cycle 34 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 587.45 | 578.89 | 578.74 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 557.55 | 576.37 | 578.07 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 579.40 | 560.17 | 559.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 594.85 | 582.70 | 574.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 588.00 | 591.25 | 583.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 593.70 | 591.25 | 583.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 596.00 | 592.20 | 584.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 601.90 | 597.00 | 588.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-08 10:15:00 | 662.09 | 632.10 | 618.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 601.10 | 623.92 | 625.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 11:15:00 | 598.00 | 615.18 | 620.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 617.00 | 606.34 | 613.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 617.00 | 606.34 | 613.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 617.00 | 606.34 | 613.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 617.00 | 606.34 | 613.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 619.40 | 608.95 | 613.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 616.35 | 608.95 | 613.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 618.85 | 610.93 | 614.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 619.70 | 610.93 | 614.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 612.00 | 611.75 | 613.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:30:00 | 611.30 | 612.16 | 613.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 611.45 | 612.16 | 613.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 611.10 | 611.71 | 613.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:00:00 | 609.90 | 610.78 | 612.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 608.30 | 609.44 | 611.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 608.30 | 609.44 | 611.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 613.10 | 610.17 | 611.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 624.40 | 610.17 | 611.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 617.00 | 611.54 | 612.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 617.00 | 611.54 | 612.28 | SL hit (close>static) qty=1.00 sl=616.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 621.00 | 613.43 | 613.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 622.15 | 616.38 | 614.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 683.00 | 687.69 | 670.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:15:00 | 702.35 | 688.89 | 672.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-21 13:15:00 | 737.47 | 718.76 | 697.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-10-21 14:15:00 | 772.59 | 731.71 | 704.86 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 39 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 705.85 | 713.27 | 713.94 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 717.75 | 714.42 | 714.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 735.35 | 718.61 | 716.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 14:15:00 | 727.20 | 728.26 | 721.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 727.20 | 728.26 | 721.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 725.50 | 735.06 | 731.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 723.75 | 735.06 | 731.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 718.50 | 731.75 | 730.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 718.50 | 731.75 | 730.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 711.00 | 727.60 | 728.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 705.90 | 719.70 | 724.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 706.25 | 702.00 | 711.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:45:00 | 708.75 | 702.00 | 711.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 705.00 | 703.88 | 710.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 700.65 | 703.88 | 710.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 698.70 | 701.32 | 707.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 698.80 | 696.48 | 699.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 665.62 | 688.59 | 694.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 15:15:00 | 663.76 | 684.83 | 692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 15:15:00 | 663.86 | 684.83 | 692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-06 09:15:00 | 630.59 | 673.33 | 686.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 42 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 660.20 | 649.44 | 648.34 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 638.55 | 646.80 | 647.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 11:15:00 | 632.65 | 643.97 | 646.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 648.00 | 639.34 | 642.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 648.00 | 639.34 | 642.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 648.00 | 639.34 | 642.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 652.10 | 639.34 | 642.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 641.45 | 639.77 | 642.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:30:00 | 634.25 | 639.31 | 641.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 15:00:00 | 634.60 | 637.03 | 640.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 659.40 | 642.09 | 641.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 659.40 | 642.09 | 641.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 11:15:00 | 672.30 | 654.08 | 649.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 688.05 | 690.12 | 681.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 688.05 | 690.12 | 681.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 687.25 | 693.56 | 688.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 680.85 | 693.56 | 688.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 677.05 | 690.26 | 687.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 678.80 | 690.26 | 687.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 671.60 | 686.53 | 685.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:00:00 | 671.60 | 686.53 | 685.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 673.30 | 683.88 | 684.70 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 689.20 | 683.63 | 682.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 692.65 | 685.44 | 683.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 701.40 | 704.84 | 698.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 704.10 | 704.84 | 698.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 696.70 | 703.21 | 698.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 699.10 | 703.21 | 698.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 693.60 | 701.29 | 698.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 694.65 | 701.29 | 698.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 720.00 | 706.24 | 701.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 701.60 | 706.24 | 701.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 704.85 | 709.34 | 705.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 704.85 | 709.34 | 705.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 711.90 | 709.86 | 706.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:15:00 | 705.80 | 709.86 | 706.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 705.80 | 709.04 | 706.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 695.50 | 709.04 | 706.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 689.00 | 705.04 | 704.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 689.00 | 705.04 | 704.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 690.55 | 702.14 | 703.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 687.00 | 695.17 | 699.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 685.10 | 683.87 | 689.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:15:00 | 686.25 | 683.87 | 689.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 690.75 | 685.24 | 689.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 690.75 | 685.24 | 689.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 686.60 | 685.51 | 689.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 691.25 | 685.51 | 689.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 676.05 | 675.38 | 679.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 678.00 | 675.38 | 679.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 679.95 | 673.35 | 675.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 679.10 | 673.35 | 675.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 675.00 | 673.68 | 675.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 662.15 | 673.68 | 675.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 666.90 | 653.41 | 652.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 11:15:00 | 666.90 | 653.41 | 652.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 686.60 | 664.81 | 658.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 674.65 | 675.87 | 668.88 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:15:00 | 683.50 | 676.18 | 669.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 10:15:00 | 679.20 | 676.35 | 670.32 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 11:00:00 | 681.05 | 677.29 | 671.30 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 667.05 | 674.51 | 671.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 667.05 | 674.51 | 671.03 | SL hit (close<ema400) qty=1.00 sl=671.03 alert=retest1 |

### Cycle 49 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 661.00 | 668.33 | 668.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 652.15 | 663.43 | 666.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 663.05 | 660.99 | 664.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 663.05 | 660.99 | 664.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 663.05 | 660.99 | 664.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 663.05 | 660.99 | 664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 663.00 | 661.39 | 663.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 659.40 | 661.39 | 663.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 669.70 | 663.06 | 664.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 671.15 | 663.06 | 664.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 686.90 | 667.82 | 666.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 702.65 | 681.82 | 674.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 698.40 | 698.67 | 686.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 698.40 | 698.67 | 686.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 700.20 | 700.02 | 689.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 700.20 | 700.02 | 689.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 694.50 | 698.91 | 690.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 693.00 | 698.91 | 690.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 699.50 | 699.03 | 690.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 690.30 | 699.03 | 690.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 723.80 | 719.96 | 710.80 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 704.10 | 710.89 | 711.55 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 720.25 | 712.23 | 712.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 744.60 | 719.63 | 715.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 746.45 | 749.19 | 740.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 746.45 | 749.19 | 740.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 737.80 | 746.86 | 741.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:00:00 | 737.80 | 746.86 | 741.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 742.10 | 745.91 | 741.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:15:00 | 732.95 | 745.91 | 741.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 732.95 | 743.32 | 741.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 749.55 | 743.32 | 741.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 742.95 | 743.48 | 741.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:45:00 | 743.00 | 743.38 | 741.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 736.40 | 740.98 | 741.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 736.40 | 740.98 | 741.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 12:15:00 | 713.30 | 734.35 | 738.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 700.20 | 699.45 | 712.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 15:00:00 | 700.20 | 699.45 | 712.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 657.50 | 638.78 | 648.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 653.45 | 638.78 | 648.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 658.75 | 642.77 | 649.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 651.25 | 644.47 | 649.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 645.20 | 646.82 | 648.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 13:15:00 | 618.69 | 629.78 | 637.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 13:15:00 | 612.94 | 629.78 | 637.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 627.40 | 619.30 | 626.29 | SL hit (close>ema200) qty=0.50 sl=619.30 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 630.00 | 617.70 | 616.10 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 612.80 | 617.32 | 617.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 608.35 | 614.71 | 616.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 619.80 | 605.61 | 608.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 619.80 | 605.61 | 608.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 619.80 | 605.61 | 608.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 619.80 | 605.61 | 608.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 626.20 | 609.73 | 609.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 631.65 | 621.53 | 616.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 623.10 | 623.94 | 619.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 619.45 | 623.94 | 619.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 616.70 | 622.49 | 619.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 616.70 | 622.49 | 619.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 613.35 | 620.67 | 618.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 613.35 | 620.67 | 618.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 605.75 | 617.68 | 617.71 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 662.05 | 622.69 | 619.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 677.00 | 633.55 | 624.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 682.55 | 686.15 | 668.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 682.55 | 686.15 | 668.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 716.70 | 724.77 | 719.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 714.30 | 724.77 | 719.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 719.20 | 723.65 | 719.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 718.85 | 723.65 | 719.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 717.30 | 722.38 | 719.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 717.25 | 722.38 | 719.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 716.00 | 721.11 | 718.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 728.90 | 721.11 | 718.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 729.75 | 722.47 | 720.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 694.30 | 718.20 | 719.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 694.30 | 718.20 | 719.23 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 726.65 | 714.50 | 713.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 745.50 | 728.64 | 723.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 721.50 | 728.79 | 724.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 721.50 | 728.79 | 724.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 721.50 | 728.79 | 724.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 721.50 | 728.79 | 724.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 720.00 | 727.03 | 724.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 720.95 | 727.03 | 724.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 718.45 | 725.31 | 723.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 719.30 | 725.31 | 723.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 733.00 | 728.57 | 725.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 725.00 | 728.57 | 725.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 721.05 | 727.06 | 725.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:00:00 | 721.05 | 727.06 | 725.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 714.20 | 724.49 | 724.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 714.20 | 724.49 | 724.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 11:15:00 | 717.45 | 723.08 | 723.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 707.85 | 717.85 | 720.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 718.30 | 713.54 | 716.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 718.30 | 713.54 | 716.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 718.30 | 713.54 | 716.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 718.30 | 713.54 | 716.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 718.15 | 714.46 | 716.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 710.00 | 714.46 | 716.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 726.85 | 708.27 | 709.94 | SL hit (close>static) qty=1.00 sl=719.20 alert=retest2 |

### Cycle 62 — BUY (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 15:15:00 | 711.05 | 702.12 | 701.85 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 694.30 | 700.55 | 701.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 679.15 | 691.27 | 696.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 707.05 | 692.62 | 695.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 707.05 | 692.62 | 695.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 707.05 | 692.62 | 695.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 707.05 | 692.62 | 695.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 707.00 | 695.50 | 696.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 683.20 | 695.50 | 696.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 691.15 | 680.20 | 679.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 691.15 | 680.20 | 679.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 701.80 | 693.45 | 688.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 688.30 | 693.79 | 689.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 688.30 | 693.79 | 689.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 688.30 | 693.79 | 689.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 686.75 | 693.79 | 689.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 692.05 | 693.44 | 689.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 695.00 | 693.45 | 690.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 697.45 | 693.45 | 690.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 700.50 | 702.95 | 700.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:45:00 | 694.00 | 699.28 | 698.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 12:15:00 | 692.25 | 697.88 | 698.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 692.25 | 697.88 | 698.18 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 709.20 | 700.32 | 699.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 726.00 | 715.43 | 708.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 755.60 | 756.13 | 742.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 755.60 | 756.13 | 742.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 758.00 | 768.91 | 758.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 758.00 | 768.91 | 758.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 722.95 | 759.72 | 755.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 724.20 | 759.72 | 755.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 743.00 | 756.38 | 754.25 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 737.65 | 752.63 | 752.74 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 757.70 | 752.39 | 752.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 779.50 | 760.08 | 756.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 769.40 | 773.64 | 766.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 769.40 | 773.64 | 766.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 769.40 | 773.64 | 766.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 785.55 | 774.51 | 768.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 758.35 | 773.89 | 772.90 | SL hit (close<static) qty=1.00 sl=759.05 alert=retest2 |

### Cycle 69 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 762.20 | 771.55 | 771.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 747.00 | 764.48 | 768.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 769.20 | 765.43 | 768.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 769.20 | 765.43 | 768.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 769.20 | 765.43 | 768.58 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 775.05 | 770.91 | 770.52 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 763.35 | 769.17 | 769.81 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 785.00 | 772.06 | 770.35 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 765.85 | 773.75 | 774.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 15:15:00 | 759.00 | 770.80 | 772.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 782.55 | 773.15 | 773.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 782.55 | 773.15 | 773.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 782.55 | 773.15 | 773.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 782.30 | 773.15 | 773.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 786.40 | 775.80 | 774.73 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 14:15:00 | 771.60 | 773.97 | 774.15 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 807.95 | 780.85 | 777.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 14:15:00 | 816.00 | 798.76 | 787.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 10:15:00 | 889.45 | 898.73 | 873.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 11:00:00 | 889.45 | 898.73 | 873.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 875.95 | 890.46 | 875.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 875.95 | 890.46 | 875.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 875.60 | 887.49 | 875.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:15:00 | 872.00 | 887.49 | 875.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 872.00 | 884.39 | 875.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 878.05 | 884.39 | 875.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 15:15:00 | 867.95 | 871.93 | 872.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 15:15:00 | 867.95 | 871.93 | 872.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 10:15:00 | 861.05 | 869.38 | 870.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 09:15:00 | 888.55 | 863.10 | 865.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 888.55 | 863.10 | 865.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 888.55 | 863.10 | 865.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 888.55 | 863.10 | 865.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 891.75 | 868.83 | 867.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 901.00 | 875.26 | 870.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 886.40 | 887.16 | 879.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 886.40 | 887.16 | 879.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 886.40 | 887.16 | 879.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 878.10 | 887.16 | 879.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 906.65 | 899.12 | 891.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:30:00 | 910.30 | 901.25 | 893.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 934.55 | 903.47 | 896.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 881.35 | 899.09 | 899.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 881.35 | 899.09 | 899.46 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 927.70 | 897.84 | 896.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 933.25 | 914.95 | 905.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 925.70 | 925.77 | 914.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 925.70 | 925.77 | 914.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 926.60 | 935.27 | 929.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 926.60 | 935.27 | 929.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 928.55 | 933.92 | 929.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 930.15 | 933.92 | 929.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 929.15 | 932.97 | 929.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 929.15 | 932.97 | 929.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 933.25 | 933.03 | 929.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 935.80 | 932.68 | 929.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:30:00 | 935.45 | 932.94 | 930.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 934.95 | 932.94 | 930.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 921.45 | 930.65 | 929.90 | SL hit (close<static) qty=1.00 sl=926.00 alert=retest2 |

### Cycle 81 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 919.85 | 928.49 | 928.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 904.95 | 923.78 | 926.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 929.20 | 924.40 | 926.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 14:15:00 | 929.20 | 924.40 | 926.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 929.20 | 924.40 | 926.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 929.20 | 924.40 | 926.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 937.90 | 927.10 | 927.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 967.40 | 927.10 | 927.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 917.95 | 923.34 | 925.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 919.55 | 923.34 | 925.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 937.20 | 926.11 | 926.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 947.55 | 926.11 | 926.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 939.65 | 928.82 | 927.88 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 894.85 | 924.89 | 927.01 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 928.65 | 915.19 | 914.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-11 09:15:00 | 947.65 | 923.61 | 919.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 09:15:00 | 318.70 | 2025-05-22 13:15:00 | 312.40 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-07-08 12:00:00 | 326.10 | 2025-07-10 09:15:00 | 333.75 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-07-09 09:45:00 | 325.25 | 2025-07-10 09:15:00 | 333.75 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-07-17 09:15:00 | 347.50 | 2025-07-18 14:15:00 | 341.50 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-07-22 13:15:00 | 335.05 | 2025-07-25 11:15:00 | 338.55 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-23 09:15:00 | 335.70 | 2025-07-25 12:15:00 | 340.15 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-23 14:45:00 | 335.95 | 2025-07-25 12:15:00 | 340.15 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-07-24 09:15:00 | 333.80 | 2025-07-25 12:15:00 | 340.15 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-07-24 13:30:00 | 332.65 | 2025-07-25 12:15:00 | 340.15 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-07-30 15:00:00 | 346.30 | 2025-08-04 14:15:00 | 380.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 09:30:00 | 344.95 | 2025-08-04 14:15:00 | 379.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 11:45:00 | 346.35 | 2025-08-04 14:15:00 | 380.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 13:30:00 | 414.35 | 2025-08-21 09:15:00 | 413.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-18 15:15:00 | 416.00 | 2025-08-21 09:15:00 | 413.25 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-19 12:00:00 | 417.95 | 2025-08-21 09:15:00 | 413.25 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-25 09:15:00 | 427.45 | 2025-09-01 09:15:00 | 470.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-25 15:15:00 | 419.00 | 2025-09-01 09:15:00 | 460.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 09:45:00 | 425.75 | 2025-09-01 09:15:00 | 468.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-06 11:30:00 | 601.90 | 2025-10-08 10:15:00 | 662.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-13 14:30:00 | 611.30 | 2025-10-14 15:15:00 | 617.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-13 15:15:00 | 611.45 | 2025-10-14 15:15:00 | 617.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-14 09:45:00 | 611.10 | 2025-10-14 15:15:00 | 617.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-14 12:00:00 | 609.90 | 2025-10-14 15:15:00 | 617.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest1 | 2025-10-20 10:15:00 | 702.35 | 2025-10-21 13:15:00 | 737.47 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-20 10:15:00 | 702.35 | 2025-10-21 14:15:00 | 772.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 10:15:00 | 700.65 | 2025-11-04 14:15:00 | 665.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 12:30:00 | 698.70 | 2025-11-04 15:15:00 | 663.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:15:00 | 698.80 | 2025-11-04 15:15:00 | 663.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:15:00 | 700.65 | 2025-11-06 09:15:00 | 630.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 12:30:00 | 698.70 | 2025-11-06 09:15:00 | 628.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 10:15:00 | 698.80 | 2025-11-06 09:15:00 | 628.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-13 11:30:00 | 634.25 | 2025-11-14 12:15:00 | 659.40 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-11-13 15:00:00 | 634.60 | 2025-11-14 12:15:00 | 659.40 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-12-09 09:15:00 | 662.15 | 2025-12-15 11:15:00 | 666.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2025-12-17 09:15:00 | 683.50 | 2025-12-17 12:15:00 | 667.05 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest1 | 2025-12-17 10:15:00 | 679.20 | 2025-12-17 12:15:00 | 667.05 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest1 | 2025-12-17 11:00:00 | 681.05 | 2025-12-17 12:15:00 | 667.05 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-02 09:15:00 | 749.55 | 2026-01-05 10:15:00 | 736.40 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-02 12:15:00 | 742.95 | 2026-01-05 10:15:00 | 736.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-02 12:45:00 | 743.00 | 2026-01-05 10:15:00 | 736.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-01-13 12:00:00 | 651.25 | 2026-01-16 13:15:00 | 618.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 645.20 | 2026-01-16 13:15:00 | 612.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 651.25 | 2026-01-19 14:15:00 | 627.40 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2026-01-14 09:15:00 | 645.20 | 2026-01-19 14:15:00 | 627.40 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest2 | 2026-02-12 09:15:00 | 728.90 | 2026-02-13 09:15:00 | 694.30 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2026-02-12 12:45:00 | 729.75 | 2026-02-13 09:15:00 | 694.30 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2026-02-24 09:15:00 | 710.00 | 2026-02-25 10:15:00 | 726.85 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-25 11:45:00 | 714.15 | 2026-02-27 15:15:00 | 711.05 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2026-02-25 12:30:00 | 713.05 | 2026-02-27 15:15:00 | 711.05 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2026-03-04 09:15:00 | 683.20 | 2026-03-10 11:15:00 | 691.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-03-12 11:45:00 | 695.00 | 2026-03-16 12:15:00 | 692.25 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-03-12 12:15:00 | 697.45 | 2026-03-16 12:15:00 | 692.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-03-16 09:30:00 | 700.50 | 2026-03-16 12:15:00 | 692.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-03-16 11:45:00 | 694.00 | 2026-03-16 12:15:00 | 692.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-03-27 13:15:00 | 785.55 | 2026-03-30 12:15:00 | 758.35 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2026-04-16 09:15:00 | 878.05 | 2026-04-16 15:15:00 | 867.95 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-04-22 12:30:00 | 910.30 | 2026-04-24 09:15:00 | 881.35 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-23 09:15:00 | 934.55 | 2026-04-24 09:15:00 | 881.35 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest2 | 2026-04-30 13:30:00 | 935.80 | 2026-05-04 10:15:00 | 921.45 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-04-30 14:30:00 | 935.45 | 2026-05-04 10:15:00 | 921.45 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-30 15:15:00 | 934.95 | 2026-05-04 10:15:00 | 921.45 | STOP_HIT | 1.00 | -1.44% |
