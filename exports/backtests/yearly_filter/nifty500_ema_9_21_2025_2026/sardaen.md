# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 591.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 87 |
| ALERT1 | 60 |
| ALERT2 | 59 |
| ALERT2_SKIP | 37 |
| ALERT3 | 158 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 38
- **Target hits / Stop hits / Partials:** 1 / 58 / 7
- **Avg / median % per leg:** 0.57% / -0.45%
- **Sum % (uncompounded):** 37.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 10 | 35.7% | 1 | 27 | 0 | -0.34% | -9.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 10 | 35.7% | 1 | 27 | 0 | -0.34% | -9.5% |
| SELL (all) | 38 | 18 | 47.4% | 0 | 31 | 7 | 1.23% | 46.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 18 | 47.4% | 0 | 31 | 7 | 1.23% | 46.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 66 | 28 | 42.4% | 1 | 58 | 7 | 0.57% | 37.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 441.45 | 430.81 | 429.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 443.15 | 433.27 | 430.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 438.00 | 440.90 | 436.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 438.00 | 440.90 | 436.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 450.05 | 442.73 | 438.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 452.20 | 444.17 | 439.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:00:00 | 451.70 | 447.28 | 441.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 450.45 | 447.92 | 442.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 450.45 | 448.42 | 443.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 454.80 | 454.22 | 451.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 453.25 | 454.22 | 451.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 471.70 | 469.42 | 465.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 469.65 | 469.42 | 465.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 468.50 | 472.43 | 470.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:00:00 | 468.50 | 472.43 | 470.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 468.60 | 471.66 | 470.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 474.60 | 471.92 | 470.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 439.45 | 464.63 | 467.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 439.45 | 464.63 | 467.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 433.80 | 442.96 | 451.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 437.90 | 437.43 | 445.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 437.90 | 437.43 | 445.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 437.90 | 437.43 | 445.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 438.45 | 437.43 | 445.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 437.75 | 436.40 | 440.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 439.00 | 436.40 | 440.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 434.75 | 435.80 | 438.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 433.80 | 436.65 | 437.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:30:00 | 434.00 | 435.32 | 436.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 439.80 | 437.11 | 436.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 439.80 | 437.11 | 436.99 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 436.65 | 436.96 | 436.99 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 437.80 | 437.15 | 437.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 439.90 | 437.81 | 437.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 442.30 | 443.25 | 440.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 13:30:00 | 442.65 | 443.25 | 440.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 442.10 | 443.02 | 440.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 443.50 | 442.51 | 441.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 438.75 | 441.69 | 440.93 | SL hit (close<static) qty=1.00 sl=440.90 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 436.40 | 439.75 | 440.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 15:15:00 | 435.80 | 438.96 | 439.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 440.55 | 439.28 | 439.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 440.55 | 439.28 | 439.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 440.55 | 439.28 | 439.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:30:00 | 440.85 | 439.28 | 439.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 438.90 | 439.20 | 439.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:15:00 | 440.55 | 439.20 | 439.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 441.65 | 439.69 | 439.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 441.65 | 439.69 | 439.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 448.00 | 441.35 | 440.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 451.40 | 446.50 | 443.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 448.10 | 449.00 | 445.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:00:00 | 448.10 | 449.00 | 445.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 453.25 | 453.02 | 449.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 453.40 | 453.02 | 449.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 453.05 | 453.32 | 450.54 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 449.35 | 450.51 | 450.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 447.40 | 449.89 | 450.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 449.40 | 449.11 | 449.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 449.40 | 449.11 | 449.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 449.40 | 449.11 | 449.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 449.40 | 449.11 | 449.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 448.20 | 448.93 | 449.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 448.35 | 448.93 | 449.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 448.90 | 448.92 | 449.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 448.90 | 448.92 | 449.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 448.80 | 448.90 | 449.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 449.65 | 448.90 | 449.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 455.00 | 450.12 | 449.89 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 448.05 | 449.80 | 449.90 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 452.15 | 450.30 | 450.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 10:15:00 | 457.60 | 452.07 | 451.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 449.05 | 451.47 | 450.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 11:15:00 | 449.05 | 451.47 | 450.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 449.05 | 451.47 | 450.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 448.00 | 451.47 | 450.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 446.30 | 450.44 | 450.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 13:15:00 | 444.10 | 449.17 | 449.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 436.65 | 431.51 | 435.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 436.65 | 431.51 | 435.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 436.65 | 431.51 | 435.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 438.40 | 431.51 | 435.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 442.50 | 433.71 | 435.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 442.50 | 433.71 | 435.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 444.95 | 437.90 | 437.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 447.05 | 440.84 | 439.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 440.00 | 441.74 | 440.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 12:15:00 | 440.00 | 441.74 | 440.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 440.00 | 441.74 | 440.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:00:00 | 440.00 | 441.74 | 440.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 438.60 | 441.11 | 439.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 438.60 | 441.11 | 439.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 439.00 | 440.69 | 439.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 440.00 | 440.69 | 439.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 440.00 | 440.55 | 439.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 447.00 | 440.55 | 439.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:45:00 | 441.45 | 440.99 | 440.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 437.80 | 443.81 | 444.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 437.80 | 443.81 | 444.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 437.05 | 442.46 | 443.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 433.50 | 433.24 | 435.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:00:00 | 433.50 | 433.24 | 435.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 432.60 | 433.14 | 434.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:45:00 | 430.55 | 432.45 | 434.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 430.50 | 432.06 | 433.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 430.80 | 431.66 | 433.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 430.80 | 427.00 | 427.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 434.85 | 428.57 | 428.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 434.85 | 428.57 | 428.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 441.25 | 431.64 | 429.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 448.20 | 448.79 | 442.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:30:00 | 446.80 | 448.79 | 442.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 444.10 | 447.77 | 444.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 444.00 | 447.77 | 444.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 440.00 | 446.22 | 443.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 14:15:00 | 445.10 | 443.99 | 443.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 445.50 | 443.84 | 443.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:45:00 | 447.20 | 444.18 | 443.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:45:00 | 444.85 | 445.52 | 444.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 444.40 | 445.30 | 444.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 445.25 | 445.30 | 444.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 446.80 | 445.60 | 444.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 449.40 | 445.63 | 445.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 443.95 | 445.30 | 444.92 | SL hit (close<static) qty=1.00 sl=444.05 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 442.85 | 444.46 | 444.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 440.85 | 443.66 | 444.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 12:15:00 | 444.35 | 442.90 | 443.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 12:15:00 | 444.35 | 442.90 | 443.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 444.35 | 442.90 | 443.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 444.35 | 442.90 | 443.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 442.70 | 442.86 | 443.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 441.00 | 442.82 | 443.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 447.35 | 443.43 | 443.61 | SL hit (close>static) qty=1.00 sl=444.35 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 454.90 | 445.73 | 444.64 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 443.75 | 448.84 | 448.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 438.85 | 446.85 | 447.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 435.75 | 433.99 | 438.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 436.25 | 433.99 | 438.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 447.65 | 437.32 | 438.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 447.65 | 437.32 | 438.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 442.75 | 438.41 | 438.96 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 446.20 | 439.97 | 439.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 447.25 | 442.25 | 440.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 443.75 | 445.31 | 443.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 443.75 | 445.31 | 443.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 443.75 | 445.31 | 443.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 451.35 | 446.29 | 444.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:15:00 | 449.65 | 446.29 | 444.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 439.30 | 443.84 | 444.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 439.30 | 443.84 | 444.44 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 512.85 | 456.76 | 450.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 534.55 | 510.72 | 485.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 565.75 | 566.23 | 543.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:00:00 | 565.75 | 566.23 | 543.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 540.20 | 556.33 | 548.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 540.20 | 556.33 | 548.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 540.95 | 553.26 | 547.77 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 535.05 | 543.02 | 544.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 533.05 | 541.03 | 543.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 553.70 | 543.56 | 544.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 553.70 | 543.56 | 544.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 553.70 | 543.56 | 544.03 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 556.50 | 546.15 | 545.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 565.80 | 556.66 | 553.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 558.95 | 559.37 | 556.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 558.95 | 559.37 | 556.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 552.50 | 557.78 | 556.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 552.50 | 557.78 | 556.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 558.45 | 557.91 | 556.36 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 553.70 | 555.70 | 555.75 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 581.60 | 560.88 | 558.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 586.90 | 577.77 | 570.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 581.80 | 585.35 | 578.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 581.80 | 585.35 | 578.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 581.80 | 585.35 | 578.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 581.80 | 585.35 | 578.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 586.95 | 585.67 | 579.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:15:00 | 591.70 | 585.67 | 579.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 591.00 | 595.80 | 595.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 591.00 | 595.80 | 595.91 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 12:15:00 | 598.50 | 596.04 | 595.93 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 592.75 | 595.49 | 595.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 592.00 | 594.79 | 595.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 595.05 | 594.85 | 595.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 595.05 | 594.85 | 595.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 595.05 | 594.85 | 595.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 595.05 | 594.85 | 595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 593.30 | 594.54 | 595.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:30:00 | 595.70 | 594.54 | 595.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 594.75 | 594.58 | 595.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:45:00 | 594.05 | 594.58 | 595.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 589.50 | 593.10 | 594.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:30:00 | 586.90 | 591.48 | 593.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 595.20 | 591.37 | 593.03 | SL hit (close>static) qty=1.00 sl=594.60 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 600.30 | 594.26 | 594.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 606.75 | 598.89 | 596.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 11:15:00 | 596.85 | 598.48 | 596.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 11:15:00 | 596.85 | 598.48 | 596.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 596.85 | 598.48 | 596.69 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 587.90 | 596.80 | 597.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 574.25 | 592.29 | 595.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 587.80 | 587.03 | 591.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:00:00 | 587.80 | 587.03 | 591.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 590.45 | 587.71 | 591.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 590.45 | 587.71 | 591.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 591.50 | 588.47 | 591.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 592.50 | 588.47 | 591.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 587.00 | 588.18 | 591.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 585.30 | 587.73 | 590.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 579.75 | 587.71 | 590.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 15:15:00 | 578.95 | 577.54 | 577.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 578.95 | 577.54 | 577.41 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 572.50 | 576.53 | 576.97 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 582.35 | 577.44 | 577.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 591.65 | 585.02 | 583.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 588.85 | 590.55 | 586.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 588.85 | 590.55 | 586.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 608.10 | 613.61 | 608.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 608.10 | 613.61 | 608.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 607.55 | 612.40 | 608.63 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 600.20 | 606.88 | 606.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 590.85 | 603.68 | 605.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 595.80 | 579.95 | 584.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 595.80 | 579.95 | 584.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 595.80 | 579.95 | 584.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 595.80 | 579.95 | 584.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 589.20 | 581.80 | 585.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 583.40 | 585.65 | 586.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:30:00 | 586.05 | 585.78 | 586.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 587.70 | 586.33 | 586.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 587.65 | 586.39 | 586.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 587.65 | 586.39 | 586.37 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 586.00 | 586.31 | 586.33 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 588.00 | 586.59 | 586.45 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 586.00 | 586.38 | 586.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 582.00 | 585.48 | 585.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 587.05 | 585.22 | 585.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 587.05 | 585.22 | 585.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 587.05 | 585.22 | 585.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 592.15 | 585.22 | 585.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 586.75 | 585.53 | 585.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 585.75 | 585.10 | 585.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:45:00 | 584.65 | 584.55 | 585.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 556.46 | 562.82 | 570.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 555.42 | 562.82 | 570.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 547.45 | 545.49 | 552.07 | SL hit (close>ema200) qty=0.50 sl=545.49 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 573.85 | 557.64 | 555.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 580.85 | 572.22 | 565.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 579.25 | 579.62 | 573.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:45:00 | 578.90 | 579.62 | 573.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 572.60 | 577.67 | 574.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 572.60 | 577.67 | 574.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 572.15 | 576.57 | 574.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 571.00 | 576.57 | 574.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 568.25 | 574.90 | 573.53 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 567.20 | 571.78 | 572.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 565.15 | 570.46 | 571.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 574.50 | 568.22 | 569.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 574.50 | 568.22 | 569.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 574.50 | 568.22 | 569.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 573.30 | 568.22 | 569.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 573.40 | 569.25 | 570.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 573.70 | 569.25 | 570.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 571.45 | 570.04 | 570.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:45:00 | 571.30 | 570.04 | 570.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 571.95 | 570.43 | 570.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 571.95 | 570.43 | 570.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 572.00 | 570.74 | 570.66 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 564.80 | 569.55 | 570.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 557.90 | 565.90 | 567.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 549.00 | 546.03 | 552.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 12:15:00 | 549.25 | 547.34 | 551.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 549.25 | 547.34 | 551.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 549.60 | 547.34 | 551.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 551.50 | 548.40 | 550.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 551.20 | 548.40 | 550.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 551.25 | 548.97 | 550.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 553.00 | 548.97 | 550.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 546.15 | 548.40 | 550.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 544.60 | 548.23 | 549.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:45:00 | 544.40 | 545.45 | 548.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:15:00 | 517.37 | 530.86 | 538.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:15:00 | 517.18 | 530.86 | 538.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 527.00 | 526.31 | 532.25 | SL hit (close>ema200) qty=0.50 sl=526.31 alert=retest2 |

### Cycle 43 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 541.00 | 536.21 | 535.71 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 534.10 | 535.52 | 535.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 530.65 | 534.54 | 535.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 11:15:00 | 538.30 | 535.29 | 535.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 11:15:00 | 538.30 | 535.29 | 535.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 538.30 | 535.29 | 535.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 538.30 | 535.29 | 535.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 540.80 | 536.39 | 535.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 548.75 | 541.55 | 539.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 14:15:00 | 544.30 | 545.27 | 542.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 14:45:00 | 544.20 | 545.27 | 542.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 547.35 | 556.51 | 552.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 547.35 | 556.51 | 552.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 544.55 | 554.11 | 551.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 544.55 | 554.11 | 551.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 537.00 | 548.04 | 549.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 14:15:00 | 535.10 | 543.85 | 546.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 544.00 | 534.62 | 538.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 544.00 | 534.62 | 538.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 544.00 | 534.62 | 538.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 545.55 | 534.62 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 543.15 | 536.32 | 538.64 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 545.85 | 540.83 | 540.36 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 536.00 | 540.01 | 540.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 534.90 | 538.36 | 539.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 529.80 | 529.42 | 532.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:45:00 | 529.00 | 529.42 | 532.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 528.80 | 529.29 | 532.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 532.50 | 529.29 | 532.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 536.85 | 531.07 | 532.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 529.85 | 532.49 | 533.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 530.65 | 531.98 | 532.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 537.90 | 533.88 | 533.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 537.90 | 533.88 | 533.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 543.20 | 536.98 | 535.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 534.70 | 537.17 | 535.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 534.70 | 537.17 | 535.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 534.70 | 537.17 | 535.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 536.50 | 537.17 | 535.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 531.80 | 536.09 | 535.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 531.80 | 536.09 | 535.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 531.90 | 535.26 | 534.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 533.20 | 534.92 | 534.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 531.25 | 534.19 | 534.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 531.25 | 534.19 | 534.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 528.30 | 533.01 | 533.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 12:15:00 | 517.75 | 517.59 | 521.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 517.75 | 517.59 | 521.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 520.50 | 518.17 | 521.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 523.00 | 518.17 | 521.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 519.05 | 518.35 | 521.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:30:00 | 521.35 | 518.35 | 521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 520.50 | 518.13 | 520.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 520.30 | 518.13 | 520.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 518.80 | 518.26 | 520.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 516.95 | 520.99 | 521.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 491.10 | 498.29 | 503.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 485.50 | 480.63 | 488.68 | SL hit (close>ema200) qty=0.50 sl=480.63 alert=retest2 |

### Cycle 51 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 491.75 | 488.59 | 488.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 498.15 | 490.50 | 489.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 492.30 | 496.69 | 493.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 492.30 | 496.69 | 493.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 492.30 | 496.69 | 493.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 492.30 | 496.69 | 493.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 488.75 | 495.10 | 493.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 488.75 | 495.10 | 493.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 489.70 | 494.02 | 493.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 489.70 | 494.02 | 493.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 492.80 | 492.64 | 492.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 491.05 | 492.64 | 492.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 486.15 | 491.34 | 492.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 483.30 | 489.74 | 491.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 481.55 | 478.32 | 480.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 481.55 | 478.32 | 480.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 481.55 | 478.32 | 480.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:45:00 | 482.10 | 478.32 | 480.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 482.30 | 479.11 | 480.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 482.45 | 479.11 | 480.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 486.00 | 481.38 | 481.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 486.00 | 481.38 | 481.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 478.00 | 475.59 | 478.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 477.05 | 475.59 | 478.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 473.70 | 475.22 | 477.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:45:00 | 470.70 | 475.00 | 477.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 482.10 | 476.68 | 477.92 | SL hit (close>static) qty=1.00 sl=478.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 491.40 | 480.47 | 479.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 507.85 | 494.06 | 489.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 506.50 | 507.08 | 499.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:30:00 | 505.20 | 507.08 | 499.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 505.50 | 509.73 | 507.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 505.75 | 509.73 | 507.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 502.80 | 508.35 | 506.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 502.80 | 508.35 | 506.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 499.70 | 506.62 | 506.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 499.70 | 506.62 | 506.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 503.55 | 505.54 | 505.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 499.45 | 504.24 | 505.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 508.00 | 504.53 | 504.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 508.00 | 504.53 | 504.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 508.00 | 504.53 | 504.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 508.00 | 504.53 | 504.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 510.00 | 505.63 | 505.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 517.80 | 510.11 | 507.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 12:15:00 | 504.10 | 509.57 | 508.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 12:15:00 | 504.10 | 509.57 | 508.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 504.10 | 509.57 | 508.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 504.10 | 509.57 | 508.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 505.50 | 508.76 | 507.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 508.75 | 509.17 | 508.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 515.10 | 524.94 | 525.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 515.10 | 524.94 | 525.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 511.70 | 516.51 | 519.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 516.55 | 514.04 | 516.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 516.55 | 514.04 | 516.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 516.55 | 514.04 | 516.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 516.60 | 514.04 | 516.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 521.75 | 515.58 | 517.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 524.10 | 515.58 | 517.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 520.80 | 516.63 | 517.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 520.80 | 516.63 | 517.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 522.35 | 518.98 | 518.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 524.95 | 520.18 | 519.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 515.80 | 519.30 | 518.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 515.80 | 519.30 | 518.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 515.80 | 519.30 | 518.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 515.70 | 519.30 | 518.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 514.55 | 518.35 | 518.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 11:15:00 | 511.20 | 516.92 | 517.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 517.70 | 515.02 | 516.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 517.70 | 515.02 | 516.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 517.70 | 515.02 | 516.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 517.70 | 515.02 | 516.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 515.25 | 515.07 | 516.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 516.75 | 515.07 | 516.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 513.70 | 514.79 | 516.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 511.85 | 512.99 | 515.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 486.26 | 501.20 | 506.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 478.85 | 477.81 | 486.08 | SL hit (close>ema200) qty=0.50 sl=477.81 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 489.30 | 486.72 | 486.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 12:15:00 | 491.00 | 488.96 | 487.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 487.05 | 489.53 | 488.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 487.05 | 489.53 | 488.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 487.05 | 489.53 | 488.67 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 486.55 | 487.99 | 488.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 479.75 | 486.02 | 487.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 468.15 | 465.54 | 470.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 468.15 | 465.54 | 470.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 468.15 | 465.54 | 470.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 469.05 | 465.54 | 470.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 467.80 | 465.83 | 468.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 468.65 | 465.83 | 468.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 468.00 | 466.26 | 468.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 470.90 | 466.26 | 468.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 470.95 | 467.20 | 469.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 468.30 | 467.20 | 469.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 470.90 | 467.94 | 469.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 470.90 | 467.94 | 469.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 472.35 | 468.82 | 469.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 472.35 | 468.82 | 469.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 13:15:00 | 473.70 | 470.55 | 470.21 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 464.05 | 469.25 | 469.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 463.00 | 468.00 | 469.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 468.65 | 468.13 | 469.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 468.65 | 468.13 | 469.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 468.65 | 468.13 | 469.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 470.30 | 468.13 | 469.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 462.50 | 467.01 | 468.42 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 475.05 | 470.14 | 469.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 476.20 | 471.35 | 470.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 483.90 | 484.23 | 481.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 11:15:00 | 484.00 | 484.23 | 481.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 481.05 | 486.06 | 484.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 481.05 | 486.06 | 484.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 477.50 | 484.35 | 483.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 481.50 | 484.35 | 483.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 471.10 | 480.85 | 482.02 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 499.25 | 484.04 | 482.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 502.05 | 494.08 | 488.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 13:15:00 | 517.55 | 518.56 | 510.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 14:00:00 | 517.55 | 518.56 | 510.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 508.75 | 523.21 | 519.30 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 14:15:00 | 514.35 | 516.91 | 517.20 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 520.80 | 517.80 | 517.52 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 512.05 | 516.48 | 516.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 510.15 | 514.92 | 516.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 09:15:00 | 505.35 | 504.49 | 508.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 505.35 | 504.49 | 508.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 505.35 | 504.49 | 508.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 500.40 | 503.83 | 506.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 12:45:00 | 500.35 | 500.17 | 503.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 494.40 | 501.15 | 503.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 507.30 | 503.47 | 503.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 507.30 | 503.47 | 503.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 510.45 | 504.85 | 503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 13:15:00 | 512.50 | 512.66 | 509.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:45:00 | 511.45 | 512.66 | 509.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 510.40 | 512.21 | 509.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 511.10 | 512.21 | 509.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 511.35 | 512.03 | 510.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 511.35 | 512.03 | 510.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 511.35 | 511.89 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 511.35 | 511.89 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 512.00 | 511.92 | 510.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 512.00 | 511.92 | 510.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 506.00 | 510.73 | 510.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 506.95 | 510.73 | 510.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 511.85 | 510.96 | 510.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 514.25 | 511.61 | 510.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 513.60 | 511.97 | 511.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:00:00 | 519.40 | 513.53 | 512.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 514.80 | 512.89 | 512.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 514.50 | 513.21 | 512.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 517.85 | 514.14 | 513.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:00:00 | 519.50 | 515.97 | 514.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 522.70 | 516.70 | 514.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 13:45:00 | 519.70 | 519.89 | 517.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 521.55 | 524.05 | 521.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 521.60 | 524.05 | 521.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 520.80 | 523.40 | 521.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:30:00 | 521.15 | 523.40 | 521.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 519.75 | 522.67 | 521.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:30:00 | 520.80 | 522.67 | 521.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 518.90 | 521.92 | 521.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 523.70 | 521.92 | 521.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 553.40 | 545.56 | 536.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 525.50 | 536.48 | 536.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 525.50 | 536.48 | 536.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 12:15:00 | 523.65 | 533.91 | 535.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 527.45 | 527.25 | 531.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 527.45 | 527.25 | 531.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 536.40 | 529.08 | 531.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 538.35 | 529.08 | 531.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 533.25 | 529.91 | 532.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:15:00 | 538.00 | 529.91 | 532.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 536.85 | 531.30 | 532.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 538.20 | 531.30 | 532.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 541.80 | 534.31 | 533.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 556.20 | 540.12 | 536.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 533.50 | 547.21 | 543.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 533.50 | 547.21 | 543.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 533.50 | 547.21 | 543.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 533.50 | 547.21 | 543.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 525.00 | 542.77 | 541.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 530.00 | 542.77 | 541.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 524.85 | 539.18 | 540.15 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 549.15 | 540.38 | 540.03 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 535.65 | 544.06 | 544.42 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 550.25 | 545.30 | 544.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 553.95 | 547.03 | 545.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 536.55 | 546.55 | 546.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 536.55 | 546.55 | 546.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 536.55 | 546.55 | 546.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 536.55 | 546.55 | 546.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 533.00 | 543.84 | 545.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 519.15 | 533.47 | 538.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 516.35 | 513.57 | 521.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 516.35 | 513.57 | 521.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 522.50 | 516.27 | 521.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 522.00 | 516.27 | 521.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 524.00 | 517.82 | 521.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 529.75 | 517.82 | 521.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 526.15 | 519.48 | 521.89 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 529.75 | 523.79 | 523.57 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 516.35 | 522.97 | 523.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 516.15 | 521.60 | 523.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 519.55 | 515.87 | 519.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 519.55 | 515.87 | 519.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 519.55 | 515.87 | 519.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 522.80 | 515.87 | 519.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 521.75 | 517.05 | 519.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 521.75 | 517.05 | 519.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 520.05 | 517.65 | 519.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 515.30 | 517.65 | 519.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 489.53 | 507.96 | 514.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 496.65 | 491.49 | 498.85 | SL hit (close>ema200) qty=0.50 sl=491.49 alert=retest2 |

### Cycle 79 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 514.00 | 503.06 | 502.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 522.20 | 506.89 | 504.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 510.45 | 515.17 | 511.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 510.45 | 515.17 | 511.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 510.45 | 515.17 | 511.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 509.90 | 515.17 | 511.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 506.90 | 513.52 | 510.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 508.45 | 513.52 | 510.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 512.10 | 513.23 | 510.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 514.65 | 513.17 | 511.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 505.10 | 511.16 | 510.65 | SL hit (close<static) qty=1.00 sl=506.05 alert=retest2 |

### Cycle 80 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 500.85 | 509.10 | 509.76 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 15:15:00 | 510.00 | 509.77 | 509.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 531.35 | 514.09 | 511.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 506.15 | 519.40 | 516.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 506.15 | 519.40 | 516.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 506.15 | 519.40 | 516.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 506.15 | 519.40 | 516.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 508.25 | 517.17 | 516.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 510.00 | 517.17 | 516.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-13 09:15:00 | 561.00 | 556.13 | 549.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 582.50 | 587.82 | 588.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 578.70 | 584.49 | 586.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 580.80 | 579.81 | 582.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 580.80 | 579.81 | 582.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 580.80 | 579.81 | 582.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:30:00 | 583.30 | 579.81 | 582.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 580.70 | 579.99 | 582.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 581.45 | 579.99 | 582.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 585.75 | 581.14 | 583.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 585.75 | 581.14 | 583.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 587.20 | 582.35 | 583.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 587.20 | 582.35 | 583.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 589.70 | 583.82 | 583.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 589.70 | 583.82 | 583.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 591.90 | 585.44 | 584.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 594.25 | 588.34 | 586.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 589.75 | 590.36 | 588.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 14:00:00 | 589.75 | 590.36 | 588.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 586.85 | 589.66 | 587.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 585.70 | 589.66 | 587.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 585.10 | 588.75 | 587.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 582.85 | 588.75 | 587.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 584.90 | 587.98 | 587.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 583.05 | 587.98 | 587.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 582.65 | 586.91 | 586.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 580.50 | 584.27 | 585.61 | Break + close below crossover candle low |

### Cycle 85 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 609.50 | 588.80 | 587.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 614.30 | 605.42 | 597.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 602.60 | 609.12 | 603.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 602.60 | 609.12 | 603.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 602.60 | 609.12 | 603.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 605.30 | 609.12 | 603.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 603.85 | 608.06 | 603.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 602.80 | 608.06 | 603.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 602.80 | 607.01 | 603.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 598.75 | 605.33 | 602.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 601.60 | 604.58 | 602.58 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 594.30 | 600.47 | 601.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 585.95 | 597.17 | 599.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 593.75 | 593.65 | 597.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:00:00 | 593.75 | 593.65 | 597.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 592.95 | 592.49 | 595.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 591.50 | 592.56 | 594.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 590.95 | 591.79 | 594.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 590.00 | 591.64 | 593.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 590.30 | 586.65 | 587.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 596.65 | 589.51 | 589.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 596.65 | 589.51 | 589.01 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 452.20 | 2025-05-26 09:15:00 | 439.45 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-05-14 11:00:00 | 451.70 | 2025-05-26 09:15:00 | 439.45 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-05-14 12:00:00 | 450.45 | 2025-05-26 09:15:00 | 439.45 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-05-14 13:00:00 | 450.45 | 2025-05-26 09:15:00 | 439.45 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-05-23 12:00:00 | 474.60 | 2025-05-26 09:15:00 | 439.45 | STOP_HIT | 1.00 | -7.41% |
| SELL | retest2 | 2025-06-02 09:15:00 | 433.80 | 2025-06-03 11:15:00 | 439.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-06-02 14:30:00 | 434.00 | 2025-06-03 11:15:00 | 439.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-06 10:15:00 | 443.50 | 2025-06-06 11:15:00 | 438.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-26 09:15:00 | 447.00 | 2025-07-01 09:15:00 | 437.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-06-26 13:45:00 | 441.45 | 2025-07-01 09:15:00 | 437.80 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-04 11:45:00 | 430.55 | 2025-07-09 13:15:00 | 434.85 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-04 14:00:00 | 430.50 | 2025-07-09 13:15:00 | 434.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-07 09:30:00 | 430.80 | 2025-07-09 13:15:00 | 434.85 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-09 12:45:00 | 430.80 | 2025-07-09 13:15:00 | 434.85 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-07-15 14:15:00 | 445.10 | 2025-07-17 12:15:00 | 443.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-16 09:15:00 | 445.50 | 2025-07-17 14:15:00 | 442.85 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-16 09:45:00 | 447.20 | 2025-07-17 14:15:00 | 442.85 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-16 14:45:00 | 444.85 | 2025-07-17 14:15:00 | 442.85 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-17 12:15:00 | 449.40 | 2025-07-17 14:15:00 | 442.85 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-07-18 15:15:00 | 441.00 | 2025-07-21 09:15:00 | 447.35 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-31 11:30:00 | 451.35 | 2025-08-01 14:15:00 | 439.30 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-07-31 12:15:00 | 449.65 | 2025-08-01 14:15:00 | 439.30 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-08-20 11:15:00 | 591.70 | 2025-08-26 09:15:00 | 591.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-08-28 14:30:00 | 586.90 | 2025-08-29 09:15:00 | 595.20 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-03 13:45:00 | 585.30 | 2025-09-08 15:15:00 | 578.95 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-09-04 09:15:00 | 579.75 | 2025-09-08 15:15:00 | 578.95 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-22 14:30:00 | 583.40 | 2025-09-23 13:15:00 | 587.65 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-23 09:30:00 | 586.05 | 2025-09-23 13:15:00 | 587.65 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-23 11:15:00 | 587.70 | 2025-09-23 13:15:00 | 587.65 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-09-25 11:30:00 | 585.75 | 2025-09-29 14:15:00 | 556.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:45:00 | 584.65 | 2025-09-29 14:15:00 | 555.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:30:00 | 585.75 | 2025-10-01 13:15:00 | 547.45 | STOP_HIT | 0.50 | 6.54% |
| SELL | retest2 | 2025-09-25 14:45:00 | 584.65 | 2025-10-01 13:15:00 | 547.45 | STOP_HIT | 0.50 | 6.36% |
| SELL | retest2 | 2025-10-16 14:00:00 | 544.60 | 2025-10-20 09:15:00 | 517.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 09:45:00 | 544.40 | 2025-10-20 09:15:00 | 517.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 14:00:00 | 544.60 | 2025-10-20 15:15:00 | 527.00 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-10-17 09:45:00 | 544.40 | 2025-10-20 15:15:00 | 527.00 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-10-23 10:00:00 | 543.05 | 2025-10-23 10:15:00 | 541.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-11-10 09:15:00 | 529.85 | 2025-11-10 11:15:00 | 537.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-11-10 09:45:00 | 530.65 | 2025-11-10 11:15:00 | 537.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-11-11 12:45:00 | 533.20 | 2025-11-11 13:15:00 | 531.25 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-11-18 09:15:00 | 516.95 | 2025-11-21 14:15:00 | 491.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 516.95 | 2025-11-25 09:15:00 | 485.50 | STOP_HIT | 0.50 | 6.08% |
| SELL | retest2 | 2025-12-09 12:45:00 | 470.70 | 2025-12-09 14:15:00 | 482.10 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-12-19 14:30:00 | 508.75 | 2025-12-29 12:15:00 | 515.10 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2026-01-06 10:30:00 | 511.85 | 2026-01-08 11:15:00 | 486.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:30:00 | 511.85 | 2026-01-12 12:15:00 | 478.85 | STOP_HIT | 0.50 | 6.45% |
| SELL | retest2 | 2026-02-12 14:45:00 | 500.40 | 2026-02-17 09:15:00 | 507.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-13 12:45:00 | 500.35 | 2026-02-17 09:15:00 | 507.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-16 09:15:00 | 494.40 | 2026-02-17 09:15:00 | 507.30 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-02-20 10:30:00 | 514.25 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2026-02-23 09:15:00 | 513.60 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2026-02-23 12:00:00 | 519.40 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2026-02-24 10:15:00 | 514.80 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2026-02-24 12:00:00 | 517.85 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2026-02-24 14:00:00 | 519.50 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-02-25 09:15:00 | 522.70 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2026-02-25 13:45:00 | 519.70 | 2026-03-04 11:15:00 | 525.50 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2026-03-20 12:15:00 | 515.30 | 2026-03-23 09:15:00 | 489.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 515.30 | 2026-03-24 11:15:00 | 496.65 | STOP_HIT | 0.50 | 3.62% |
| BUY | retest2 | 2026-03-27 13:15:00 | 514.65 | 2026-03-27 15:15:00 | 505.10 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-04-02 11:15:00 | 510.00 | 2026-04-13 09:15:00 | 561.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 12:00:00 | 591.50 | 2026-05-07 11:15:00 | 596.65 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-05-04 12:30:00 | 590.95 | 2026-05-07 11:15:00 | 596.65 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-05-05 10:00:00 | 590.00 | 2026-05-07 11:15:00 | 596.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-05-07 09:45:00 | 590.30 | 2026-05-07 11:15:00 | 596.65 | STOP_HIT | 1.00 | -1.08% |
