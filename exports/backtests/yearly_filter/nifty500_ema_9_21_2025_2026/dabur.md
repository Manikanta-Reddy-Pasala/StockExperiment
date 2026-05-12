# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 487.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 44 |
| ALERT2 | 45 |
| ALERT2_SKIP | 22 |
| ALERT3 | 113 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 65 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 51
- **Target hits / Stop hits / Partials:** 3 / 65 / 7
- **Avg / median % per leg:** 0.55% / -0.65%
- **Sum % (uncompounded):** 41.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 5 | 22.7% | 2 | 20 | 0 | 0.20% | 4.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 5 | 22.7% | 2 | 20 | 0 | 0.20% | 4.4% |
| SELL (all) | 53 | 19 | 35.8% | 1 | 45 | 7 | 0.70% | 37.0% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 18 | 36.0% | 1 | 42 | 7 | 0.74% | 37.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 23 | 31.9% | 3 | 62 | 7 | 0.57% | 41.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 475.90 | 471.55 | 471.15 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 469.70 | 471.63 | 471.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 467.90 | 470.88 | 471.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 470.35 | 470.00 | 470.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 470.35 | 470.00 | 470.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 470.35 | 470.00 | 470.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 12:15:00 | 468.15 | 470.20 | 470.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 468.55 | 469.53 | 470.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:30:00 | 468.05 | 469.13 | 469.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 10:15:00 | 473.05 | 469.92 | 470.22 | SL hit (close>static) qty=1.00 sl=472.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 473.20 | 470.57 | 470.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 474.30 | 471.32 | 470.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 480.65 | 481.84 | 478.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 12:45:00 | 480.40 | 481.84 | 478.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 479.35 | 481.34 | 479.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 479.35 | 481.34 | 479.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 475.75 | 480.22 | 478.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 475.75 | 480.22 | 478.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 476.10 | 479.40 | 478.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 480.10 | 479.40 | 478.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 476.30 | 479.38 | 479.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 476.30 | 479.38 | 479.60 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 483.00 | 480.16 | 479.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 483.90 | 482.80 | 482.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 483.20 | 483.88 | 483.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 15:15:00 | 483.20 | 483.88 | 483.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 483.20 | 483.88 | 483.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 482.30 | 483.88 | 483.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 483.35 | 483.77 | 483.17 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 481.10 | 482.82 | 482.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 479.95 | 481.83 | 482.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 13:15:00 | 481.40 | 480.95 | 481.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 13:15:00 | 481.40 | 480.95 | 481.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 481.40 | 480.95 | 481.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 481.40 | 480.95 | 481.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 481.75 | 481.11 | 481.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 481.75 | 481.11 | 481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 483.15 | 481.52 | 481.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 487.30 | 481.52 | 481.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 486.45 | 482.50 | 482.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 489.50 | 483.90 | 482.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 13:15:00 | 482.25 | 483.85 | 483.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 482.25 | 483.85 | 483.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 482.25 | 483.85 | 483.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 482.25 | 483.85 | 483.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 484.25 | 483.93 | 483.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 485.10 | 484.05 | 483.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 13:00:00 | 485.50 | 484.46 | 483.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 490.30 | 484.90 | 484.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 483.25 | 486.66 | 487.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 483.25 | 486.66 | 487.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 482.45 | 485.82 | 486.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 484.00 | 483.55 | 485.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:45:00 | 484.15 | 483.55 | 485.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 485.10 | 484.04 | 484.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 485.10 | 484.04 | 484.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 485.70 | 484.37 | 484.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 485.70 | 484.37 | 484.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 486.50 | 485.31 | 485.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 488.00 | 486.13 | 485.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 488.85 | 490.01 | 488.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 488.85 | 490.01 | 488.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 488.85 | 490.01 | 488.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 489.10 | 490.01 | 488.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 488.15 | 489.64 | 488.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 488.15 | 489.64 | 488.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 486.05 | 488.92 | 488.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 484.95 | 488.92 | 488.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 481.15 | 487.37 | 487.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 479.15 | 485.72 | 486.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 469.50 | 469.10 | 473.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 469.50 | 469.10 | 473.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 473.20 | 470.50 | 473.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 473.20 | 470.50 | 473.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 471.95 | 470.79 | 473.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 470.65 | 471.26 | 472.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:30:00 | 470.55 | 471.43 | 472.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 469.75 | 471.23 | 472.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:30:00 | 470.25 | 470.63 | 471.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 469.90 | 470.49 | 471.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 469.90 | 470.49 | 471.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 467.35 | 466.85 | 468.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 467.35 | 466.85 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 466.90 | 466.35 | 467.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 466.90 | 466.35 | 467.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 467.70 | 466.62 | 467.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 467.70 | 466.62 | 467.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 467.95 | 466.88 | 467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 467.95 | 466.88 | 467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 467.50 | 467.01 | 467.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 471.95 | 467.01 | 467.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 470.65 | 467.74 | 467.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 470.65 | 467.74 | 467.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 473.25 | 469.44 | 468.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 480.70 | 483.98 | 481.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 480.70 | 483.98 | 481.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 480.70 | 483.98 | 481.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 480.70 | 483.98 | 481.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 481.10 | 483.40 | 481.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 481.00 | 483.40 | 481.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 483.45 | 484.69 | 483.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 483.45 | 484.69 | 483.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 485.10 | 484.77 | 483.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 486.50 | 484.18 | 483.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:45:00 | 486.00 | 485.15 | 484.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-11 15:15:00 | 535.15 | 527.26 | 522.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 520.55 | 527.00 | 527.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 518.50 | 521.30 | 523.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 14:15:00 | 514.50 | 514.40 | 517.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 514.50 | 514.40 | 517.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 512.25 | 514.22 | 516.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:30:00 | 510.80 | 511.88 | 513.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 520.80 | 514.19 | 513.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 520.80 | 514.19 | 513.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 11:15:00 | 521.35 | 515.62 | 514.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 518.60 | 518.86 | 516.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 518.55 | 518.86 | 516.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 515.20 | 518.12 | 516.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:45:00 | 515.65 | 518.12 | 516.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 516.70 | 517.84 | 516.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:00:00 | 517.70 | 517.70 | 516.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 517.70 | 517.70 | 516.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 521.80 | 529.82 | 530.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 521.80 | 529.82 | 530.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 520.30 | 524.55 | 526.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 517.25 | 515.87 | 518.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 517.25 | 515.87 | 518.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 517.60 | 516.22 | 518.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 516.65 | 516.22 | 518.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 511.45 | 515.26 | 518.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 510.35 | 513.75 | 516.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:30:00 | 510.50 | 512.36 | 515.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 525.50 | 506.32 | 505.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 525.50 | 506.32 | 505.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 532.00 | 525.11 | 520.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 527.90 | 529.39 | 524.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 525.85 | 529.39 | 524.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 524.00 | 528.31 | 524.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 524.85 | 528.31 | 524.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 524.30 | 527.51 | 524.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:15:00 | 524.05 | 527.51 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 524.10 | 526.83 | 524.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 523.60 | 526.83 | 524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 521.95 | 525.85 | 523.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 521.95 | 525.85 | 523.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 518.60 | 524.40 | 523.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 518.60 | 524.40 | 523.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 518.70 | 522.07 | 522.44 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 524.05 | 520.23 | 520.03 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 517.40 | 520.49 | 520.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 515.40 | 519.48 | 520.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 518.20 | 516.64 | 518.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 518.20 | 516.64 | 518.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 520.55 | 517.42 | 518.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 520.55 | 517.42 | 518.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 521.00 | 518.14 | 518.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 521.65 | 518.14 | 518.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 520.65 | 519.17 | 519.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 524.15 | 521.96 | 520.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 546.20 | 550.43 | 546.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 546.20 | 550.43 | 546.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 546.20 | 550.43 | 546.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 546.20 | 550.43 | 546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 543.50 | 549.04 | 545.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 543.50 | 549.04 | 545.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 545.75 | 548.39 | 545.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 546.50 | 548.39 | 545.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 547.75 | 548.31 | 546.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 541.25 | 545.89 | 546.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 541.25 | 545.89 | 546.14 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 546.60 | 545.57 | 545.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 547.65 | 545.98 | 545.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 544.20 | 545.85 | 545.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 544.20 | 545.85 | 545.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 544.20 | 545.85 | 545.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 545.35 | 545.85 | 545.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 543.50 | 545.38 | 545.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 537.35 | 543.40 | 544.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 541.00 | 539.43 | 541.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 541.00 | 539.43 | 541.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 541.00 | 539.43 | 541.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 540.80 | 539.43 | 541.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 540.40 | 539.63 | 541.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 541.00 | 539.63 | 541.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 542.15 | 540.11 | 540.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 538.55 | 540.19 | 540.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 539.50 | 540.10 | 540.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 539.05 | 539.85 | 540.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 537.65 | 535.87 | 536.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 535.50 | 535.80 | 536.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:00:00 | 533.90 | 535.42 | 536.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 536.75 | 535.81 | 536.30 | SL hit (close>static) qty=1.00 sl=536.60 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 489.10 | 487.54 | 487.41 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 484.30 | 487.06 | 487.30 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 13:15:00 | 487.75 | 487.04 | 487.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 491.10 | 487.85 | 487.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 506.40 | 506.60 | 502.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:45:00 | 506.55 | 506.60 | 502.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 505.50 | 505.72 | 503.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 505.50 | 505.72 | 503.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 501.30 | 508.35 | 506.81 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 503.95 | 506.38 | 506.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 502.30 | 504.81 | 505.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 504.00 | 503.75 | 504.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 504.00 | 503.75 | 504.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 504.00 | 503.75 | 504.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 507.25 | 503.75 | 504.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 504.85 | 503.97 | 504.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 506.95 | 503.97 | 504.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 507.25 | 504.63 | 505.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 507.25 | 504.63 | 505.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 507.20 | 505.14 | 505.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 507.20 | 505.14 | 505.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 507.30 | 505.57 | 505.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 508.50 | 506.56 | 505.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 503.50 | 505.95 | 505.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 503.50 | 505.95 | 505.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 503.50 | 505.95 | 505.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 503.50 | 505.95 | 505.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 504.00 | 505.56 | 505.55 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 501.00 | 504.65 | 505.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 492.15 | 500.76 | 503.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 492.25 | 491.71 | 495.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:00:00 | 492.25 | 491.71 | 495.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 500.40 | 493.15 | 495.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 500.40 | 493.15 | 495.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 503.70 | 495.26 | 496.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 503.70 | 495.26 | 496.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 503.85 | 498.10 | 497.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 511.55 | 502.46 | 499.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 516.10 | 519.11 | 513.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 516.10 | 519.11 | 513.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 516.10 | 519.11 | 513.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 523.55 | 518.81 | 517.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 522.25 | 519.50 | 517.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:30:00 | 522.25 | 520.25 | 518.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:45:00 | 522.25 | 520.75 | 518.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 518.25 | 520.41 | 519.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 518.25 | 520.41 | 519.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 520.95 | 520.52 | 519.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 521.85 | 521.00 | 519.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 522.05 | 522.29 | 520.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 514.15 | 522.50 | 522.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 514.15 | 522.50 | 522.68 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 525.30 | 522.09 | 521.69 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 518.75 | 521.81 | 521.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 515.85 | 519.37 | 520.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 515.15 | 513.66 | 515.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 515.15 | 513.66 | 515.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 515.00 | 513.93 | 515.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 516.50 | 513.93 | 515.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 517.35 | 514.61 | 515.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 517.35 | 514.61 | 515.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 517.55 | 515.20 | 515.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 517.55 | 515.20 | 515.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 518.15 | 516.28 | 516.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 519.25 | 517.38 | 516.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 517.30 | 517.56 | 517.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 517.30 | 517.56 | 517.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 517.30 | 517.56 | 517.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 517.30 | 517.56 | 517.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 518.50 | 517.74 | 517.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 517.85 | 517.74 | 517.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 515.75 | 517.89 | 517.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 517.55 | 517.89 | 517.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 514.00 | 517.12 | 517.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 514.00 | 517.12 | 517.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 515.65 | 516.82 | 516.98 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 518.60 | 516.97 | 516.95 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 516.30 | 516.84 | 516.89 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 518.60 | 516.89 | 516.89 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 516.10 | 516.73 | 516.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 514.25 | 515.67 | 516.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 508.85 | 507.69 | 510.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 508.85 | 507.69 | 510.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 508.80 | 506.79 | 508.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 508.80 | 506.79 | 508.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 508.50 | 507.13 | 508.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 508.60 | 507.13 | 508.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 508.35 | 507.37 | 508.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 508.80 | 507.37 | 508.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 507.65 | 507.43 | 508.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 505.70 | 506.76 | 508.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 507.20 | 507.05 | 508.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 509.50 | 507.20 | 507.87 | SL hit (close>static) qty=1.00 sl=509.10 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 510.50 | 508.35 | 508.31 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 506.85 | 508.05 | 508.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 505.15 | 507.47 | 507.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 505.00 | 504.82 | 506.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:15:00 | 499.50 | 504.82 | 506.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 11:30:00 | 503.50 | 504.49 | 505.65 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 504.15 | 504.42 | 505.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 505.50 | 504.42 | 505.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 503.90 | 504.32 | 505.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 504.40 | 504.32 | 505.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 504.95 | 504.23 | 505.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 504.50 | 504.23 | 505.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 505.50 | 504.49 | 505.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 505.50 | 504.49 | 505.17 | SL hit (close>ema400) qty=1.00 sl=505.17 alert=retest1 |

### Cycle 41 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 503.25 | 499.63 | 499.19 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 496.20 | 498.75 | 498.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 494.65 | 497.34 | 498.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 493.50 | 492.95 | 494.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 493.50 | 492.95 | 494.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 493.80 | 493.06 | 493.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 493.80 | 493.06 | 493.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 494.25 | 493.30 | 493.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 495.30 | 493.30 | 493.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 494.70 | 493.58 | 494.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 494.35 | 493.58 | 494.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 493.75 | 493.61 | 493.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 493.65 | 493.61 | 493.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 493.20 | 493.53 | 493.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 496.00 | 494.02 | 494.10 | SL hit (close>static) qty=1.00 sl=495.25 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 495.45 | 494.31 | 494.22 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 493.85 | 494.13 | 494.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 493.15 | 493.86 | 494.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 494.00 | 493.89 | 494.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 494.00 | 493.89 | 494.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 494.00 | 493.89 | 494.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 494.05 | 493.89 | 494.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 492.80 | 493.67 | 493.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 13:15:00 | 492.70 | 493.67 | 493.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:45:00 | 492.55 | 493.10 | 493.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 492.45 | 489.74 | 489.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 494.55 | 490.70 | 490.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 494.55 | 490.70 | 490.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 505.35 | 494.87 | 492.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 501.50 | 501.81 | 498.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 501.50 | 501.81 | 498.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 498.90 | 500.77 | 498.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 498.60 | 500.77 | 498.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 499.40 | 500.50 | 498.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 501.75 | 500.12 | 498.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 12:15:00 | 518.65 | 521.41 | 521.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 518.65 | 521.41 | 521.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 10:15:00 | 515.70 | 519.50 | 520.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 517.55 | 516.29 | 518.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 517.55 | 516.29 | 518.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 517.55 | 516.29 | 518.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 510.35 | 515.42 | 516.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:45:00 | 511.00 | 513.02 | 514.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 511.40 | 512.66 | 514.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 517.90 | 512.77 | 512.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 517.90 | 512.77 | 512.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 525.25 | 515.27 | 513.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 524.10 | 525.75 | 521.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 524.10 | 525.75 | 521.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 520.15 | 524.63 | 521.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 520.15 | 524.63 | 521.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 518.60 | 523.42 | 521.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 520.70 | 523.42 | 521.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 515.80 | 520.48 | 520.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 515.80 | 520.48 | 520.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 515.45 | 519.48 | 519.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 513.40 | 518.26 | 519.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 12:15:00 | 513.55 | 512.23 | 515.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:00:00 | 513.55 | 512.23 | 515.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 515.05 | 512.46 | 514.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 515.05 | 512.46 | 514.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 517.20 | 513.41 | 514.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 509.50 | 511.81 | 514.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 514.70 | 508.99 | 510.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 514.25 | 510.41 | 511.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 508.25 | 505.02 | 503.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 517.25 | 519.39 | 516.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 517.25 | 519.39 | 516.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 518.40 | 519.19 | 516.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 520.10 | 519.27 | 516.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 520.20 | 519.60 | 517.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 512.90 | 518.25 | 517.28 | SL hit (close<static) qty=1.00 sl=516.10 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 512.50 | 516.01 | 516.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 511.30 | 515.07 | 516.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 515.00 | 514.20 | 515.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:00:00 | 515.00 | 514.20 | 515.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 514.45 | 514.25 | 515.32 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 517.05 | 515.64 | 515.56 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 506.45 | 514.12 | 514.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 505.00 | 511.46 | 513.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 510.75 | 510.69 | 512.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 510.75 | 510.69 | 512.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 512.45 | 510.98 | 512.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 512.45 | 510.98 | 512.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 508.70 | 510.53 | 512.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 507.70 | 509.60 | 511.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 507.25 | 505.89 | 508.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 515.00 | 511.60 | 510.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 512.60 | 519.86 | 518.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 516.35 | 519.16 | 518.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 517.25 | 518.93 | 518.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 510.00 | 517.28 | 517.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 510.00 | 517.28 | 517.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 496.45 | 507.32 | 511.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 488.15 | 488.01 | 494.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 485.95 | 488.01 | 494.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 478.15 | 471.88 | 476.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 478.15 | 471.88 | 476.97 | SL hit (close>ema400) qty=1.00 sl=476.97 alert=retest1 |

### Cycle 55 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 437.70 | 427.71 | 427.55 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 422.15 | 427.73 | 428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 418.85 | 424.01 | 426.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 418.75 | 414.63 | 418.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 415.25 | 414.75 | 418.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 414.60 | 415.48 | 417.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 413.20 | 415.31 | 417.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 414.60 | 415.31 | 417.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 419.60 | 413.82 | 414.76 | SL hit (close>static) qty=1.00 sl=419.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 424.60 | 415.89 | 414.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 428.80 | 421.66 | 418.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 424.40 | 427.97 | 428.17 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 430.00 | 428.37 | 428.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 435.70 | 430.12 | 429.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 431.90 | 432.27 | 430.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 428.95 | 431.61 | 430.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 428.95 | 431.61 | 430.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 428.60 | 431.01 | 430.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 428.50 | 431.01 | 430.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 15:15:00 | 427.80 | 429.91 | 430.14 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 436.50 | 431.23 | 430.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 440.90 | 433.16 | 431.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 440.00 | 441.87 | 439.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 441.75 | 441.87 | 439.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 450.25 | 443.55 | 440.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 453.20 | 443.55 | 440.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 452.55 | 448.02 | 444.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 455.90 | 454.10 | 453.89 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 452.00 | 453.75 | 453.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 450.10 | 452.80 | 453.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 455.80 | 452.49 | 452.49 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 442.15 | 450.84 | 451.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 440.95 | 445.60 | 448.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 449.30 | 445.76 | 448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 449.35 | 446.48 | 448.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 449.75 | 446.48 | 448.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 446.55 | 446.80 | 448.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 444.60 | 446.80 | 448.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:45:00 | 445.00 | 446.57 | 448.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 453.45 | 447.56 | 448.06 | SL hit (close>static) qty=1.00 sl=448.90 alert=retest2 |

### Cycle 67 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 454.50 | 448.94 | 448.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 455.55 | 450.27 | 449.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 463.35 | 463.59 | 459.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 463.35 | 463.59 | 459.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 12:15:00 | 468.15 | 2025-05-16 10:15:00 | 473.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-15 14:00:00 | 468.55 | 2025-05-16 10:15:00 | 473.05 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-05-16 09:30:00 | 468.05 | 2025-05-16 10:15:00 | 473.05 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-21 09:15:00 | 480.10 | 2025-05-22 13:15:00 | 476.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-03 09:15:00 | 485.10 | 2025-06-06 11:15:00 | 483.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-06-03 13:00:00 | 485.50 | 2025-06-06 11:15:00 | 483.25 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-04 09:15:00 | 490.30 | 2025-06-06 11:15:00 | 483.25 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-17 11:45:00 | 470.65 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-06-17 13:30:00 | 470.55 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-06-18 09:45:00 | 469.75 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-06-18 13:30:00 | 470.25 | 2025-06-24 09:15:00 | 470.65 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-07-02 12:15:00 | 486.50 | 2025-07-11 15:15:00 | 535.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 13:45:00 | 486.00 | 2025-07-11 15:15:00 | 534.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-25 11:30:00 | 510.80 | 2025-07-28 10:15:00 | 520.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-07-29 14:00:00 | 517.70 | 2025-08-05 09:15:00 | 521.80 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-07-29 15:00:00 | 517.70 | 2025-08-05 09:15:00 | 521.80 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-08-08 12:00:00 | 510.35 | 2025-08-18 09:15:00 | 525.50 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-08-08 13:30:00 | 510.50 | 2025-08-18 09:15:00 | 525.50 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-09-05 13:15:00 | 546.50 | 2025-09-09 09:15:00 | 541.25 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-08 09:30:00 | 547.75 | 2025-09-09 09:15:00 | 541.25 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-16 11:15:00 | 538.55 | 2025-09-19 13:15:00 | 536.75 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-09-16 11:45:00 | 539.50 | 2025-09-24 10:15:00 | 511.62 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-09-16 12:45:00 | 539.05 | 2025-09-24 10:15:00 | 512.52 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-09-19 09:30:00 | 537.65 | 2025-09-24 10:15:00 | 512.10 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-09-16 11:45:00 | 539.50 | 2025-09-24 11:15:00 | 517.35 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-09-16 12:45:00 | 539.05 | 2025-09-24 11:15:00 | 517.35 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-09-19 09:30:00 | 537.65 | 2025-09-24 11:15:00 | 517.35 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-09-19 12:00:00 | 533.90 | 2025-09-25 14:15:00 | 510.77 | PARTIAL | 0.50 | 4.33% |
| SELL | retest2 | 2025-09-22 09:15:00 | 532.50 | 2025-09-25 14:15:00 | 505.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 12:00:00 | 533.90 | 2025-10-01 10:15:00 | 483.88 | TARGET_HIT | 0.50 | 9.37% |
| SELL | retest2 | 2025-09-22 09:15:00 | 532.50 | 2025-10-01 12:15:00 | 490.15 | STOP_HIT | 0.50 | 7.95% |
| BUY | retest2 | 2025-11-12 11:15:00 | 523.55 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-11-12 12:00:00 | 522.25 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-12 13:30:00 | 522.25 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-12 14:45:00 | 522.25 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-13 11:45:00 | 521.85 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-14 10:30:00 | 522.05 | 2025-11-18 09:15:00 | 514.15 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-05 09:30:00 | 505.70 | 2025-12-05 13:15:00 | 509.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-05 11:15:00 | 507.20 | 2025-12-05 13:15:00 | 509.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-12-09 09:15:00 | 499.50 | 2025-12-10 09:15:00 | 505.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2025-12-09 11:30:00 | 503.50 | 2025-12-10 09:15:00 | 505.50 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-12-10 12:00:00 | 499.60 | 2025-12-16 11:15:00 | 503.25 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-10 15:15:00 | 500.00 | 2025-12-16 11:15:00 | 503.25 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-12-12 10:00:00 | 499.10 | 2025-12-16 11:15:00 | 503.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-22 11:15:00 | 493.65 | 2025-12-22 12:15:00 | 496.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-22 12:00:00 | 493.20 | 2025-12-22 12:15:00 | 496.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-23 13:15:00 | 492.70 | 2025-12-30 13:15:00 | 494.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-12-24 10:45:00 | 492.55 | 2025-12-30 13:15:00 | 494.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-12-30 13:00:00 | 492.45 | 2025-12-30 13:15:00 | 494.55 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-02 09:15:00 | 501.75 | 2026-01-13 12:15:00 | 518.65 | STOP_HIT | 1.00 | 3.37% |
| SELL | retest2 | 2026-01-19 09:15:00 | 510.35 | 2026-01-21 15:15:00 | 517.90 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-20 09:45:00 | 511.00 | 2026-01-21 15:15:00 | 517.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-20 10:45:00 | 511.40 | 2026-01-21 15:15:00 | 517.90 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-29 09:30:00 | 509.50 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-01-30 10:00:00 | 514.70 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2026-01-30 11:15:00 | 514.25 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2026-02-12 12:45:00 | 520.10 | 2026-02-13 09:15:00 | 512.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-12 13:30:00 | 520.20 | 2026-02-13 09:15:00 | 512.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-19 11:30:00 | 507.70 | 2026-02-23 11:15:00 | 512.60 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-20 11:00:00 | 507.25 | 2026-02-23 11:15:00 | 512.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-27 11:45:00 | 517.25 | 2026-03-02 09:15:00 | 510.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest1 | 2026-03-06 09:15:00 | 485.95 | 2026-03-10 10:15:00 | 478.15 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2026-03-11 09:30:00 | 476.90 | 2026-03-13 14:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 475.00 | 2026-03-16 09:15:00 | 451.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:30:00 | 476.90 | 2026-03-16 14:15:00 | 459.80 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-03-11 13:00:00 | 475.00 | 2026-03-16 14:15:00 | 459.80 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-04-01 13:30:00 | 414.60 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-01 14:30:00 | 413.20 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-01 15:15:00 | 414.60 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-06 11:15:00 | 413.50 | 2026-04-08 09:15:00 | 424.60 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-04-06 12:30:00 | 409.05 | 2026-04-08 09:15:00 | 424.60 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-04-21 10:15:00 | 453.20 | 2026-04-24 13:15:00 | 448.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-21 15:15:00 | 452.55 | 2026-04-24 13:15:00 | 448.95 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-05-04 13:15:00 | 444.60 | 2026-05-05 09:15:00 | 453.45 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-05-04 13:45:00 | 445.00 | 2026-05-05 09:15:00 | 453.45 | STOP_HIT | 1.00 | -1.90% |
