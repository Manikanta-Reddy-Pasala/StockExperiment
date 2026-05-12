# ITC (ITC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 307.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 162 |
| ALERT1 | 99 |
| ALERT2 | 97 |
| ALERT2_SKIP | 49 |
| ALERT3 | 261 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 139 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 134 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 97
- **Target hits / Stop hits / Partials:** 6 / 134 / 2
- **Avg / median % per leg:** 0.10% / -0.41%
- **Sum % (uncompounded):** 14.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 21 | 30.4% | 5 | 64 | 0 | 0.32% | 22.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 69 | 21 | 30.4% | 5 | 64 | 0 | 0.32% | 22.0% |
| SELL (all) | 73 | 24 | 32.9% | 1 | 70 | 2 | -0.10% | -7.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.18% | -1.2% |
| SELL @ 3rd Alert (retest2) | 72 | 24 | 33.3% | 1 | 69 | 2 | -0.08% | -6.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.18% | -1.2% |
| retest2 (combined) | 141 | 45 | 31.9% | 6 | 133 | 2 | 0.11% | 16.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 434.80 | 430.70 | 430.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 14:15:00 | 436.55 | 433.54 | 431.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 435.00 | 435.20 | 433.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 13:15:00 | 433.60 | 434.67 | 433.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 433.60 | 434.67 | 433.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:30:00 | 433.70 | 434.67 | 433.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 434.60 | 434.65 | 433.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 436.65 | 434.70 | 433.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 436.55 | 437.88 | 437.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 436.55 | 437.88 | 437.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 13:15:00 | 433.50 | 436.58 | 437.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 11:15:00 | 430.05 | 429.98 | 432.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 12:00:00 | 430.05 | 429.98 | 432.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 431.40 | 430.38 | 431.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:45:00 | 431.60 | 430.38 | 431.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 428.05 | 426.35 | 428.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 428.05 | 426.35 | 428.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 427.95 | 426.67 | 428.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:00:00 | 426.90 | 426.72 | 428.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:30:00 | 426.75 | 426.82 | 427.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 425.55 | 426.82 | 427.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 432.55 | 427.99 | 428.23 | SL hit (close>static) qty=1.00 sl=429.35 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 431.00 | 428.87 | 428.60 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 421.55 | 428.15 | 428.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 408.80 | 421.85 | 425.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 15:15:00 | 419.85 | 418.76 | 422.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 09:15:00 | 422.80 | 418.76 | 422.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 430.65 | 421.13 | 423.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 430.65 | 421.13 | 423.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 433.70 | 423.65 | 424.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 433.70 | 423.65 | 424.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 429.15 | 424.75 | 424.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 436.10 | 429.34 | 427.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 437.05 | 437.90 | 435.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:45:00 | 436.70 | 437.90 | 435.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 435.50 | 437.26 | 435.91 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 14:15:00 | 433.10 | 435.10 | 435.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 15:15:00 | 432.75 | 434.63 | 435.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 11:15:00 | 432.95 | 432.93 | 433.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-13 12:00:00 | 432.95 | 432.93 | 433.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 432.05 | 431.23 | 431.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 433.45 | 431.23 | 431.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 432.75 | 431.71 | 431.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 432.45 | 431.71 | 431.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 426.60 | 429.53 | 430.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 425.70 | 429.53 | 430.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:45:00 | 425.60 | 428.78 | 430.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 12:45:00 | 425.85 | 427.92 | 429.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 425.40 | 423.43 | 423.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 425.40 | 423.43 | 423.28 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 421.95 | 423.44 | 423.49 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 426.40 | 424.03 | 423.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 426.90 | 424.85 | 424.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 424.50 | 425.44 | 424.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 14:15:00 | 424.50 | 425.44 | 424.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 424.50 | 425.44 | 424.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:30:00 | 424.90 | 425.44 | 424.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 424.95 | 425.34 | 424.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 425.90 | 425.34 | 424.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 12:45:00 | 425.75 | 425.43 | 425.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 426.45 | 426.63 | 426.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:15:00 | 425.75 | 426.63 | 426.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 425.75 | 426.45 | 426.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 427.70 | 426.45 | 426.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-18 11:15:00 | 468.49 | 465.91 | 463.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 466.30 | 468.75 | 468.89 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 475.20 | 470.10 | 469.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 476.80 | 471.44 | 470.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 10:15:00 | 485.70 | 490.74 | 485.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 10:15:00 | 485.70 | 490.74 | 485.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 485.70 | 490.74 | 485.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 485.70 | 490.74 | 485.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 489.20 | 490.43 | 486.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:45:00 | 491.55 | 490.74 | 486.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 490.25 | 490.72 | 487.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 490.35 | 490.72 | 487.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 490.70 | 491.07 | 488.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 496.15 | 497.24 | 494.85 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 490.20 | 493.47 | 493.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 490.20 | 493.47 | 493.87 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 15:15:00 | 495.50 | 493.67 | 493.60 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 492.15 | 493.36 | 493.47 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 12:15:00 | 494.85 | 493.52 | 493.50 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 490.90 | 493.11 | 493.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 483.95 | 487.72 | 489.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 487.95 | 487.15 | 488.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 487.95 | 487.15 | 488.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 487.95 | 487.15 | 488.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 485.70 | 487.20 | 488.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 11:15:00 | 490.05 | 487.77 | 488.84 | SL hit (close>static) qty=1.00 sl=489.95 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 492.65 | 489.18 | 489.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 496.35 | 493.03 | 491.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 494.30 | 495.10 | 493.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 494.30 | 495.10 | 493.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 494.30 | 495.10 | 493.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:45:00 | 496.70 | 495.67 | 494.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 496.95 | 496.41 | 495.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 490.25 | 494.41 | 494.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 490.25 | 494.41 | 494.78 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 497.75 | 494.37 | 494.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 498.80 | 495.26 | 494.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 15:15:00 | 500.00 | 502.04 | 499.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 15:15:00 | 500.00 | 502.04 | 499.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 500.00 | 502.04 | 499.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:00:00 | 503.15 | 502.26 | 500.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 499.05 | 501.31 | 499.99 | SL hit (close<static) qty=1.00 sl=499.60 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 501.40 | 504.05 | 504.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 15:15:00 | 500.40 | 502.24 | 503.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 15:15:00 | 500.15 | 499.66 | 501.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 09:45:00 | 499.80 | 499.51 | 501.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 500.75 | 499.76 | 500.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 500.75 | 499.76 | 500.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 503.30 | 500.47 | 501.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:30:00 | 502.95 | 500.47 | 501.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 507.05 | 501.78 | 501.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 510.95 | 504.32 | 503.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 509.30 | 510.87 | 508.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 509.30 | 510.87 | 508.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 508.60 | 510.42 | 508.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 506.85 | 510.42 | 508.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 506.75 | 509.68 | 508.55 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 507.20 | 507.86 | 507.95 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 509.35 | 508.02 | 508.00 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 501.45 | 507.30 | 507.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 499.40 | 505.17 | 506.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 505.35 | 503.47 | 505.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 505.35 | 503.47 | 505.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 505.35 | 503.47 | 505.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 505.35 | 503.47 | 505.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 506.45 | 504.06 | 505.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 506.45 | 504.06 | 505.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 506.90 | 504.63 | 505.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 506.90 | 504.63 | 505.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 510.60 | 505.83 | 505.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 510.60 | 505.83 | 505.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 13:15:00 | 512.15 | 507.09 | 506.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 513.10 | 510.35 | 508.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 514.15 | 515.28 | 512.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 514.15 | 515.28 | 512.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 512.55 | 514.22 | 513.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 512.55 | 514.22 | 513.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 511.15 | 513.60 | 512.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 511.15 | 513.60 | 512.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 509.25 | 512.73 | 512.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 509.25 | 512.73 | 512.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 513.10 | 514.78 | 513.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 513.10 | 514.78 | 513.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 514.40 | 514.70 | 513.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:00:00 | 515.00 | 514.07 | 513.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 511.75 | 513.55 | 513.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 511.75 | 513.55 | 513.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 509.95 | 512.50 | 513.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 510.10 | 509.34 | 510.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 510.10 | 509.34 | 510.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 510.10 | 509.34 | 510.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 510.10 | 509.34 | 510.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 511.65 | 509.80 | 510.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 511.65 | 509.80 | 510.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 510.75 | 509.99 | 510.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:45:00 | 508.90 | 509.51 | 510.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 513.75 | 509.37 | 510.00 | SL hit (close>static) qty=1.00 sl=511.65 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 514.75 | 510.94 | 510.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 516.75 | 512.67 | 511.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 515.50 | 516.26 | 514.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 515.50 | 516.26 | 514.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 515.50 | 516.26 | 514.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 515.50 | 516.26 | 514.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 514.95 | 515.99 | 514.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 514.95 | 515.99 | 514.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 514.85 | 515.77 | 514.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:30:00 | 516.00 | 515.71 | 514.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:15:00 | 515.90 | 515.71 | 514.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 513.30 | 515.26 | 514.91 | SL hit (close<static) qty=1.00 sl=514.20 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 515.90 | 520.12 | 520.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 514.95 | 517.69 | 518.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 13:15:00 | 514.20 | 514.07 | 515.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 13:45:00 | 513.70 | 514.07 | 515.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 515.90 | 513.89 | 515.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 515.90 | 513.89 | 515.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 514.35 | 513.98 | 515.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 513.90 | 513.98 | 515.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 516.90 | 514.57 | 515.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 516.90 | 514.57 | 515.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 504.15 | 512.48 | 514.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:45:00 | 501.40 | 507.51 | 509.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:30:00 | 501.50 | 506.27 | 508.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:45:00 | 501.10 | 504.46 | 507.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 15:15:00 | 496.75 | 495.10 | 495.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 496.75 | 495.10 | 495.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 498.70 | 496.00 | 495.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 496.20 | 496.28 | 495.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 496.20 | 496.28 | 495.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 496.20 | 496.28 | 495.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 496.20 | 496.28 | 495.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 497.95 | 496.61 | 495.90 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 493.05 | 495.48 | 495.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 492.25 | 494.45 | 495.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 12:15:00 | 486.75 | 485.79 | 489.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:00:00 | 486.75 | 485.79 | 489.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 486.20 | 484.90 | 486.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 486.20 | 484.90 | 486.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 483.65 | 484.20 | 486.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 485.85 | 484.20 | 486.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 484.30 | 484.43 | 485.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 14:00:00 | 482.75 | 484.28 | 485.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 490.50 | 478.00 | 479.53 | SL hit (close>static) qty=1.00 sl=485.90 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 11:15:00 | 485.25 | 481.31 | 480.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 490.70 | 487.52 | 485.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 488.10 | 489.84 | 487.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 488.10 | 489.84 | 487.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 488.10 | 489.84 | 487.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:45:00 | 488.00 | 489.84 | 487.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 488.55 | 489.58 | 487.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:15:00 | 488.00 | 489.58 | 487.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 486.50 | 488.97 | 487.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 486.15 | 488.97 | 487.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 485.90 | 488.35 | 487.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:30:00 | 485.85 | 488.35 | 487.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 484.05 | 488.13 | 487.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 484.05 | 488.13 | 487.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 482.50 | 487.01 | 487.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 479.20 | 483.66 | 485.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 15:15:00 | 480.95 | 480.85 | 482.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 09:15:00 | 481.75 | 480.85 | 482.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 480.70 | 480.94 | 482.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:30:00 | 479.70 | 480.27 | 481.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 10:30:00 | 480.10 | 478.61 | 479.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 11:15:00 | 479.95 | 478.61 | 479.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 472.65 | 469.23 | 469.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 472.65 | 469.23 | 469.12 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 466.60 | 468.91 | 469.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 458.15 | 466.76 | 468.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 461.45 | 460.35 | 463.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 461.45 | 460.35 | 463.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 466.50 | 461.60 | 463.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 466.50 | 461.60 | 463.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 471.30 | 463.54 | 464.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 471.30 | 463.54 | 464.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 475.25 | 465.88 | 465.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 478.05 | 469.61 | 466.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 473.45 | 475.83 | 473.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 473.45 | 475.83 | 473.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 473.45 | 475.83 | 473.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 473.45 | 475.83 | 473.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 475.10 | 475.69 | 473.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 474.70 | 475.69 | 473.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 473.35 | 475.22 | 473.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 473.35 | 475.22 | 473.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 476.50 | 475.48 | 474.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:30:00 | 477.10 | 475.47 | 474.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:45:00 | 478.00 | 475.72 | 474.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 15:15:00 | 476.95 | 475.72 | 474.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 09:15:00 | 473.25 | 474.92 | 474.87 | SL hit (close<static) qty=1.00 sl=473.30 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 09:15:00 | 468.40 | 475.29 | 475.69 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 473.50 | 471.30 | 471.11 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 467.80 | 470.73 | 470.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 12:15:00 | 465.90 | 469.32 | 470.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 465.75 | 465.26 | 466.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 465.75 | 465.26 | 466.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 466.75 | 465.57 | 466.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 463.95 | 465.67 | 466.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:00:00 | 464.60 | 465.08 | 465.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 12:15:00 | 467.85 | 461.54 | 462.97 | SL hit (close>static) qty=1.00 sl=467.35 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 469.80 | 464.86 | 464.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 11:15:00 | 473.00 | 468.57 | 467.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 469.55 | 469.97 | 468.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 14:15:00 | 469.55 | 469.97 | 468.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 469.55 | 469.97 | 468.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 468.50 | 469.97 | 468.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 468.80 | 469.85 | 468.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 468.80 | 469.85 | 468.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 469.45 | 469.77 | 468.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:00:00 | 471.70 | 470.17 | 469.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 14:00:00 | 471.65 | 470.47 | 469.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 15:15:00 | 471.25 | 470.37 | 469.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 466.80 | 469.69 | 469.32 | SL hit (close<static) qty=1.00 sl=468.50 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 467.20 | 468.99 | 469.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 15:15:00 | 466.00 | 467.79 | 468.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 15:15:00 | 465.90 | 464.94 | 466.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 471.20 | 464.94 | 466.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 470.50 | 466.05 | 466.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 468.90 | 466.05 | 466.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 10:15:00 | 474.20 | 467.68 | 467.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 11:15:00 | 475.00 | 469.15 | 468.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 11:15:00 | 477.20 | 477.62 | 475.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 12:00:00 | 477.20 | 477.62 | 475.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 478.00 | 477.47 | 476.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 479.85 | 477.96 | 476.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 479.70 | 478.90 | 477.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 10:30:00 | 480.30 | 479.10 | 477.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 476.35 | 477.47 | 477.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 476.35 | 477.47 | 477.48 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 477.80 | 477.54 | 477.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 11:15:00 | 479.45 | 477.92 | 477.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 483.10 | 483.82 | 482.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:30:00 | 483.70 | 483.82 | 482.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 484.50 | 486.16 | 484.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:30:00 | 485.10 | 486.16 | 484.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 484.55 | 485.83 | 484.33 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 480.45 | 483.41 | 483.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 461.65 | 479.14 | 481.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 12:15:00 | 444.50 | 442.76 | 450.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 12:45:00 | 445.00 | 442.76 | 450.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 450.85 | 445.22 | 450.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 449.65 | 445.22 | 450.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 449.65 | 446.10 | 450.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 449.60 | 446.10 | 450.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 453.65 | 447.61 | 450.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 453.65 | 447.61 | 450.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 458.75 | 449.84 | 451.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 458.75 | 449.84 | 451.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 450.15 | 450.71 | 451.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 446.85 | 450.66 | 451.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 13:15:00 | 440.70 | 437.70 | 437.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 440.70 | 437.70 | 437.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 441.25 | 439.22 | 438.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 437.35 | 439.34 | 438.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 14:15:00 | 437.35 | 439.34 | 438.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 437.35 | 439.34 | 438.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:30:00 | 436.65 | 439.34 | 438.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 437.95 | 439.06 | 438.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 441.30 | 439.06 | 438.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 12:00:00 | 438.55 | 439.00 | 438.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 13:00:00 | 439.05 | 439.01 | 438.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 14:15:00 | 436.95 | 438.34 | 438.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 14:15:00 | 436.95 | 438.34 | 438.49 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 440.00 | 438.69 | 438.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 441.00 | 439.16 | 438.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 14:15:00 | 441.35 | 441.65 | 440.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 14:15:00 | 441.35 | 441.65 | 440.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 441.35 | 441.65 | 440.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 441.20 | 441.65 | 440.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 441.45 | 441.57 | 440.73 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 437.60 | 440.34 | 440.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 11:15:00 | 435.80 | 439.06 | 439.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 436.00 | 434.37 | 435.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 436.00 | 434.37 | 435.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 436.00 | 434.37 | 435.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 436.00 | 434.37 | 435.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 437.35 | 434.97 | 435.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 437.35 | 434.97 | 435.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 435.35 | 435.04 | 435.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:45:00 | 432.75 | 434.48 | 435.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 441.60 | 436.55 | 436.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 441.60 | 436.55 | 436.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 13:15:00 | 446.90 | 441.03 | 438.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 10:15:00 | 455.95 | 457.28 | 450.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 11:00:00 | 455.95 | 457.28 | 450.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 450.85 | 454.42 | 452.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 449.85 | 454.42 | 452.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 451.15 | 453.76 | 452.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 449.90 | 453.76 | 452.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 453.50 | 454.40 | 452.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 453.50 | 454.40 | 452.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 454.95 | 454.51 | 453.04 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 448.85 | 452.36 | 452.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 13:15:00 | 447.60 | 451.41 | 452.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 11:15:00 | 412.75 | 412.40 | 417.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 11:45:00 | 413.25 | 412.40 | 417.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 414.85 | 411.99 | 415.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:45:00 | 413.20 | 411.99 | 415.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 411.90 | 411.97 | 414.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 409.55 | 411.49 | 414.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 13:45:00 | 410.40 | 410.95 | 413.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 410.40 | 410.84 | 413.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:45:00 | 410.40 | 410.18 | 412.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 408.20 | 408.80 | 410.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 406.75 | 408.80 | 410.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 405.80 | 408.20 | 410.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:45:00 | 406.70 | 406.73 | 408.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:00:00 | 406.85 | 407.08 | 408.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 400.95 | 401.13 | 402.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 401.70 | 401.13 | 402.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 402.40 | 401.59 | 402.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:30:00 | 403.10 | 401.59 | 402.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 402.25 | 401.73 | 402.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 402.50 | 401.73 | 402.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 400.95 | 401.63 | 402.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 402.30 | 401.63 | 402.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 402.60 | 401.82 | 402.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 402.60 | 401.82 | 402.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 402.80 | 402.02 | 402.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 402.80 | 402.02 | 402.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 404.75 | 402.57 | 402.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 404.75 | 402.57 | 402.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-25 13:15:00 | 406.20 | 403.29 | 402.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 13:15:00 | 406.20 | 403.29 | 402.92 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 401.15 | 402.86 | 402.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 395.95 | 400.63 | 401.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 396.90 | 396.31 | 398.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 396.90 | 396.31 | 398.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 396.90 | 396.31 | 398.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 398.25 | 396.31 | 398.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 398.25 | 396.94 | 398.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 398.25 | 396.94 | 398.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 397.40 | 397.03 | 398.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 395.95 | 397.09 | 397.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:15:00 | 396.10 | 395.56 | 396.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 12:15:00 | 401.45 | 397.01 | 396.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 401.45 | 397.01 | 396.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 402.85 | 398.64 | 397.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 403.20 | 403.81 | 401.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 11:45:00 | 403.65 | 403.81 | 401.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 406.65 | 404.28 | 402.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:45:00 | 407.65 | 405.26 | 404.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:45:00 | 406.90 | 406.43 | 405.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 11:00:00 | 406.80 | 406.51 | 405.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 15:15:00 | 407.75 | 409.29 | 409.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 15:15:00 | 407.75 | 409.29 | 409.43 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 410.65 | 409.71 | 409.60 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 14:15:00 | 409.00 | 409.47 | 409.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 09:15:00 | 406.00 | 408.76 | 409.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 13:15:00 | 405.60 | 404.17 | 405.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 13:15:00 | 405.60 | 404.17 | 405.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 405.60 | 404.17 | 405.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:00:00 | 405.60 | 404.17 | 405.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 403.80 | 404.10 | 405.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 11:45:00 | 403.05 | 403.85 | 404.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:15:00 | 402.75 | 403.75 | 404.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 14:15:00 | 406.00 | 403.93 | 404.64 | SL hit (close>static) qty=1.00 sl=405.60 alert=retest2 |

### Cycle 57 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 408.60 | 405.72 | 405.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 11:15:00 | 411.45 | 406.86 | 405.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 410.00 | 410.40 | 408.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 12:15:00 | 409.95 | 410.40 | 408.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 409.70 | 410.40 | 409.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 409.35 | 410.40 | 409.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 409.20 | 410.16 | 409.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 408.95 | 410.16 | 409.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 409.45 | 410.02 | 409.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 408.85 | 410.02 | 409.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 408.60 | 409.73 | 409.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 408.05 | 409.73 | 409.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 409.50 | 409.69 | 409.08 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 407.40 | 408.78 | 408.82 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 410.00 | 409.03 | 408.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 412.35 | 410.02 | 409.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 409.40 | 410.52 | 409.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 12:15:00 | 409.40 | 410.52 | 409.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 409.40 | 410.52 | 409.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 409.40 | 410.52 | 409.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 409.95 | 410.41 | 409.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 409.95 | 410.41 | 409.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 410.00 | 410.32 | 409.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:30:00 | 409.80 | 410.32 | 409.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 410.25 | 410.31 | 409.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 411.25 | 410.31 | 409.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 412.60 | 410.77 | 410.19 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 406.75 | 409.65 | 409.80 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 409.75 | 408.98 | 408.96 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 13:15:00 | 407.25 | 408.76 | 408.87 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 09:15:00 | 409.05 | 408.96 | 408.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 10:15:00 | 411.30 | 409.43 | 409.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 409.05 | 409.37 | 409.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 12:15:00 | 409.05 | 409.37 | 409.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 409.05 | 409.37 | 409.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:45:00 | 408.95 | 409.37 | 409.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 409.00 | 409.30 | 409.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 408.25 | 409.30 | 409.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 409.55 | 409.35 | 409.20 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 400.70 | 407.61 | 408.44 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 413.35 | 407.87 | 407.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 414.75 | 409.24 | 408.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 419.35 | 419.94 | 417.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 419.35 | 419.94 | 417.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 422.50 | 420.86 | 418.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 11:45:00 | 423.20 | 421.66 | 419.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 13:45:00 | 424.15 | 422.40 | 420.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:45:00 | 423.25 | 422.71 | 420.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:30:00 | 423.85 | 423.17 | 421.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 422.60 | 424.62 | 423.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:45:00 | 422.20 | 424.62 | 423.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 422.10 | 424.12 | 423.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:30:00 | 423.45 | 423.60 | 422.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:00:00 | 423.30 | 423.60 | 422.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 15:15:00 | 422.95 | 423.37 | 422.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 424.80 | 428.54 | 429.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 424.80 | 428.54 | 429.02 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 428.75 | 428.56 | 428.55 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 425.40 | 427.93 | 428.27 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 429.10 | 427.70 | 427.55 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 425.50 | 427.26 | 427.36 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 430.30 | 427.96 | 427.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 434.00 | 429.70 | 428.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 435.75 | 435.76 | 433.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 12:00:00 | 435.75 | 435.76 | 433.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 433.00 | 435.11 | 433.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 434.10 | 435.11 | 433.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 432.30 | 434.55 | 433.59 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 430.75 | 432.99 | 433.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 15:15:00 | 430.15 | 431.92 | 432.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 12:15:00 | 429.70 | 429.07 | 430.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 13:00:00 | 429.70 | 429.07 | 430.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 429.30 | 429.12 | 430.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 429.55 | 429.12 | 430.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 430.15 | 429.32 | 430.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:30:00 | 430.05 | 429.32 | 430.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 429.40 | 429.34 | 430.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 427.10 | 429.34 | 430.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 433.85 | 427.42 | 428.41 | SL hit (close>static) qty=1.00 sl=431.25 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 432.20 | 429.41 | 429.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 434.55 | 431.09 | 430.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 431.35 | 432.43 | 431.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 431.35 | 432.43 | 431.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 431.35 | 432.43 | 431.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 430.70 | 432.43 | 431.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 431.60 | 432.27 | 431.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 430.30 | 432.27 | 431.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 430.60 | 431.93 | 431.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 430.60 | 431.93 | 431.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 426.80 | 430.91 | 430.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 425.80 | 430.91 | 430.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 427.35 | 430.20 | 430.36 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 434.60 | 430.28 | 429.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 435.85 | 433.39 | 431.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 434.05 | 434.86 | 433.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 434.05 | 434.86 | 433.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 433.90 | 434.67 | 433.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:30:00 | 433.85 | 434.67 | 433.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 433.05 | 434.35 | 433.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 433.05 | 434.35 | 433.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 434.75 | 434.43 | 433.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:15:00 | 436.10 | 434.43 | 433.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 435.95 | 435.92 | 435.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 435.85 | 435.77 | 435.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 433.20 | 434.85 | 434.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 433.20 | 434.85 | 434.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 432.40 | 434.36 | 434.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 434.90 | 428.80 | 430.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 434.90 | 428.80 | 430.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 434.90 | 428.80 | 430.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 434.90 | 428.80 | 430.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 438.00 | 430.64 | 431.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 438.00 | 430.64 | 431.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 437.15 | 431.94 | 431.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 439.30 | 435.53 | 433.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 437.85 | 439.53 | 437.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 437.85 | 439.53 | 437.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 437.85 | 439.53 | 437.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 437.85 | 439.53 | 437.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 440.60 | 439.74 | 437.41 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 434.70 | 436.37 | 436.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 421.65 | 433.43 | 435.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 418.75 | 418.15 | 420.75 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 09:15:00 | 415.30 | 418.15 | 420.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 419.70 | 418.29 | 420.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 420.20 | 418.68 | 420.14 | SL hit (close>ema400) qty=1.00 sl=420.14 alert=retest1 |

### Cycle 79 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 419.15 | 418.31 | 418.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 419.45 | 418.54 | 418.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 426.00 | 426.40 | 424.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 426.00 | 426.40 | 424.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 423.75 | 425.73 | 424.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 423.75 | 425.73 | 424.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 421.15 | 424.82 | 424.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 421.15 | 424.82 | 424.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 420.95 | 424.04 | 424.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 421.38 | 422.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 417.85 | 416.98 | 419.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 417.85 | 416.98 | 419.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 418.45 | 417.41 | 418.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 418.60 | 417.41 | 418.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 418.00 | 417.82 | 418.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 416.10 | 417.32 | 418.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 416.25 | 417.12 | 417.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 415.80 | 417.06 | 417.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 415.85 | 416.64 | 416.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 418.35 | 416.82 | 417.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 418.35 | 416.82 | 417.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 418.10 | 417.08 | 417.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 418.40 | 417.08 | 417.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 418.90 | 417.44 | 417.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 418.90 | 417.44 | 417.26 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 414.50 | 417.12 | 417.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 15:15:00 | 413.40 | 415.70 | 416.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 417.90 | 416.14 | 416.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 417.90 | 416.14 | 416.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 417.90 | 416.14 | 416.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 417.90 | 416.14 | 416.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 418.45 | 416.60 | 416.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 418.45 | 416.60 | 416.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 420.10 | 417.30 | 417.02 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 414.50 | 416.68 | 416.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 14:15:00 | 414.15 | 416.17 | 416.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 417.95 | 416.22 | 416.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 10:15:00 | 417.95 | 416.22 | 416.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 417.95 | 416.22 | 416.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 417.95 | 416.22 | 416.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 416.75 | 416.33 | 416.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:30:00 | 416.45 | 416.50 | 416.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 417.25 | 416.65 | 416.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 417.25 | 416.65 | 416.60 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 15:15:00 | 416.20 | 416.55 | 416.56 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 418.65 | 416.97 | 416.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 419.15 | 417.66 | 417.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 419.05 | 419.14 | 418.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:30:00 | 418.95 | 419.14 | 418.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 418.65 | 419.04 | 418.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 418.65 | 419.04 | 418.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 418.00 | 418.93 | 418.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 418.00 | 418.93 | 418.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 418.35 | 418.82 | 418.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 418.35 | 418.82 | 418.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 416.10 | 417.86 | 418.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 415.00 | 416.39 | 417.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 12:15:00 | 414.10 | 413.87 | 414.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:30:00 | 414.05 | 413.87 | 414.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 414.15 | 413.92 | 414.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 414.90 | 413.92 | 414.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 413.60 | 413.77 | 414.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 412.35 | 413.48 | 414.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 415.35 | 413.44 | 413.85 | SL hit (close>static) qty=1.00 sl=415.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 416.55 | 414.47 | 414.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 418.00 | 416.53 | 415.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 418.00 | 418.54 | 417.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:00:00 | 418.00 | 418.54 | 417.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 417.20 | 418.27 | 417.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 417.20 | 418.27 | 417.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 416.70 | 417.96 | 417.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:45:00 | 417.65 | 417.89 | 417.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 417.55 | 417.48 | 417.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 416.00 | 417.15 | 417.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 416.00 | 417.15 | 417.19 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 418.40 | 417.24 | 417.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 10:15:00 | 419.00 | 417.59 | 417.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 423.20 | 423.40 | 421.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 10:00:00 | 423.20 | 423.40 | 421.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 423.15 | 424.07 | 423.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 422.55 | 423.75 | 422.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 422.20 | 423.44 | 422.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 421.50 | 423.44 | 422.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 421.15 | 422.55 | 422.56 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 422.95 | 422.61 | 422.58 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 422.05 | 422.50 | 422.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 421.25 | 422.14 | 422.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 410.45 | 409.94 | 411.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 410.25 | 409.94 | 411.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 410.15 | 409.80 | 410.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:30:00 | 411.00 | 409.80 | 410.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 409.50 | 408.29 | 408.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 409.50 | 408.29 | 408.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 410.85 | 408.80 | 409.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 411.75 | 408.80 | 409.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 411.00 | 409.24 | 409.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 412.50 | 409.89 | 409.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 13:15:00 | 417.00 | 417.18 | 415.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 417.00 | 417.18 | 415.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 414.05 | 416.50 | 415.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:45:00 | 414.15 | 416.50 | 415.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 413.35 | 415.87 | 415.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 413.60 | 415.87 | 415.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 414.15 | 414.86 | 414.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 412.30 | 414.18 | 414.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 15:15:00 | 412.40 | 412.37 | 413.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 413.45 | 412.37 | 413.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 414.00 | 412.70 | 413.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 414.05 | 412.70 | 413.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 412.55 | 412.67 | 413.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:45:00 | 412.35 | 412.66 | 413.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 414.20 | 413.52 | 413.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 414.20 | 413.52 | 413.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 416.15 | 414.42 | 413.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 416.95 | 417.00 | 416.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 416.40 | 416.88 | 416.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 416.40 | 416.88 | 416.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 416.40 | 416.88 | 416.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 416.10 | 416.72 | 416.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 415.80 | 416.59 | 416.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 414.85 | 416.24 | 415.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 414.95 | 416.24 | 415.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 414.00 | 415.79 | 415.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 414.00 | 415.79 | 415.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 415.05 | 415.64 | 415.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 413.00 | 414.43 | 415.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 407.80 | 407.53 | 409.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 407.80 | 407.53 | 409.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 409.40 | 408.57 | 409.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:15:00 | 408.40 | 408.62 | 409.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 402.00 | 401.48 | 401.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 402.00 | 401.48 | 401.45 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 400.80 | 401.38 | 401.43 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 404.90 | 402.08 | 401.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 12:15:00 | 408.20 | 404.62 | 403.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 10:15:00 | 407.40 | 407.43 | 405.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 407.40 | 407.43 | 405.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 405.55 | 407.05 | 405.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 405.55 | 407.05 | 405.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 405.15 | 406.67 | 405.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:45:00 | 405.20 | 406.67 | 405.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 406.00 | 406.54 | 405.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 406.45 | 406.37 | 405.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:00:00 | 406.80 | 406.37 | 405.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:45:00 | 406.35 | 406.46 | 405.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 12:30:00 | 406.30 | 406.45 | 405.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 405.00 | 406.16 | 405.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 405.00 | 406.16 | 405.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 406.80 | 406.29 | 405.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 409.10 | 406.43 | 405.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 407.90 | 413.34 | 411.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 406.70 | 410.89 | 410.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 406.70 | 410.89 | 410.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 405.45 | 409.80 | 410.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 13:15:00 | 408.65 | 408.34 | 409.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 14:00:00 | 408.65 | 408.34 | 409.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 407.70 | 408.21 | 408.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 408.70 | 408.21 | 408.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 409.45 | 408.48 | 408.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 409.45 | 408.48 | 408.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 409.85 | 408.76 | 408.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 409.75 | 408.76 | 408.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 410.70 | 409.33 | 409.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 411.05 | 409.95 | 409.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 413.20 | 414.47 | 413.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 10:00:00 | 413.20 | 414.47 | 413.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 416.00 | 414.78 | 413.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 416.65 | 414.78 | 413.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 412.50 | 413.60 | 413.46 | SL hit (close<static) qty=1.00 sl=412.85 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 412.60 | 413.22 | 413.30 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 413.70 | 413.39 | 413.36 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 413.10 | 413.33 | 413.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 410.85 | 412.81 | 413.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 411.40 | 410.75 | 411.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 411.40 | 410.75 | 411.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 411.40 | 410.75 | 411.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 411.55 | 410.75 | 411.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 412.85 | 411.17 | 411.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 412.85 | 411.17 | 411.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 411.50 | 411.24 | 411.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:30:00 | 411.15 | 411.19 | 411.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 410.85 | 411.65 | 411.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 405.15 | 403.11 | 402.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 14:15:00 | 405.15 | 403.11 | 402.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 406.45 | 404.08 | 403.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 15:15:00 | 404.40 | 405.21 | 404.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 404.40 | 405.21 | 404.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 404.40 | 405.21 | 404.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 404.05 | 404.70 | 404.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 403.15 | 404.39 | 404.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 402.65 | 404.39 | 404.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 12:15:00 | 402.25 | 403.70 | 403.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 400.90 | 403.14 | 403.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 403.65 | 402.15 | 402.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 403.65 | 402.15 | 402.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 403.65 | 402.15 | 402.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 403.65 | 402.15 | 402.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 405.50 | 402.82 | 403.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 405.50 | 402.82 | 403.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 405.50 | 403.36 | 403.27 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 400.95 | 403.62 | 403.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 400.20 | 402.17 | 402.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 401.40 | 400.76 | 401.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 401.40 | 400.76 | 401.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 401.40 | 400.76 | 401.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 401.40 | 400.76 | 401.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 399.50 | 400.50 | 401.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 398.75 | 400.50 | 401.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 399.05 | 399.81 | 400.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:00:00 | 399.25 | 399.79 | 400.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 402.30 | 400.22 | 400.35 | SL hit (close>static) qty=1.00 sl=401.35 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 402.25 | 400.63 | 400.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 403.20 | 401.14 | 400.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 400.25 | 401.50 | 401.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 400.25 | 401.50 | 401.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 400.25 | 401.50 | 401.14 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 399.10 | 400.76 | 400.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 397.90 | 399.53 | 400.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 399.60 | 398.15 | 398.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 399.60 | 398.15 | 398.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 399.60 | 398.15 | 398.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 399.15 | 398.15 | 398.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 399.75 | 398.47 | 399.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 399.75 | 398.47 | 399.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 399.70 | 398.78 | 399.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 400.00 | 398.78 | 399.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 399.35 | 398.90 | 399.11 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 400.30 | 399.34 | 399.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 401.20 | 399.71 | 399.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 415.00 | 415.60 | 412.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 415.00 | 415.60 | 412.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 414.50 | 415.47 | 413.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 416.40 | 415.50 | 413.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 416.30 | 415.77 | 414.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 416.85 | 415.99 | 414.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 417.00 | 419.59 | 419.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 417.00 | 419.59 | 419.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 413.95 | 418.46 | 419.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 406.35 | 405.77 | 408.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:45:00 | 406.40 | 405.77 | 408.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 407.15 | 406.10 | 407.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 407.15 | 406.10 | 407.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 405.20 | 405.92 | 406.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 404.50 | 406.44 | 406.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 403.95 | 405.94 | 406.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 14:15:00 | 408.65 | 406.34 | 406.38 | SL hit (close>static) qty=1.00 sl=407.20 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 408.25 | 406.72 | 406.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 409.40 | 407.26 | 406.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 407.10 | 407.23 | 406.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:45:00 | 407.30 | 407.23 | 406.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 407.45 | 407.27 | 406.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 407.20 | 407.27 | 406.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 406.70 | 407.16 | 406.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:45:00 | 406.70 | 407.16 | 406.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 407.50 | 407.22 | 406.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 407.15 | 407.22 | 406.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 407.00 | 407.18 | 406.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 406.60 | 407.18 | 406.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 407.50 | 407.24 | 406.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 405.95 | 407.24 | 406.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 405.10 | 406.82 | 406.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 403.85 | 405.59 | 406.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 405.75 | 404.49 | 405.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 405.75 | 404.49 | 405.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 405.75 | 404.49 | 405.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 405.30 | 404.49 | 405.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 406.50 | 404.89 | 405.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 406.95 | 404.89 | 405.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 405.85 | 405.34 | 405.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 406.45 | 405.34 | 405.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 405.90 | 405.45 | 405.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 406.55 | 405.68 | 405.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 10:15:00 | 406.50 | 407.08 | 406.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 406.50 | 407.08 | 406.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 406.50 | 407.08 | 406.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 406.50 | 407.08 | 406.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 405.30 | 406.73 | 406.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 404.50 | 406.73 | 406.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 404.75 | 406.33 | 406.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 405.10 | 406.33 | 406.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 405.00 | 406.07 | 406.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 403.05 | 405.46 | 405.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 403.15 | 402.86 | 404.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 402.60 | 402.68 | 403.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 402.60 | 402.68 | 403.36 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 12:15:00 | 404.00 | 403.49 | 403.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 404.30 | 403.60 | 403.48 | Break + close above crossover candle high |

### Cycle 120 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 401.60 | 403.31 | 403.38 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 404.25 | 403.55 | 403.48 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 401.55 | 403.26 | 403.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 400.40 | 401.66 | 402.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 401.55 | 401.47 | 402.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:00:00 | 401.55 | 401.47 | 402.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 401.45 | 400.97 | 401.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 401.55 | 400.97 | 401.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 403.00 | 401.37 | 401.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 403.00 | 401.37 | 401.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 402.65 | 401.63 | 401.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:30:00 | 402.85 | 401.63 | 401.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 401.55 | 401.62 | 401.78 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 403.10 | 401.91 | 401.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 404.20 | 402.88 | 402.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 403.40 | 403.91 | 403.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 403.40 | 403.91 | 403.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 403.40 | 403.91 | 403.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 403.50 | 403.91 | 403.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 402.50 | 403.86 | 403.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 402.50 | 403.86 | 403.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 401.00 | 403.29 | 403.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 401.00 | 403.29 | 403.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 402.00 | 402.94 | 403.01 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 404.45 | 403.04 | 403.03 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 11:15:00 | 402.95 | 403.02 | 403.02 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 403.85 | 403.19 | 403.10 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 400.35 | 402.54 | 402.81 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 403.30 | 402.91 | 402.90 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 402.00 | 402.85 | 402.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 400.80 | 402.13 | 402.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 402.30 | 401.18 | 401.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 402.30 | 401.18 | 401.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 402.30 | 401.18 | 401.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 402.30 | 401.18 | 401.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 401.75 | 401.30 | 401.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 12:45:00 | 401.50 | 401.46 | 401.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 403.10 | 402.10 | 401.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 403.10 | 402.10 | 401.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 405.10 | 402.70 | 402.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 402.00 | 402.64 | 402.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 402.00 | 402.64 | 402.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 402.00 | 402.64 | 402.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 402.00 | 402.64 | 402.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 403.00 | 402.71 | 402.37 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 399.90 | 401.78 | 402.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 399.65 | 401.04 | 401.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 401.00 | 400.91 | 401.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 401.00 | 400.91 | 401.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 401.00 | 400.91 | 401.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 401.15 | 400.91 | 401.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 400.90 | 400.91 | 401.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 400.90 | 400.91 | 401.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 401.05 | 400.99 | 401.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 401.35 | 400.99 | 401.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 401.20 | 400.82 | 401.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 401.20 | 400.82 | 401.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 401.00 | 400.86 | 401.12 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 402.00 | 401.40 | 401.32 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 15:15:00 | 400.95 | 401.25 | 401.27 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 402.50 | 401.50 | 401.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 403.85 | 402.80 | 402.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 405.85 | 406.48 | 405.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:45:00 | 405.90 | 406.48 | 405.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 405.00 | 406.25 | 405.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 405.15 | 406.25 | 405.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 404.70 | 405.94 | 405.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 405.30 | 405.74 | 405.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 404.05 | 405.40 | 405.16 | SL hit (close<static) qty=1.00 sl=404.35 alert=retest2 |

### Cycle 136 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 404.35 | 404.94 | 404.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 402.55 | 404.46 | 404.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 402.40 | 401.77 | 402.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 402.40 | 401.77 | 402.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 402.40 | 401.77 | 402.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 402.40 | 401.77 | 402.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 402.80 | 401.97 | 402.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 402.65 | 401.97 | 402.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 403.15 | 402.21 | 402.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 403.15 | 402.21 | 402.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 403.20 | 402.41 | 402.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 403.00 | 402.41 | 402.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 379.35 | 398.08 | 400.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 376.35 | 392.60 | 397.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:15:00 | 357.53 | 369.01 | 382.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-06 13:15:00 | 338.72 | 346.26 | 353.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 322.10 | 321.17 | 321.07 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 315.20 | 320.12 | 320.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 311.20 | 317.21 | 319.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 312.10 | 310.80 | 314.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 312.10 | 310.80 | 314.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 313.90 | 311.42 | 314.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:30:00 | 314.25 | 311.42 | 314.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 314.90 | 312.12 | 314.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 314.90 | 312.12 | 314.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 314.75 | 312.64 | 314.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 313.55 | 312.64 | 314.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 313.20 | 312.76 | 314.50 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 316.50 | 315.28 | 315.24 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 313.10 | 314.96 | 315.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 312.15 | 313.65 | 314.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 320.00 | 312.66 | 312.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 10:15:00 | 320.00 | 312.66 | 312.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 320.00 | 312.66 | 312.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 320.00 | 312.66 | 312.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 325.15 | 315.16 | 314.09 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 318.00 | 320.18 | 320.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 317.15 | 319.58 | 320.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 318.95 | 318.45 | 319.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 12:15:00 | 318.95 | 318.45 | 319.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 318.95 | 318.45 | 319.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:45:00 | 319.15 | 318.45 | 319.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 319.60 | 318.68 | 319.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 319.80 | 318.68 | 319.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 317.30 | 318.40 | 318.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 316.65 | 318.20 | 318.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:00:00 | 316.75 | 316.04 | 316.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 326.05 | 318.84 | 317.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 326.05 | 318.84 | 317.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 330.00 | 324.69 | 321.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 328.30 | 329.70 | 326.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 328.30 | 329.70 | 326.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 328.00 | 329.25 | 327.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 328.00 | 329.25 | 327.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 325.80 | 328.56 | 327.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 325.80 | 328.56 | 327.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 325.80 | 328.01 | 326.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 326.55 | 328.01 | 326.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 326.40 | 326.74 | 326.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 326.40 | 326.74 | 326.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 324.65 | 326.32 | 326.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 324.40 | 324.33 | 325.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 324.40 | 324.33 | 325.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 322.95 | 323.92 | 324.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:45:00 | 321.70 | 323.44 | 324.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 305.61 | 309.20 | 310.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 305.95 | 305.84 | 307.78 | SL hit (close>ema200) qty=0.50 sl=305.84 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 308.90 | 308.44 | 308.38 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 307.90 | 308.33 | 308.34 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 309.25 | 308.52 | 308.42 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 305.65 | 308.05 | 308.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 304.50 | 306.58 | 307.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 304.70 | 303.29 | 304.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 304.70 | 303.29 | 304.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 304.70 | 303.29 | 304.74 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-03-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 15:15:00 | 307.45 | 305.27 | 305.11 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 09:15:00 | 304.50 | 305.36 | 305.37 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 306.40 | 305.57 | 305.47 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 13:15:00 | 305.30 | 305.41 | 305.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 14:15:00 | 304.10 | 305.15 | 305.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 304.65 | 301.34 | 302.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 304.65 | 301.34 | 302.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 304.65 | 301.34 | 302.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 304.65 | 301.34 | 302.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 307.00 | 302.47 | 303.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 307.00 | 302.47 | 303.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 307.35 | 303.45 | 303.42 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 300.00 | 303.53 | 303.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 15:15:00 | 299.80 | 302.78 | 303.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 13:15:00 | 292.30 | 291.95 | 295.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:45:00 | 292.05 | 291.95 | 295.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 296.10 | 292.73 | 294.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 296.10 | 292.73 | 294.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 298.25 | 293.84 | 294.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 298.25 | 293.84 | 294.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 295.75 | 295.51 | 295.49 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 293.00 | 295.00 | 295.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 292.55 | 294.40 | 294.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 292.70 | 291.02 | 292.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 292.70 | 291.02 | 292.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 292.70 | 291.02 | 292.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 292.75 | 291.02 | 292.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 292.45 | 291.31 | 292.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 289.55 | 292.07 | 292.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 291.25 | 291.26 | 291.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 293.45 | 291.70 | 291.97 | SL hit (close>static) qty=1.00 sl=293.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 293.45 | 292.17 | 292.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 295.15 | 293.02 | 292.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 299.15 | 303.08 | 302.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 299.15 | 303.08 | 302.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 299.15 | 303.08 | 302.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 299.55 | 303.08 | 302.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 299.25 | 301.45 | 301.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 298.65 | 300.50 | 301.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 302.65 | 300.67 | 301.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 302.65 | 300.67 | 301.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 302.65 | 300.67 | 301.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 302.65 | 300.67 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 301.90 | 300.91 | 301.22 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 303.50 | 301.77 | 301.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 308.00 | 304.14 | 303.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 305.75 | 306.28 | 305.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 305.75 | 306.28 | 305.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 305.20 | 306.06 | 305.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 305.20 | 306.06 | 305.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 304.90 | 305.83 | 305.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 305.65 | 305.83 | 305.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 305.40 | 305.74 | 305.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 12:00:00 | 305.55 | 305.63 | 305.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 13:45:00 | 305.45 | 305.91 | 305.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 306.05 | 306.81 | 306.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:30:00 | 306.50 | 306.81 | 306.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 306.95 | 306.84 | 306.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 304.65 | 305.77 | 305.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 304.65 | 305.77 | 305.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 303.70 | 305.02 | 305.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 305.00 | 303.36 | 304.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 305.00 | 303.36 | 304.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 305.00 | 303.36 | 304.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 305.05 | 303.36 | 304.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 304.10 | 303.51 | 304.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:30:00 | 303.55 | 303.48 | 304.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 303.35 | 303.48 | 304.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 304.40 | 304.32 | 304.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 304.40 | 304.32 | 304.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 12:15:00 | 305.30 | 304.51 | 304.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 15:15:00 | 304.10 | 304.46 | 304.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 15:15:00 | 304.10 | 304.46 | 304.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 304.10 | 304.46 | 304.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 311.50 | 304.46 | 304.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 310.25 | 311.85 | 312.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 310.25 | 311.85 | 312.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 13:15:00 | 308.90 | 309.94 | 310.55 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 10:00:00 | 428.95 | 2024-05-16 14:15:00 | 430.95 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-05-14 15:15:00 | 429.20 | 2024-05-16 14:15:00 | 430.95 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-05-15 11:15:00 | 429.25 | 2024-05-17 10:15:00 | 434.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-05-15 12:00:00 | 429.05 | 2024-05-17 10:15:00 | 434.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-05-16 10:45:00 | 425.95 | 2024-05-17 10:15:00 | 434.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-05-16 13:00:00 | 425.20 | 2024-05-17 10:15:00 | 434.80 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-05-22 09:15:00 | 436.65 | 2024-05-24 14:15:00 | 436.55 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-05-31 12:00:00 | 426.90 | 2024-06-03 09:15:00 | 432.55 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-05-31 13:30:00 | 426.75 | 2024-06-03 09:15:00 | 432.55 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-05-31 14:15:00 | 425.55 | 2024-06-03 09:15:00 | 432.55 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-06-19 10:15:00 | 425.70 | 2024-06-26 10:15:00 | 425.40 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-06-19 10:45:00 | 425.60 | 2024-06-26 10:15:00 | 425.40 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-06-19 12:45:00 | 425.85 | 2024-06-26 10:15:00 | 425.40 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-07-01 09:15:00 | 425.90 | 2024-07-18 11:15:00 | 468.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-01 12:45:00 | 425.75 | 2024-07-18 11:15:00 | 468.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 14:30:00 | 426.45 | 2024-07-18 11:15:00 | 468.33 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2024-07-02 15:15:00 | 425.75 | 2024-07-18 12:15:00 | 469.10 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2024-07-03 09:15:00 | 427.70 | 2024-07-18 14:15:00 | 470.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 12:45:00 | 491.55 | 2024-07-30 14:15:00 | 490.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-07-25 14:45:00 | 490.25 | 2024-07-30 14:15:00 | 490.20 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-07-25 15:15:00 | 490.35 | 2024-07-30 14:15:00 | 490.20 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-07-26 09:45:00 | 490.70 | 2024-07-30 14:15:00 | 490.20 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-08-06 10:30:00 | 485.70 | 2024-08-06 11:15:00 | 490.05 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-08-06 14:30:00 | 484.60 | 2024-08-07 09:15:00 | 491.70 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-08-12 10:45:00 | 496.70 | 2024-08-13 13:15:00 | 490.25 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-13 10:00:00 | 496.95 | 2024-08-13 13:15:00 | 490.25 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-08-20 10:00:00 | 503.15 | 2024-08-20 11:15:00 | 499.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-08-21 10:15:00 | 504.50 | 2024-08-27 11:15:00 | 501.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-08-22 14:00:00 | 503.45 | 2024-08-27 11:15:00 | 501.40 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-08-23 10:30:00 | 503.15 | 2024-08-27 11:15:00 | 501.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-08-26 10:15:00 | 507.50 | 2024-08-27 11:15:00 | 501.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-08-26 13:45:00 | 507.10 | 2024-08-27 11:15:00 | 501.40 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-09-16 10:00:00 | 515.00 | 2024-09-16 11:15:00 | 511.75 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-18 12:45:00 | 508.90 | 2024-09-19 09:15:00 | 513.75 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-19 12:15:00 | 510.00 | 2024-09-20 09:15:00 | 511.90 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-09-19 12:45:00 | 509.80 | 2024-09-20 09:15:00 | 511.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-09-19 14:30:00 | 509.15 | 2024-09-20 09:15:00 | 511.90 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-09-24 12:30:00 | 516.00 | 2024-09-25 09:15:00 | 513.30 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-09-24 13:15:00 | 515.90 | 2024-09-25 09:15:00 | 513.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-09-25 13:45:00 | 516.35 | 2024-09-30 12:15:00 | 515.90 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-10-09 09:45:00 | 501.40 | 2024-10-14 15:15:00 | 496.75 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2024-10-09 10:30:00 | 501.50 | 2024-10-14 15:15:00 | 496.75 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2024-10-09 12:45:00 | 501.10 | 2024-10-14 15:15:00 | 496.75 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2024-10-22 14:00:00 | 482.75 | 2024-10-25 09:15:00 | 490.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-11-07 09:30:00 | 479.70 | 2024-11-19 11:15:00 | 472.65 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2024-11-11 10:30:00 | 480.10 | 2024-11-19 11:15:00 | 472.65 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2024-11-11 11:15:00 | 479.95 | 2024-11-19 11:15:00 | 472.65 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-11-27 13:30:00 | 477.10 | 2024-11-29 09:15:00 | 473.25 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-27 14:45:00 | 478.00 | 2024-11-29 09:15:00 | 473.25 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-27 15:15:00 | 476.95 | 2024-11-29 09:15:00 | 473.25 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-11-29 10:45:00 | 477.45 | 2024-12-03 09:15:00 | 468.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-12-02 11:45:00 | 477.60 | 2024-12-03 09:15:00 | 468.40 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-12-02 13:45:00 | 478.05 | 2024-12-03 09:15:00 | 468.40 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-12 09:15:00 | 463.95 | 2024-12-13 12:15:00 | 467.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-12 11:00:00 | 464.60 | 2024-12-13 12:15:00 | 467.85 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-12-18 13:00:00 | 471.70 | 2024-12-19 10:15:00 | 466.80 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-18 14:00:00 | 471.65 | 2024-12-19 10:15:00 | 466.80 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-12-18 15:15:00 | 471.25 | 2024-12-19 10:15:00 | 466.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-12-27 11:45:00 | 479.85 | 2024-12-31 09:15:00 | 476.35 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-12-27 15:15:00 | 479.70 | 2024-12-31 09:15:00 | 476.35 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-12-30 10:30:00 | 480.30 | 2024-12-31 09:15:00 | 476.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-01-10 09:15:00 | 446.85 | 2025-01-17 13:15:00 | 440.70 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-01-22 09:15:00 | 441.30 | 2025-01-22 14:15:00 | 436.95 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-22 12:00:00 | 438.55 | 2025-01-22 14:15:00 | 436.95 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-01-22 13:00:00 | 439.05 | 2025-01-22 14:15:00 | 436.95 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-01-30 13:45:00 | 432.75 | 2025-01-31 09:15:00 | 441.60 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-02-14 12:00:00 | 409.55 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-02-14 13:45:00 | 410.40 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-02-14 15:00:00 | 410.40 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-02-17 10:45:00 | 410.40 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-02-18 10:15:00 | 406.75 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-02-18 11:00:00 | 405.80 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-02-19 10:45:00 | 406.70 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-02-19 14:00:00 | 406.85 | 2025-02-25 13:15:00 | 406.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-03-04 09:15:00 | 395.95 | 2025-03-05 12:15:00 | 401.45 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-03-05 11:15:00 | 396.10 | 2025-03-05 12:15:00 | 401.45 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-11 09:45:00 | 407.65 | 2025-03-17 15:15:00 | 407.75 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-03-12 09:45:00 | 406.90 | 2025-03-17 15:15:00 | 407.75 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2025-03-12 11:00:00 | 406.80 | 2025-03-17 15:15:00 | 407.75 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-03-21 11:45:00 | 403.05 | 2025-03-21 14:15:00 | 406.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-03-21 13:15:00 | 402.75 | 2025-03-21 14:15:00 | 406.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-04-16 11:45:00 | 423.20 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-04-16 13:45:00 | 424.15 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-04-16 14:45:00 | 423.25 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-04-17 10:30:00 | 423.85 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-04-21 12:30:00 | 423.45 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-04-21 13:00:00 | 423.30 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-04-21 15:15:00 | 422.95 | 2025-04-25 11:15:00 | 424.80 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-05-09 09:15:00 | 427.10 | 2025-05-12 09:15:00 | 433.85 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-19 15:15:00 | 436.10 | 2025-05-21 14:15:00 | 433.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-21 09:30:00 | 435.95 | 2025-05-21 14:15:00 | 433.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-21 11:45:00 | 435.85 | 2025-05-21 14:15:00 | 433.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-06-02 09:15:00 | 415.30 | 2025-06-02 12:15:00 | 420.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-06-03 09:45:00 | 418.30 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-03 11:00:00 | 417.75 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-03 11:30:00 | 418.05 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-03 12:45:00 | 416.50 | 2025-06-05 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-17 12:30:00 | 416.10 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-06-18 11:00:00 | 416.25 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-06-18 12:15:00 | 415.80 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-19 14:30:00 | 415.85 | 2025-06-20 11:15:00 | 418.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-25 12:30:00 | 416.45 | 2025-06-25 13:15:00 | 417.25 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-07-04 13:15:00 | 412.35 | 2025-07-07 09:15:00 | 415.35 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-10 12:45:00 | 417.65 | 2025-07-11 10:15:00 | 416.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-11 09:15:00 | 417.55 | 2025-07-11 10:15:00 | 416.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-08-07 11:45:00 | 412.35 | 2025-08-08 10:15:00 | 414.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-20 12:15:00 | 408.40 | 2025-08-28 11:15:00 | 402.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-09-02 09:30:00 | 406.45 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-09-02 10:00:00 | 406.80 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-09-02 10:45:00 | 406.35 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-02 12:30:00 | 406.30 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-09-03 09:15:00 | 409.10 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-05 09:45:00 | 407.90 | 2025-09-05 11:15:00 | 406.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-09-12 11:15:00 | 416.65 | 2025-09-15 11:15:00 | 412.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-18 13:30:00 | 411.15 | 2025-09-26 14:15:00 | 405.15 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2025-09-19 09:15:00 | 410.85 | 2025-09-26 14:15:00 | 405.15 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-10-08 11:15:00 | 398.75 | 2025-10-10 09:15:00 | 402.30 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-08 14:00:00 | 399.05 | 2025-10-10 09:15:00 | 402.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-10-09 12:00:00 | 399.25 | 2025-10-10 09:15:00 | 402.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-24 12:15:00 | 416.40 | 2025-11-03 10:15:00 | 417.00 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-10-24 13:30:00 | 416.30 | 2025-11-03 10:15:00 | 417.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-10-24 15:00:00 | 416.85 | 2025-11-03 10:15:00 | 417.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-11-14 09:15:00 | 404.50 | 2025-11-14 14:15:00 | 408.65 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-14 10:00:00 | 403.95 | 2025-11-14 14:15:00 | 408.65 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-15 12:45:00 | 401.50 | 2025-12-16 09:15:00 | 403.10 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-12-26 11:45:00 | 405.30 | 2025-12-26 12:15:00 | 404.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-01 10:45:00 | 376.35 | 2026-01-02 09:15:00 | 357.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:45:00 | 376.35 | 2026-01-06 13:15:00 | 338.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 316.65 | 2026-02-17 09:15:00 | 326.05 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-02-16 12:00:00 | 316.75 | 2026-02-17 09:15:00 | 326.05 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2026-02-20 09:15:00 | 326.55 | 2026-02-23 10:15:00 | 326.40 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-02-25 10:45:00 | 321.70 | 2026-03-09 09:15:00 | 305.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 10:45:00 | 321.70 | 2026-03-10 10:15:00 | 305.95 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-04-02 09:15:00 | 289.55 | 2026-04-02 14:15:00 | 293.45 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-02 13:45:00 | 291.25 | 2026-04-02 14:15:00 | 293.45 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-04-21 09:15:00 | 305.65 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-04-21 10:00:00 | 305.40 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-04-21 12:00:00 | 305.55 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-04-21 13:45:00 | 305.45 | 2026-04-23 10:15:00 | 304.65 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-04-27 11:30:00 | 303.55 | 2026-04-28 11:15:00 | 304.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-04-27 12:00:00 | 303.35 | 2026-04-28 11:15:00 | 304.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-04-29 09:15:00 | 311.50 | 2026-05-05 10:15:00 | 310.25 | STOP_HIT | 1.00 | -0.40% |
