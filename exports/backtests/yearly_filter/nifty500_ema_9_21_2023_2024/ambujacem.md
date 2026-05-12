# Ambuja Cements Ltd. (AMBUJACEM)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 443.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 151 |
| ALERT2 | 151 |
| ALERT2_SKIP | 73 |
| ALERT3 | 467 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 186 |
| PARTIAL | 13 |
| TARGET_HIT | 2 |
| STOP_HIT | 190 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 202 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 135
- **Target hits / Stop hits / Partials:** 2 / 187 / 13
- **Avg / median % per leg:** 0.13% / -0.74%
- **Sum % (uncompounded):** 26.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 79 | 24 | 30.4% | 2 | 77 | 0 | -0.00% | -0.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.95% | -1.9% |
| BUY @ 3rd Alert (retest2) | 77 | 24 | 31.2% | 2 | 75 | 0 | 0.02% | 1.7% |
| SELL (all) | 123 | 43 | 35.0% | 0 | 110 | 13 | 0.22% | 27.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.98% | -1.0% |
| SELL @ 3rd Alert (retest2) | 122 | 43 | 35.2% | 0 | 109 | 13 | 0.23% | 28.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.96% | -2.9% |
| retest2 (combined) | 199 | 67 | 33.7% | 2 | 184 | 13 | 0.15% | 29.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 12:15:00 | 407.35 | 408.59 | 408.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 13:15:00 | 405.40 | 407.95 | 408.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 09:15:00 | 404.90 | 403.74 | 405.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 09:15:00 | 404.90 | 403.74 | 405.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 404.90 | 403.74 | 405.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:00:00 | 404.90 | 403.74 | 405.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 404.80 | 403.95 | 405.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:45:00 | 405.00 | 403.95 | 405.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 406.10 | 403.57 | 404.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 14:45:00 | 406.95 | 403.57 | 404.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 406.45 | 404.15 | 404.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 407.00 | 404.15 | 404.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 405.20 | 404.36 | 404.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 11:00:00 | 405.20 | 404.36 | 404.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 406.45 | 404.78 | 404.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 12:00:00 | 406.45 | 404.78 | 404.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 12:15:00 | 403.60 | 404.54 | 404.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 13:30:00 | 402.65 | 404.04 | 404.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 13:45:00 | 401.00 | 401.36 | 402.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 15:00:00 | 402.50 | 401.59 | 402.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 09:15:00 | 415.50 | 404.76 | 403.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 09:15:00 | 415.50 | 404.76 | 403.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 426.70 | 409.15 | 405.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 15:15:00 | 427.00 | 427.08 | 421.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 09:15:00 | 427.70 | 427.08 | 421.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 422.40 | 425.92 | 422.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:00:00 | 422.40 | 425.92 | 422.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 424.15 | 425.57 | 422.22 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 12:15:00 | 416.80 | 420.60 | 421.06 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 12:15:00 | 421.95 | 421.05 | 421.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 13:15:00 | 423.85 | 421.61 | 421.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 12:15:00 | 425.30 | 425.43 | 423.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 13:00:00 | 425.30 | 425.43 | 423.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 427.00 | 429.06 | 427.30 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 15:15:00 | 423.75 | 426.51 | 426.73 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 430.70 | 427.35 | 427.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 435.70 | 430.07 | 428.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 15:15:00 | 438.50 | 438.75 | 436.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:15:00 | 451.05 | 438.75 | 436.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 449.00 | 454.47 | 451.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 449.00 | 454.47 | 451.57 | SL hit (close<ema400) qty=1.00 sl=451.57 alert=retest1 |

### Cycle 7 — SELL (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 09:15:00 | 457.90 | 460.44 | 460.50 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 11:15:00 | 461.25 | 460.64 | 460.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 12:15:00 | 464.00 | 461.31 | 460.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 14:15:00 | 460.75 | 461.74 | 461.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 14:15:00 | 460.75 | 461.74 | 461.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 460.75 | 461.74 | 461.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 15:00:00 | 460.75 | 461.74 | 461.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 461.60 | 461.71 | 461.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:30:00 | 459.00 | 461.13 | 461.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 10:15:00 | 457.35 | 460.37 | 460.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 12:15:00 | 454.15 | 458.59 | 459.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 450.85 | 450.44 | 453.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-21 09:30:00 | 450.45 | 450.44 | 453.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 450.70 | 448.51 | 450.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:15:00 | 449.35 | 448.51 | 450.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 447.70 | 448.35 | 450.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 10:45:00 | 445.30 | 447.68 | 449.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:15:00 | 423.03 | 430.64 | 437.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-26 14:15:00 | 432.25 | 428.50 | 433.58 | SL hit (close>ema200) qty=0.50 sl=428.50 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 440.10 | 434.99 | 434.87 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 432.25 | 434.80 | 435.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 11:15:00 | 431.05 | 434.05 | 434.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 430.60 | 429.96 | 432.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 430.60 | 429.96 | 432.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 430.60 | 429.96 | 432.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:30:00 | 433.40 | 429.96 | 432.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 434.25 | 430.82 | 432.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:45:00 | 434.50 | 430.82 | 432.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 434.80 | 431.62 | 432.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 12:00:00 | 434.80 | 431.62 | 432.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 434.20 | 432.27 | 432.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:00:00 | 434.20 | 432.27 | 432.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2023-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 14:15:00 | 435.75 | 432.97 | 432.94 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 15:15:00 | 433.10 | 433.38 | 433.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 09:15:00 | 432.30 | 433.16 | 433.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 13:15:00 | 433.30 | 431.86 | 432.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 13:15:00 | 433.30 | 431.86 | 432.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 13:15:00 | 433.30 | 431.86 | 432.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 13:45:00 | 432.75 | 431.86 | 432.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 432.50 | 431.99 | 432.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 15:15:00 | 433.00 | 431.99 | 432.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 433.00 | 432.19 | 432.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 09:30:00 | 431.00 | 432.19 | 432.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 11:15:00 | 434.30 | 432.70 | 432.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 11:15:00 | 434.30 | 432.70 | 432.68 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 12:15:00 | 432.05 | 432.57 | 432.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 09:15:00 | 427.60 | 431.15 | 431.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 10:15:00 | 421.85 | 421.15 | 425.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 11:00:00 | 421.85 | 421.15 | 425.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 419.35 | 420.61 | 423.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 14:30:00 | 421.75 | 420.61 | 423.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 421.35 | 419.75 | 420.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 15:00:00 | 421.35 | 419.75 | 420.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 420.60 | 419.92 | 420.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 09:15:00 | 421.00 | 419.92 | 420.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 421.25 | 420.19 | 420.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 10:30:00 | 418.60 | 420.35 | 420.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 13:45:00 | 418.00 | 419.49 | 420.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 09:45:00 | 418.40 | 417.11 | 418.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 11:15:00 | 417.95 | 417.69 | 417.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 416.60 | 417.47 | 417.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-19 11:15:00 | 420.50 | 418.18 | 417.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 420.50 | 418.18 | 417.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 11:15:00 | 422.10 | 419.90 | 419.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 15:15:00 | 420.25 | 420.46 | 419.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 09:15:00 | 418.30 | 420.46 | 419.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 417.15 | 419.80 | 419.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:30:00 | 416.50 | 419.80 | 419.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 418.05 | 419.45 | 419.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 10:30:00 | 417.40 | 419.45 | 419.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 11:15:00 | 417.10 | 418.98 | 419.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 416.00 | 418.33 | 418.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 422.40 | 418.80 | 418.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 422.40 | 418.80 | 418.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 422.40 | 418.80 | 418.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:00:00 | 422.40 | 418.80 | 418.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 426.45 | 420.33 | 419.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 433.70 | 424.72 | 422.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 10:15:00 | 459.00 | 461.67 | 456.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 10:15:00 | 459.00 | 461.67 | 456.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 459.00 | 461.67 | 456.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:45:00 | 459.75 | 461.67 | 456.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 456.30 | 460.14 | 457.92 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 454.00 | 456.72 | 456.73 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 14:15:00 | 460.05 | 456.96 | 456.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 15:15:00 | 463.00 | 458.17 | 457.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 14:15:00 | 471.80 | 472.12 | 468.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-04 15:00:00 | 471.80 | 472.12 | 468.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 469.40 | 470.97 | 469.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:00:00 | 469.40 | 470.97 | 469.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 471.40 | 471.06 | 469.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 14:45:00 | 473.75 | 471.60 | 469.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 09:15:00 | 467.45 | 471.14 | 469.83 | SL hit (close<static) qty=1.00 sl=468.75 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 11:15:00 | 461.70 | 467.62 | 468.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 455.60 | 459.54 | 462.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 461.50 | 459.39 | 461.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 461.50 | 459.39 | 461.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 461.50 | 459.39 | 461.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 461.50 | 459.39 | 461.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 460.05 | 459.52 | 461.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:30:00 | 462.25 | 459.52 | 461.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 443.75 | 438.05 | 441.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 14:00:00 | 443.75 | 438.05 | 441.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 445.75 | 439.59 | 441.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 15:00:00 | 445.75 | 439.59 | 441.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 444.40 | 441.45 | 442.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 11:00:00 | 444.40 | 441.45 | 442.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 11:15:00 | 455.45 | 444.25 | 443.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 464.90 | 457.35 | 452.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 462.20 | 463.09 | 459.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 09:45:00 | 460.85 | 463.09 | 459.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 452.35 | 460.98 | 459.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 452.35 | 460.98 | 459.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 455.05 | 459.80 | 459.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 457.55 | 459.80 | 459.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 11:15:00 | 452.80 | 457.80 | 458.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 11:15:00 | 452.80 | 457.80 | 458.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 448.50 | 454.92 | 456.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 12:15:00 | 439.80 | 437.88 | 443.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-28 13:00:00 | 439.80 | 437.88 | 443.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 445.50 | 439.97 | 442.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:00:00 | 445.50 | 439.97 | 442.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 443.75 | 440.73 | 442.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 13:30:00 | 440.90 | 441.62 | 442.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 441.10 | 441.58 | 442.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 09:45:00 | 441.00 | 441.79 | 442.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 12:15:00 | 444.80 | 443.05 | 442.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 444.80 | 443.05 | 442.98 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 430.15 | 441.12 | 442.21 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 11:15:00 | 441.65 | 436.93 | 436.41 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 12:15:00 | 434.20 | 438.42 | 438.64 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 15:15:00 | 438.80 | 438.16 | 438.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 440.55 | 438.64 | 438.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 12:15:00 | 438.70 | 438.86 | 438.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 12:15:00 | 438.70 | 438.86 | 438.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 438.70 | 438.86 | 438.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 13:00:00 | 438.70 | 438.86 | 438.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 438.55 | 438.80 | 438.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 14:45:00 | 439.05 | 438.75 | 438.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 441.10 | 438.64 | 438.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-18 09:15:00 | 441.35 | 443.58 | 443.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 441.35 | 443.58 | 443.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 10:15:00 | 439.30 | 442.72 | 443.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 420.20 | 420.17 | 423.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 12:00:00 | 420.20 | 420.17 | 423.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 422.20 | 420.68 | 423.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 13:30:00 | 423.35 | 420.68 | 423.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 423.30 | 421.20 | 423.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:45:00 | 424.75 | 421.20 | 423.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 423.70 | 421.70 | 423.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 425.45 | 421.70 | 423.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 428.75 | 423.11 | 424.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:00:00 | 428.75 | 423.11 | 424.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 429.25 | 424.34 | 424.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:45:00 | 429.10 | 424.34 | 424.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 432.90 | 426.05 | 425.24 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 13:15:00 | 423.70 | 427.36 | 427.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 14:15:00 | 419.65 | 425.82 | 426.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 10:15:00 | 427.10 | 425.22 | 426.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 10:15:00 | 427.10 | 425.22 | 426.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 427.10 | 425.22 | 426.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 11:00:00 | 427.10 | 425.22 | 426.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 11:15:00 | 426.30 | 425.44 | 426.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 11:45:00 | 426.60 | 425.44 | 426.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 426.85 | 425.72 | 426.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 13:15:00 | 427.75 | 425.72 | 426.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 427.35 | 426.05 | 426.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 14:45:00 | 426.05 | 425.79 | 426.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 10:15:00 | 430.10 | 426.85 | 426.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 430.10 | 426.85 | 426.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 431.35 | 427.75 | 427.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 429.10 | 430.41 | 429.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 429.10 | 430.41 | 429.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 429.10 | 430.41 | 429.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:00:00 | 429.10 | 430.41 | 429.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 426.20 | 429.57 | 428.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 426.20 | 429.57 | 428.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 427.45 | 429.15 | 428.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 14:15:00 | 428.50 | 429.15 | 428.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 14:15:00 | 430.45 | 432.18 | 432.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 430.45 | 432.18 | 432.22 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 436.05 | 432.77 | 432.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 436.75 | 433.95 | 433.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 444.50 | 445.18 | 442.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 09:15:00 | 445.00 | 445.18 | 442.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 442.00 | 444.55 | 442.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:45:00 | 442.60 | 444.55 | 442.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 442.55 | 444.15 | 442.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:45:00 | 442.20 | 444.15 | 442.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 442.25 | 443.77 | 442.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:00:00 | 442.25 | 443.77 | 442.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 439.50 | 442.71 | 442.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 439.50 | 442.71 | 442.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 439.85 | 442.14 | 441.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 441.10 | 442.14 | 441.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 09:15:00 | 440.50 | 441.81 | 441.85 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 10:15:00 | 442.85 | 442.02 | 441.94 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 09:15:00 | 440.30 | 442.09 | 442.09 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 10:15:00 | 442.20 | 442.11 | 442.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 13:15:00 | 442.60 | 442.28 | 442.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 15:15:00 | 441.50 | 442.24 | 442.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 15:15:00 | 441.50 | 442.24 | 442.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 441.50 | 442.24 | 442.19 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 441.05 | 442.00 | 442.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 436.30 | 440.86 | 441.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 14:15:00 | 437.30 | 435.44 | 437.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 14:15:00 | 437.30 | 435.44 | 437.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 437.30 | 435.44 | 437.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:45:00 | 439.25 | 435.44 | 437.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 438.55 | 436.06 | 437.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:15:00 | 437.50 | 436.06 | 437.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 435.55 | 435.96 | 437.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 10:15:00 | 434.90 | 435.96 | 437.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 15:15:00 | 413.15 | 422.02 | 427.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 14:15:00 | 414.80 | 414.02 | 418.21 | SL hit (close>ema200) qty=0.50 sl=414.02 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 421.80 | 419.18 | 418.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 15:15:00 | 422.90 | 420.66 | 419.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 421.30 | 421.68 | 420.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 12:00:00 | 421.30 | 421.68 | 420.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 421.10 | 423.01 | 421.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 421.10 | 423.01 | 421.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 421.30 | 422.67 | 421.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:30:00 | 421.90 | 422.67 | 421.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 413.05 | 420.74 | 421.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 408.40 | 416.80 | 419.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 418.00 | 415.39 | 417.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 418.00 | 415.39 | 417.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 418.00 | 415.39 | 417.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:15:00 | 419.40 | 415.39 | 417.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 420.85 | 416.48 | 418.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 420.85 | 416.48 | 418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 421.30 | 417.45 | 418.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 420.70 | 417.45 | 418.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-11-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 15:15:00 | 419.50 | 419.18 | 419.17 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 417.95 | 419.01 | 419.10 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 12:15:00 | 419.90 | 419.20 | 419.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 13:15:00 | 421.00 | 419.56 | 419.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 13:15:00 | 421.00 | 421.04 | 420.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 13:15:00 | 421.00 | 421.04 | 420.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 421.00 | 421.04 | 420.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:45:00 | 420.95 | 421.04 | 420.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 421.50 | 421.13 | 420.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 14:45:00 | 420.25 | 421.13 | 420.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 420.70 | 421.08 | 420.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:45:00 | 420.30 | 421.08 | 420.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 420.40 | 420.94 | 420.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:30:00 | 420.00 | 420.94 | 420.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 419.00 | 420.55 | 420.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:00:00 | 419.00 | 420.55 | 420.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 419.40 | 420.32 | 420.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:30:00 | 418.55 | 420.32 | 420.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 13:15:00 | 419.25 | 420.11 | 420.22 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 422.20 | 420.45 | 420.33 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 13:15:00 | 419.15 | 420.36 | 420.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 417.10 | 419.24 | 419.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 14:15:00 | 418.20 | 418.19 | 419.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 15:00:00 | 418.20 | 418.19 | 419.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 421.45 | 418.80 | 419.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:45:00 | 419.85 | 419.22 | 419.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 12:00:00 | 419.70 | 419.32 | 419.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 12:45:00 | 419.25 | 419.33 | 419.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 421.40 | 419.73 | 419.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 421.40 | 419.73 | 419.52 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 417.95 | 419.60 | 419.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 11:15:00 | 417.50 | 418.61 | 419.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 13:15:00 | 418.90 | 418.60 | 419.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 13:15:00 | 418.90 | 418.60 | 419.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 418.90 | 418.60 | 419.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 13:45:00 | 418.85 | 418.60 | 419.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 420.05 | 418.89 | 419.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 15:00:00 | 420.05 | 418.89 | 419.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 420.35 | 419.18 | 419.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:15:00 | 421.90 | 419.18 | 419.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 425.05 | 420.35 | 419.74 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 13:15:00 | 419.95 | 421.04 | 421.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 10:15:00 | 418.30 | 420.04 | 420.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 14:15:00 | 415.60 | 414.75 | 416.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 15:00:00 | 415.60 | 414.75 | 416.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 413.70 | 413.51 | 414.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:00:00 | 413.70 | 413.51 | 414.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 413.25 | 413.46 | 414.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:45:00 | 415.05 | 413.46 | 414.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 415.20 | 413.81 | 414.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:15:00 | 425.35 | 413.81 | 414.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 427.45 | 416.54 | 416.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 10:15:00 | 429.70 | 419.17 | 417.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 11:15:00 | 428.85 | 429.37 | 424.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 12:00:00 | 428.85 | 429.37 | 424.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 498.05 | 503.09 | 498.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:00:00 | 498.05 | 503.09 | 498.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 495.80 | 501.64 | 498.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:45:00 | 496.00 | 501.64 | 498.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 489.00 | 499.11 | 497.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 489.00 | 499.11 | 497.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 494.50 | 496.50 | 496.46 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 494.75 | 496.15 | 496.30 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 501.00 | 497.12 | 496.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 11:15:00 | 503.25 | 498.34 | 497.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 14:15:00 | 501.00 | 502.41 | 500.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 501.00 | 502.41 | 500.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 501.00 | 502.41 | 500.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 499.70 | 502.41 | 500.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 501.00 | 502.13 | 500.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 505.35 | 502.13 | 500.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 10:15:00 | 516.90 | 521.05 | 521.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 10:15:00 | 516.90 | 521.05 | 521.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 11:15:00 | 514.95 | 519.83 | 520.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 13:15:00 | 500.05 | 499.96 | 506.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 13:30:00 | 501.15 | 499.96 | 506.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 506.35 | 502.13 | 506.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:30:00 | 506.75 | 502.13 | 506.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 505.10 | 502.72 | 506.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:00:00 | 503.15 | 502.81 | 505.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 15:15:00 | 503.10 | 502.43 | 504.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 507.25 | 503.50 | 504.88 | SL hit (close>static) qty=1.00 sl=506.95 alert=retest2 |

### Cycle 56 — BUY (started 2023-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 15:15:00 | 508.00 | 505.62 | 505.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 517.85 | 508.07 | 506.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 513.75 | 515.04 | 512.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 513.75 | 515.04 | 512.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 513.75 | 515.04 | 512.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 513.75 | 515.04 | 512.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 516.00 | 515.23 | 513.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 09:15:00 | 514.50 | 515.23 | 513.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 515.30 | 515.24 | 513.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:15:00 | 520.80 | 515.96 | 513.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 14:30:00 | 519.35 | 517.55 | 515.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 10:15:00 | 533.80 | 540.78 | 541.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 533.80 | 540.78 | 541.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 529.40 | 536.21 | 538.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 536.20 | 535.29 | 537.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 10:15:00 | 536.30 | 535.49 | 537.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 536.30 | 535.49 | 537.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:30:00 | 537.90 | 535.49 | 537.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 526.85 | 526.17 | 529.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:15:00 | 525.65 | 526.17 | 529.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 11:15:00 | 533.15 | 528.25 | 529.75 | SL hit (close>static) qty=1.00 sl=531.50 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 13:15:00 | 534.30 | 530.69 | 530.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 14:15:00 | 539.15 | 532.39 | 531.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 11:15:00 | 533.90 | 535.76 | 533.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 11:15:00 | 533.90 | 535.76 | 533.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 11:15:00 | 533.90 | 535.76 | 533.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 12:00:00 | 533.90 | 535.76 | 533.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 12:15:00 | 533.15 | 535.24 | 533.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 12:45:00 | 532.40 | 535.24 | 533.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 13:15:00 | 533.80 | 534.95 | 533.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 09:15:00 | 535.80 | 534.22 | 533.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 09:15:00 | 528.10 | 533.00 | 533.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 528.10 | 533.00 | 533.01 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 533.15 | 533.02 | 533.02 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 12:15:00 | 532.90 | 533.00 | 533.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 13:15:00 | 530.25 | 532.45 | 532.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 15:15:00 | 533.00 | 532.22 | 532.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 15:15:00 | 533.00 | 532.22 | 532.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 533.00 | 532.22 | 532.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:15:00 | 529.95 | 532.22 | 532.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 531.30 | 532.04 | 532.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 13:00:00 | 526.40 | 530.46 | 531.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 15:15:00 | 527.65 | 529.94 | 531.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 536.95 | 530.98 | 531.40 | SL hit (close>static) qty=1.00 sl=534.95 alert=retest2 |

### Cycle 62 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 531.50 | 527.15 | 527.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 537.00 | 529.55 | 528.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 11:15:00 | 528.90 | 535.85 | 533.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 11:15:00 | 528.90 | 535.85 | 533.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 528.90 | 535.85 | 533.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 528.90 | 535.85 | 533.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 526.45 | 533.97 | 533.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 526.45 | 533.97 | 533.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 526.85 | 532.55 | 532.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 515.55 | 525.91 | 529.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 526.60 | 524.98 | 527.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 526.60 | 524.98 | 527.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 526.60 | 524.98 | 527.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 527.85 | 524.98 | 527.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 528.00 | 525.59 | 527.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 532.00 | 525.59 | 527.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 533.95 | 527.26 | 528.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:45:00 | 536.85 | 527.26 | 528.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 531.45 | 528.10 | 528.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:30:00 | 533.35 | 528.10 | 528.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 531.55 | 528.79 | 528.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:30:00 | 534.95 | 528.79 | 528.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 12:15:00 | 535.45 | 530.12 | 529.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 13:15:00 | 541.70 | 532.44 | 530.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 570.50 | 574.39 | 564.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 570.50 | 574.39 | 564.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 562.30 | 574.90 | 568.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:45:00 | 560.00 | 574.90 | 568.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 555.25 | 570.97 | 567.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 14:00:00 | 555.25 | 570.97 | 567.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 559.45 | 568.67 | 566.78 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 09:15:00 | 559.80 | 565.35 | 565.51 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 12:15:00 | 566.55 | 565.52 | 565.51 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 564.90 | 565.50 | 565.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 12:15:00 | 559.60 | 564.14 | 564.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 10:15:00 | 561.55 | 560.18 | 562.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 10:15:00 | 561.55 | 560.18 | 562.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 561.55 | 560.18 | 562.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 11:00:00 | 561.55 | 560.18 | 562.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 11:15:00 | 561.20 | 560.39 | 562.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 12:15:00 | 558.30 | 560.39 | 562.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 13:15:00 | 559.25 | 560.80 | 562.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 14:15:00 | 558.70 | 560.71 | 562.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-06 11:45:00 | 559.60 | 558.73 | 560.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 559.20 | 558.83 | 560.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 12:30:00 | 560.00 | 558.83 | 560.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 562.35 | 559.53 | 560.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:00:00 | 562.35 | 559.53 | 560.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-06 14:15:00 | 566.75 | 560.98 | 560.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 566.75 | 560.98 | 560.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 575.50 | 564.76 | 562.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 14:15:00 | 567.60 | 567.61 | 565.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-07 15:00:00 | 567.60 | 567.61 | 565.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 566.70 | 567.49 | 565.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:30:00 | 565.80 | 567.49 | 565.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 563.35 | 566.66 | 565.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:00:00 | 563.35 | 566.66 | 565.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 567.75 | 566.88 | 565.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 11:30:00 | 563.45 | 566.88 | 565.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 564.70 | 566.61 | 565.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:00:00 | 564.70 | 566.61 | 565.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 562.25 | 565.74 | 565.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:45:00 | 561.40 | 565.74 | 565.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-02-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 15:15:00 | 562.00 | 564.99 | 565.07 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 09:15:00 | 566.35 | 565.26 | 565.19 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 558.15 | 563.84 | 564.55 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 11:15:00 | 572.70 | 565.61 | 565.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 12:15:00 | 574.50 | 567.39 | 566.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 570.90 | 571.24 | 568.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 10:00:00 | 570.90 | 571.24 | 568.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 570.75 | 571.14 | 568.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 11:00:00 | 570.75 | 571.14 | 568.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 565.10 | 569.93 | 568.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 12:00:00 | 565.10 | 569.93 | 568.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 567.25 | 569.40 | 568.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 567.25 | 569.40 | 568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 572.45 | 570.01 | 568.75 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 15:15:00 | 562.00 | 567.47 | 567.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 556.75 | 564.07 | 566.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 568.25 | 564.58 | 565.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 568.25 | 564.58 | 565.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 568.25 | 564.58 | 565.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 568.25 | 564.58 | 565.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 565.85 | 564.84 | 565.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:15:00 | 567.55 | 564.84 | 565.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 571.05 | 566.08 | 566.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 571.05 | 566.08 | 566.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 10:15:00 | 569.00 | 566.66 | 566.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 573.35 | 569.85 | 568.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 13:15:00 | 570.30 | 570.81 | 569.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 13:15:00 | 570.30 | 570.81 | 569.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 570.30 | 570.81 | 569.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 13:45:00 | 570.50 | 570.81 | 569.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 583.50 | 584.58 | 581.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 583.50 | 584.58 | 581.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 580.50 | 583.76 | 581.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 580.50 | 583.76 | 581.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 587.70 | 584.55 | 582.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 15:00:00 | 588.10 | 585.20 | 582.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 15:00:00 | 589.55 | 589.82 | 586.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 09:30:00 | 588.40 | 588.39 | 586.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 10:30:00 | 588.95 | 588.18 | 586.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 585.50 | 587.65 | 586.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 585.50 | 587.65 | 586.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 586.00 | 587.32 | 586.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:45:00 | 585.75 | 587.32 | 586.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 592.65 | 588.37 | 587.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 09:15:00 | 593.00 | 589.29 | 587.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 11:30:00 | 593.60 | 590.82 | 588.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 12:00:00 | 594.00 | 590.82 | 588.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 590.20 | 600.85 | 602.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 590.20 | 600.85 | 602.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 585.65 | 597.81 | 600.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 597.70 | 594.11 | 597.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 10:15:00 | 597.70 | 594.11 | 597.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 597.70 | 594.11 | 597.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 11:00:00 | 597.70 | 594.11 | 597.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 602.65 | 595.81 | 597.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:00:00 | 602.65 | 595.81 | 597.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 603.70 | 597.39 | 598.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:00:00 | 603.70 | 597.39 | 598.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 606.25 | 599.92 | 599.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 612.60 | 603.27 | 601.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 613.70 | 615.10 | 610.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 10:00:00 | 613.70 | 615.10 | 610.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 611.95 | 615.26 | 613.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:15:00 | 612.40 | 615.26 | 613.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 616.00 | 615.40 | 613.62 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 602.95 | 612.40 | 612.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 592.55 | 601.03 | 603.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 580.70 | 571.06 | 579.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 10:15:00 | 580.70 | 571.06 | 579.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 580.70 | 571.06 | 579.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 580.70 | 571.06 | 579.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 580.25 | 572.90 | 579.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 580.25 | 572.90 | 579.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 581.50 | 574.62 | 580.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:00:00 | 581.50 | 574.62 | 580.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 580.95 | 575.89 | 580.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:30:00 | 578.15 | 575.89 | 580.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 582.35 | 577.18 | 580.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 582.35 | 577.18 | 580.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 584.00 | 578.54 | 580.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 590.60 | 578.54 | 580.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 581.50 | 580.29 | 581.15 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 586.85 | 581.89 | 581.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 597.00 | 584.91 | 583.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 588.30 | 589.90 | 586.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 588.30 | 589.90 | 586.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 588.30 | 589.90 | 586.24 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 578.00 | 584.72 | 585.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 571.25 | 578.65 | 581.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 579.65 | 577.81 | 580.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 13:00:00 | 579.65 | 577.81 | 580.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 576.50 | 577.55 | 579.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 14:15:00 | 574.95 | 577.55 | 579.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 587.00 | 579.14 | 579.99 | SL hit (close>static) qty=1.00 sl=580.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 589.50 | 581.21 | 580.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 591.45 | 586.95 | 584.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 588.70 | 588.70 | 586.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 588.70 | 588.70 | 586.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 590.20 | 589.37 | 586.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 11:15:00 | 599.25 | 590.69 | 587.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 12:15:00 | 597.30 | 591.75 | 588.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 13:00:00 | 597.10 | 592.82 | 589.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 14:00:00 | 597.80 | 593.81 | 589.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 630.30 | 632.65 | 628.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 630.00 | 632.65 | 628.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 626.20 | 631.36 | 628.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 626.20 | 631.36 | 628.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 628.10 | 630.71 | 628.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 11:15:00 | 631.40 | 627.69 | 627.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 12:15:00 | 631.85 | 628.15 | 627.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 11:00:00 | 632.45 | 629.00 | 628.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 13:30:00 | 631.35 | 630.13 | 629.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 628.85 | 629.86 | 629.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:30:00 | 632.65 | 630.33 | 629.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 10:00:00 | 632.25 | 630.33 | 629.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 11:15:00 | 632.45 | 630.42 | 629.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 13:15:00 | 622.60 | 629.02 | 629.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 622.60 | 629.02 | 629.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 15:15:00 | 622.00 | 626.93 | 628.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 627.35 | 627.01 | 628.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 627.35 | 627.01 | 628.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 627.35 | 627.01 | 628.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:45:00 | 627.15 | 627.01 | 628.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 625.25 | 626.66 | 627.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 627.45 | 626.66 | 627.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 628.05 | 626.75 | 627.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 13:00:00 | 628.05 | 626.75 | 627.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 628.00 | 627.00 | 627.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 13:45:00 | 628.65 | 627.00 | 627.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 625.10 | 626.62 | 627.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 14:30:00 | 627.00 | 626.62 | 627.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 625.75 | 626.16 | 627.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 11:45:00 | 621.45 | 624.55 | 626.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 627.85 | 617.68 | 616.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 627.85 | 617.68 | 616.66 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 606.65 | 616.16 | 617.23 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 617.30 | 615.75 | 615.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 626.65 | 618.50 | 616.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 640.75 | 641.22 | 634.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:30:00 | 646.55 | 641.22 | 634.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 635.70 | 640.12 | 634.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 634.05 | 640.12 | 634.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 630.25 | 638.14 | 634.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 630.25 | 638.14 | 634.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 633.25 | 637.16 | 634.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:00:00 | 636.25 | 636.98 | 634.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 629.35 | 634.17 | 634.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 629.35 | 634.17 | 634.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 09:15:00 | 618.75 | 628.90 | 631.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 10:15:00 | 622.90 | 622.34 | 625.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-02 10:45:00 | 624.50 | 622.34 | 625.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 624.85 | 622.60 | 625.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 14:00:00 | 624.85 | 622.60 | 625.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 625.30 | 623.14 | 625.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 14:30:00 | 629.75 | 623.14 | 625.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 624.10 | 623.33 | 625.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:15:00 | 627.90 | 623.33 | 625.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 632.35 | 625.13 | 625.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:45:00 | 631.75 | 625.13 | 625.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 626.30 | 625.37 | 625.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:15:00 | 623.35 | 625.37 | 625.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 12:45:00 | 620.05 | 623.38 | 624.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 592.18 | 602.45 | 610.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 589.05 | 602.45 | 610.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 12:15:00 | 598.30 | 596.91 | 602.97 | SL hit (close>ema200) qty=0.50 sl=596.91 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 603.15 | 588.89 | 587.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 605.85 | 592.29 | 588.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 609.05 | 610.90 | 604.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 609.05 | 610.90 | 604.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 607.50 | 610.53 | 606.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 607.35 | 610.53 | 606.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 615.40 | 611.50 | 607.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:15:00 | 616.70 | 613.47 | 610.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 628.95 | 633.57 | 634.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 628.95 | 633.57 | 634.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 627.05 | 630.81 | 632.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 11:15:00 | 629.35 | 629.22 | 631.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 12:00:00 | 629.35 | 629.22 | 631.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 630.95 | 629.63 | 630.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:45:00 | 631.05 | 629.63 | 630.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 630.20 | 629.74 | 630.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 626.85 | 630.05 | 630.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:00:00 | 627.30 | 626.85 | 628.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:45:00 | 627.00 | 624.59 | 626.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 13:15:00 | 638.45 | 628.62 | 628.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 638.45 | 628.62 | 628.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 659.85 | 636.47 | 631.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 627.70 | 655.63 | 647.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 627.70 | 655.63 | 647.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 627.70 | 655.63 | 647.17 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 572.00 | 638.90 | 640.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 537.05 | 618.53 | 630.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 587.65 | 581.38 | 599.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 587.65 | 581.38 | 599.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 596.25 | 586.33 | 598.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 596.25 | 586.33 | 598.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 600.90 | 589.25 | 598.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 609.45 | 589.25 | 598.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 616.50 | 594.70 | 600.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 616.95 | 594.70 | 600.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 613.50 | 605.65 | 604.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 617.20 | 609.17 | 606.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 639.70 | 641.44 | 633.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 639.70 | 641.44 | 633.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 659.10 | 670.35 | 667.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 659.10 | 670.35 | 667.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 665.80 | 669.44 | 667.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 658.80 | 669.44 | 667.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 665.70 | 668.73 | 667.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:45:00 | 665.50 | 668.73 | 667.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 667.00 | 668.38 | 667.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 666.95 | 668.38 | 667.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 666.00 | 667.91 | 667.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 666.00 | 667.91 | 667.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 666.00 | 667.52 | 667.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 659.75 | 667.52 | 667.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 667.50 | 667.52 | 667.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 662.05 | 667.52 | 667.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 10:15:00 | 660.40 | 666.10 | 666.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 13:15:00 | 659.10 | 662.95 | 665.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 663.30 | 661.74 | 663.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 663.30 | 661.74 | 663.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 663.30 | 661.74 | 663.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 663.30 | 661.74 | 663.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 665.60 | 662.51 | 663.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 664.00 | 662.51 | 663.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 660.50 | 662.11 | 663.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 660.50 | 662.11 | 663.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 659.25 | 661.54 | 663.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:15:00 | 660.15 | 661.54 | 663.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 663.00 | 661.83 | 663.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:00:00 | 663.00 | 661.83 | 663.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 656.55 | 660.77 | 662.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 656.00 | 660.77 | 662.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:30:00 | 653.00 | 655.87 | 657.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:45:00 | 655.90 | 655.48 | 655.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 658.15 | 656.01 | 655.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 658.15 | 656.01 | 655.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 667.00 | 658.17 | 656.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 11:15:00 | 658.45 | 658.54 | 657.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 658.45 | 658.54 | 657.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 658.45 | 658.54 | 657.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:45:00 | 657.65 | 658.54 | 657.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 656.00 | 658.03 | 657.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 656.00 | 658.03 | 657.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 655.60 | 657.54 | 656.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 655.60 | 657.54 | 656.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 660.75 | 658.18 | 657.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 659.00 | 658.18 | 657.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 685.50 | 688.65 | 679.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 685.50 | 688.65 | 679.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 686.70 | 690.33 | 686.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 686.70 | 690.33 | 686.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 678.30 | 687.93 | 686.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 675.50 | 687.93 | 686.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 682.20 | 686.78 | 685.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 683.45 | 685.88 | 685.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 681.05 | 684.37 | 684.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 681.05 | 684.37 | 684.78 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 688.50 | 684.94 | 684.89 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 681.95 | 685.07 | 685.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 676.40 | 681.81 | 683.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 680.40 | 679.41 | 681.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 680.40 | 679.41 | 681.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 680.40 | 679.41 | 681.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 680.40 | 679.41 | 681.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 664.85 | 676.16 | 679.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 659.50 | 676.16 | 679.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:00:00 | 660.50 | 673.03 | 677.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 12:00:00 | 662.10 | 670.84 | 676.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 12:45:00 | 661.70 | 668.98 | 675.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 677.00 | 668.54 | 672.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 673.35 | 668.54 | 672.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 14:15:00 | 679.70 | 675.38 | 674.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 679.70 | 675.38 | 674.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 684.50 | 678.29 | 676.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 15:15:00 | 683.15 | 683.43 | 680.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:15:00 | 690.85 | 683.43 | 680.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 686.45 | 685.38 | 683.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 684.60 | 685.38 | 683.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 680.80 | 684.46 | 683.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 680.80 | 684.46 | 683.13 | SL hit (close<ema400) qty=1.00 sl=683.13 alert=retest1 |

### Cycle 97 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 675.05 | 683.63 | 683.64 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 687.95 | 683.24 | 682.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 689.50 | 684.61 | 683.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 680.10 | 684.18 | 683.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 680.10 | 684.18 | 683.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 680.10 | 684.18 | 683.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 683.20 | 684.18 | 683.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 688.10 | 684.96 | 683.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 680.05 | 684.96 | 683.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 687.95 | 687.25 | 685.40 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 14:15:00 | 682.75 | 684.39 | 684.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 11:15:00 | 675.65 | 682.14 | 683.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 684.15 | 680.03 | 681.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 684.15 | 680.03 | 681.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 684.15 | 680.03 | 681.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 684.50 | 680.03 | 681.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 682.10 | 680.45 | 681.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 684.70 | 680.45 | 681.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 684.65 | 681.29 | 681.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 684.65 | 681.29 | 681.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 687.70 | 682.57 | 682.44 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 678.95 | 683.54 | 683.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 665.20 | 675.44 | 677.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 643.75 | 642.54 | 650.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 12:30:00 | 637.55 | 640.48 | 647.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 643.80 | 635.62 | 642.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 643.80 | 635.62 | 642.58 | SL hit (close>ema400) qty=1.00 sl=642.58 alert=retest1 |

### Cycle 102 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 641.00 | 630.55 | 629.39 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 11:15:00 | 627.70 | 631.88 | 631.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 15:15:00 | 624.80 | 628.90 | 630.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 629.10 | 628.94 | 630.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 629.10 | 628.94 | 630.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 629.10 | 628.94 | 630.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 629.10 | 628.94 | 630.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 628.35 | 628.82 | 630.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 11:30:00 | 627.30 | 628.55 | 629.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 12:45:00 | 627.90 | 628.47 | 629.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 627.45 | 628.85 | 629.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 636.55 | 629.73 | 629.74 | SL hit (close>static) qty=1.00 sl=631.75 alert=retest2 |

### Cycle 104 — BUY (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 13:15:00 | 634.65 | 630.68 | 630.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 15:15:00 | 639.50 | 632.97 | 631.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 14:15:00 | 633.90 | 635.35 | 633.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 14:15:00 | 633.90 | 635.35 | 633.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 633.90 | 635.35 | 633.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 633.90 | 635.35 | 633.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 635.00 | 635.28 | 633.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 638.00 | 635.28 | 633.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 633.25 | 634.87 | 633.70 | SL hit (close<static) qty=1.00 sl=633.35 alert=retest2 |

### Cycle 105 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 629.95 | 632.73 | 632.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 624.45 | 627.75 | 629.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 611.40 | 610.49 | 616.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 611.40 | 610.49 | 616.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 618.75 | 612.67 | 615.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 617.65 | 612.67 | 615.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 618.20 | 613.77 | 616.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 617.95 | 613.77 | 616.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 616.85 | 615.17 | 616.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 616.85 | 615.17 | 616.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 618.00 | 615.73 | 616.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 619.00 | 615.73 | 616.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 619.80 | 616.55 | 616.65 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 622.30 | 617.70 | 617.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 12:15:00 | 624.80 | 620.07 | 619.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 622.85 | 629.43 | 626.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 622.85 | 629.43 | 626.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 622.85 | 629.43 | 626.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 622.85 | 629.43 | 626.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 622.35 | 628.01 | 626.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:45:00 | 626.50 | 628.21 | 626.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:15:00 | 625.60 | 627.57 | 626.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 619.55 | 624.62 | 625.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 619.55 | 624.62 | 625.20 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 630.00 | 625.77 | 625.31 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 621.60 | 625.74 | 626.30 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 628.90 | 626.69 | 626.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 632.05 | 627.76 | 626.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 629.00 | 630.02 | 628.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 629.00 | 630.02 | 628.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 629.00 | 629.82 | 628.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 632.35 | 629.82 | 628.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 10:15:00 | 628.40 | 629.67 | 628.80 | SL hit (close<static) qty=1.00 sl=628.65 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 626.90 | 628.22 | 628.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 624.35 | 627.45 | 627.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 624.70 | 624.53 | 625.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 624.70 | 624.53 | 625.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 624.70 | 624.53 | 625.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 624.70 | 624.53 | 625.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 622.70 | 624.17 | 625.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:30:00 | 622.25 | 623.42 | 625.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 622.05 | 616.93 | 616.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 622.05 | 616.93 | 616.33 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 613.95 | 617.55 | 617.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 610.50 | 615.47 | 616.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 616.35 | 614.44 | 615.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 616.35 | 614.44 | 615.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 616.35 | 614.44 | 615.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 616.35 | 614.44 | 615.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 615.30 | 614.61 | 615.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 612.25 | 614.61 | 615.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 14:15:00 | 625.00 | 616.70 | 616.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 625.00 | 616.70 | 616.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 631.80 | 620.73 | 618.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 14:15:00 | 634.85 | 636.00 | 630.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 14:30:00 | 632.90 | 636.00 | 630.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 630.00 | 634.80 | 630.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 627.80 | 633.22 | 630.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 626.00 | 631.78 | 629.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 625.50 | 631.78 | 629.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 634.05 | 631.79 | 630.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:30:00 | 628.00 | 631.79 | 630.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 632.05 | 631.95 | 630.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:30:00 | 630.70 | 631.95 | 630.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 629.50 | 631.68 | 630.69 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 620.00 | 628.66 | 629.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 617.50 | 625.00 | 627.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 599.85 | 597.61 | 605.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 599.85 | 597.61 | 605.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 605.05 | 599.71 | 604.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 605.05 | 599.71 | 604.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 610.00 | 601.77 | 604.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 611.75 | 601.77 | 604.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 614.70 | 604.36 | 605.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 614.70 | 604.36 | 605.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 616.65 | 606.81 | 606.62 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 598.80 | 607.69 | 607.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 588.50 | 603.85 | 606.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 587.50 | 587.02 | 592.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:00:00 | 587.50 | 587.02 | 592.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 589.55 | 588.28 | 590.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:30:00 | 588.40 | 588.57 | 590.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:00:00 | 587.35 | 588.32 | 589.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 585.00 | 588.84 | 589.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 588.40 | 587.09 | 587.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 583.60 | 586.39 | 587.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:00:00 | 580.15 | 585.14 | 586.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 558.98 | 574.42 | 580.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 558.98 | 574.42 | 580.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 14:15:00 | 572.85 | 571.62 | 576.25 | SL hit (close>ema200) qty=0.50 sl=571.62 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 577.60 | 559.88 | 558.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 579.70 | 573.96 | 568.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 577.50 | 578.32 | 573.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 09:15:00 | 577.15 | 578.32 | 573.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 579.30 | 578.51 | 574.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 584.60 | 578.51 | 574.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:15:00 | 580.40 | 580.52 | 576.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:30:00 | 580.90 | 580.53 | 577.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 581.15 | 580.53 | 577.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 571.25 | 579.40 | 577.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 571.25 | 579.40 | 577.81 | SL hit (close<static) qty=1.00 sl=574.35 alert=retest2 |

### Cycle 119 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 568.95 | 575.51 | 576.21 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 578.25 | 573.71 | 573.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 580.95 | 575.16 | 574.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 575.40 | 578.70 | 576.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 575.40 | 578.70 | 576.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 575.40 | 578.70 | 576.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 575.40 | 578.70 | 576.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 573.55 | 577.67 | 576.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 572.60 | 577.67 | 576.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 13:15:00 | 572.50 | 575.03 | 575.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 14:15:00 | 570.50 | 574.13 | 574.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 566.40 | 565.94 | 569.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 566.40 | 565.94 | 569.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 561.60 | 562.35 | 565.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 562.85 | 562.35 | 565.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 551.70 | 547.45 | 549.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 551.70 | 547.45 | 549.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 549.95 | 547.95 | 549.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 553.25 | 547.95 | 549.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 550.50 | 548.46 | 549.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 550.50 | 548.46 | 549.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 549.80 | 548.73 | 549.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 549.80 | 548.73 | 549.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 550.10 | 549.00 | 549.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 554.45 | 549.00 | 549.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 557.90 | 550.78 | 550.62 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 471.60 | 535.71 | 544.17 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 13:15:00 | 517.30 | 506.91 | 505.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 524.90 | 516.03 | 512.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 15:15:00 | 568.60 | 568.70 | 562.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 09:15:00 | 568.60 | 568.70 | 562.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 565.45 | 567.99 | 564.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 565.45 | 567.99 | 564.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 570.65 | 567.91 | 565.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:00:00 | 572.90 | 568.90 | 566.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:30:00 | 573.00 | 570.17 | 567.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:30:00 | 573.60 | 570.21 | 568.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 10:45:00 | 575.75 | 575.79 | 573.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 573.30 | 575.29 | 573.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:30:00 | 574.80 | 575.29 | 573.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 572.70 | 574.77 | 573.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 568.45 | 572.27 | 572.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 568.45 | 572.27 | 572.65 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 574.25 | 572.47 | 572.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 577.50 | 574.00 | 573.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 573.75 | 574.09 | 573.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 573.75 | 574.09 | 573.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 573.75 | 574.09 | 573.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 573.75 | 574.09 | 573.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 576.95 | 574.66 | 573.70 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 570.80 | 573.13 | 573.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 568.50 | 571.86 | 572.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 563.90 | 562.59 | 565.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 563.90 | 562.59 | 565.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 560.80 | 562.30 | 565.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 564.50 | 562.30 | 565.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 567.00 | 563.24 | 565.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:45:00 | 568.20 | 563.24 | 565.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 560.55 | 562.70 | 564.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:30:00 | 565.85 | 562.70 | 564.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 548.95 | 551.14 | 555.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:45:00 | 545.65 | 549.93 | 554.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 553.00 | 550.11 | 549.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 553.00 | 550.11 | 549.85 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 547.65 | 549.61 | 549.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 544.10 | 548.51 | 549.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 11:15:00 | 538.50 | 537.74 | 541.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 12:00:00 | 538.50 | 537.74 | 541.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 538.20 | 538.33 | 540.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:30:00 | 540.50 | 538.33 | 540.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 540.20 | 538.82 | 540.55 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 547.60 | 541.73 | 541.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 548.95 | 544.24 | 542.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 546.60 | 546.84 | 545.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 546.60 | 546.84 | 545.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 540.00 | 545.88 | 545.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 539.50 | 545.88 | 545.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 527.95 | 542.30 | 543.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 527.10 | 534.03 | 535.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 513.95 | 505.30 | 511.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 513.95 | 505.30 | 511.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 513.95 | 505.30 | 511.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 513.95 | 505.30 | 511.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 509.10 | 506.06 | 511.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 507.75 | 506.06 | 511.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:45:00 | 507.65 | 506.64 | 511.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 516.20 | 510.67 | 512.17 | SL hit (close>static) qty=1.00 sl=515.05 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 522.00 | 514.81 | 513.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 536.95 | 521.62 | 517.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 534.60 | 535.24 | 528.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 534.60 | 535.24 | 528.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 532.00 | 533.89 | 530.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 530.35 | 533.89 | 530.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 535.20 | 535.23 | 532.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 544.65 | 535.23 | 532.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 10:45:00 | 536.40 | 535.84 | 533.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:15:00 | 535.95 | 536.08 | 534.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 531.85 | 535.23 | 534.08 | SL hit (close<static) qty=1.00 sl=532.80 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 529.95 | 532.98 | 533.29 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 538.15 | 534.10 | 533.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 538.90 | 535.94 | 534.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 15:15:00 | 550.00 | 550.59 | 545.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-27 09:15:00 | 544.55 | 550.59 | 545.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 547.40 | 549.96 | 545.82 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 14:15:00 | 535.55 | 543.20 | 543.73 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 15:15:00 | 547.00 | 543.09 | 542.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 548.80 | 544.57 | 543.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 11:15:00 | 537.35 | 543.12 | 542.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 11:15:00 | 537.35 | 543.12 | 542.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 537.35 | 543.12 | 542.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 537.35 | 543.12 | 542.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 535.80 | 541.66 | 542.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-29 13:15:00 | 525.00 | 538.33 | 540.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 512.95 | 511.57 | 520.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:45:00 | 514.70 | 511.57 | 520.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 513.00 | 513.27 | 516.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:45:00 | 515.80 | 513.27 | 516.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 506.10 | 497.87 | 502.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 506.10 | 497.87 | 502.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 506.75 | 499.65 | 503.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 506.75 | 499.65 | 503.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 511.40 | 505.15 | 504.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 513.45 | 506.81 | 505.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 517.85 | 518.16 | 513.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 517.85 | 518.16 | 513.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 514.90 | 516.94 | 514.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 514.90 | 516.94 | 514.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 512.45 | 516.04 | 514.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 513.00 | 516.04 | 514.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 513.90 | 515.61 | 514.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 521.35 | 514.63 | 513.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 516.15 | 515.26 | 514.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 511.50 | 514.51 | 514.12 | SL hit (close<static) qty=1.00 sl=512.30 alert=retest2 |

### Cycle 139 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 510.05 | 513.16 | 513.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 506.55 | 511.43 | 512.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 10:15:00 | 509.05 | 508.15 | 510.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 10:15:00 | 509.05 | 508.15 | 510.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 509.05 | 508.15 | 510.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:00:00 | 509.05 | 508.15 | 510.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 503.35 | 498.95 | 502.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 503.35 | 498.95 | 502.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 498.40 | 498.84 | 502.45 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 507.15 | 503.76 | 503.69 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 495.85 | 502.21 | 503.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 492.90 | 500.35 | 502.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 488.15 | 486.82 | 491.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 488.15 | 486.82 | 491.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 476.60 | 485.33 | 489.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 476.25 | 485.33 | 489.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 475.00 | 479.94 | 481.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 11:45:00 | 475.70 | 477.48 | 479.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 470.80 | 467.57 | 467.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 470.80 | 467.57 | 467.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 474.95 | 469.05 | 468.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 493.75 | 496.06 | 489.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 493.75 | 496.06 | 489.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 504.60 | 499.14 | 493.85 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 487.55 | 493.26 | 493.33 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 495.20 | 493.51 | 493.36 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 487.60 | 492.57 | 493.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 485.65 | 488.73 | 490.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 488.05 | 487.90 | 489.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 488.05 | 487.90 | 489.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 488.05 | 487.90 | 489.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 490.00 | 487.90 | 489.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 490.60 | 488.44 | 489.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:00:00 | 490.60 | 488.44 | 489.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 491.95 | 489.14 | 489.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 491.95 | 489.14 | 489.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 492.05 | 489.73 | 489.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:30:00 | 492.50 | 489.73 | 489.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 490.70 | 490.06 | 490.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 490.90 | 490.06 | 490.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 491.00 | 490.25 | 490.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 497.00 | 491.60 | 490.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 504.30 | 504.81 | 501.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:00:00 | 504.30 | 504.81 | 501.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 519.00 | 522.46 | 518.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 528.55 | 522.46 | 518.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 529.60 | 534.79 | 535.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 529.60 | 534.79 | 535.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 526.20 | 532.31 | 533.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 533.20 | 532.47 | 533.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 533.20 | 532.47 | 533.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 533.20 | 532.47 | 533.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 533.20 | 532.47 | 533.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 533.80 | 532.74 | 533.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:45:00 | 532.90 | 532.74 | 533.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:30:00 | 532.90 | 533.38 | 533.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 540.90 | 535.20 | 534.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 540.90 | 535.20 | 534.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 542.80 | 536.72 | 535.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 530.80 | 537.98 | 536.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 530.80 | 537.98 | 536.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 530.80 | 537.98 | 536.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 530.80 | 537.98 | 536.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 534.80 | 537.34 | 536.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 535.90 | 537.34 | 536.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 532.20 | 535.84 | 536.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 532.20 | 535.84 | 536.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 528.85 | 534.45 | 535.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 10:15:00 | 529.55 | 529.06 | 532.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 11:00:00 | 529.55 | 529.06 | 532.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 519.05 | 527.06 | 531.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 12:15:00 | 517.90 | 527.06 | 531.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 532.05 | 527.08 | 529.27 | SL hit (close>static) qty=1.00 sl=531.90 alert=retest2 |

### Cycle 150 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 536.95 | 531.44 | 530.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 15:15:00 | 537.80 | 534.30 | 532.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 533.50 | 534.14 | 532.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 533.50 | 534.14 | 532.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 533.50 | 534.14 | 532.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:30:00 | 535.75 | 534.14 | 532.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 534.85 | 534.28 | 532.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:30:00 | 534.55 | 534.28 | 532.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 570.45 | 577.65 | 574.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 570.45 | 577.65 | 574.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 571.65 | 576.45 | 574.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:45:00 | 572.60 | 575.76 | 574.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 572.05 | 575.19 | 574.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 570.60 | 573.52 | 573.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 570.60 | 573.52 | 573.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 556.10 | 569.50 | 571.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 539.50 | 534.49 | 538.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 539.50 | 534.49 | 538.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 539.50 | 534.49 | 538.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 538.95 | 534.49 | 538.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 545.40 | 536.67 | 538.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 545.40 | 536.67 | 538.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 542.60 | 540.62 | 540.39 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 534.05 | 539.25 | 539.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 532.80 | 536.76 | 538.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 536.80 | 535.24 | 537.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 536.80 | 535.24 | 537.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 536.80 | 535.24 | 537.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 536.80 | 535.24 | 537.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 537.70 | 535.73 | 537.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 538.15 | 535.73 | 537.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 535.10 | 535.60 | 536.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:45:00 | 533.25 | 535.13 | 536.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 531.20 | 535.05 | 536.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 542.30 | 529.28 | 530.06 | SL hit (close>static) qty=1.00 sl=538.25 alert=retest2 |

### Cycle 154 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 541.45 | 531.72 | 531.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 543.90 | 541.09 | 538.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 541.00 | 541.61 | 539.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 541.00 | 541.61 | 539.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 541.00 | 541.61 | 539.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 541.85 | 541.61 | 539.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 562.60 | 564.98 | 561.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 562.60 | 564.98 | 561.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 563.75 | 564.73 | 561.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 565.50 | 565.21 | 562.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 567.50 | 567.88 | 567.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 567.50 | 567.88 | 567.92 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 15:15:00 | 570.00 | 568.30 | 568.11 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 560.60 | 566.76 | 567.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 10:15:00 | 559.75 | 565.36 | 566.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 563.95 | 563.10 | 564.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 563.95 | 563.10 | 564.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 563.95 | 563.10 | 564.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 562.70 | 563.10 | 564.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 563.70 | 563.22 | 564.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:45:00 | 565.65 | 563.22 | 564.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 560.95 | 562.77 | 564.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 13:00:00 | 560.20 | 562.25 | 563.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 559.90 | 561.90 | 563.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 560.00 | 560.80 | 562.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 560.15 | 560.77 | 561.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 557.55 | 555.76 | 557.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 559.10 | 555.76 | 557.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 556.20 | 555.85 | 557.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 14:30:00 | 554.80 | 555.63 | 557.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:00:00 | 554.75 | 555.63 | 557.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 555.75 | 553.17 | 552.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 555.75 | 553.17 | 552.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 557.65 | 554.98 | 553.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 561.05 | 562.71 | 560.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 561.05 | 562.71 | 560.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 561.05 | 562.71 | 560.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 560.70 | 562.71 | 560.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 560.95 | 562.35 | 560.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 560.95 | 562.35 | 560.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 562.00 | 562.28 | 560.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 559.50 | 561.18 | 560.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 555.30 | 560.00 | 560.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 549.05 | 557.13 | 558.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 544.95 | 544.32 | 548.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 544.95 | 544.32 | 548.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 548.35 | 545.13 | 548.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 548.35 | 545.13 | 548.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 552.25 | 546.55 | 548.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 552.25 | 546.55 | 548.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 550.30 | 547.30 | 548.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 549.70 | 547.95 | 549.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 554.35 | 549.79 | 549.80 | SL hit (close>static) qty=1.00 sl=552.75 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 554.40 | 550.71 | 550.21 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 547.15 | 549.85 | 550.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 545.05 | 548.18 | 549.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 533.90 | 533.87 | 538.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 533.90 | 533.87 | 538.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 536.85 | 534.39 | 537.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 536.85 | 534.39 | 537.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 536.30 | 534.77 | 537.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 537.50 | 534.77 | 537.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 536.90 | 535.47 | 537.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 536.90 | 535.47 | 537.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 535.65 | 535.50 | 536.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 536.35 | 535.50 | 536.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 546.20 | 537.54 | 537.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 546.20 | 537.54 | 537.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 556.90 | 541.41 | 539.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 560.25 | 545.18 | 541.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 11:15:00 | 590.00 | 590.94 | 585.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:45:00 | 589.30 | 590.94 | 585.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 589.35 | 589.68 | 586.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:45:00 | 586.65 | 589.68 | 586.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 587.60 | 592.11 | 590.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 587.60 | 592.11 | 590.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 588.05 | 591.30 | 589.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:45:00 | 585.70 | 591.30 | 589.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 592.00 | 590.86 | 590.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:30:00 | 590.00 | 590.86 | 590.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 590.00 | 590.68 | 590.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 590.95 | 590.68 | 590.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 585.25 | 589.60 | 589.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 581.55 | 587.99 | 588.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 588.00 | 587.99 | 588.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 588.00 | 587.99 | 588.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 588.00 | 587.99 | 588.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 588.00 | 587.99 | 588.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 589.15 | 588.22 | 588.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 589.15 | 588.22 | 588.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 590.00 | 588.58 | 588.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 590.00 | 588.58 | 588.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 591.60 | 589.18 | 589.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 591.60 | 589.18 | 589.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 591.55 | 589.66 | 589.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 593.10 | 590.34 | 589.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 590.80 | 591.42 | 590.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:00:00 | 590.80 | 591.42 | 590.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 591.10 | 591.36 | 590.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 591.10 | 591.36 | 590.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 590.75 | 591.24 | 590.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 590.25 | 591.24 | 590.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 590.00 | 590.99 | 590.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 588.10 | 590.99 | 590.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 584.35 | 589.66 | 589.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 582.20 | 586.63 | 588.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 590.15 | 586.66 | 587.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 590.15 | 586.66 | 587.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 590.15 | 586.66 | 587.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 590.65 | 586.66 | 587.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 587.45 | 586.82 | 587.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 584.15 | 586.88 | 587.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 585.05 | 586.75 | 587.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 589.95 | 587.86 | 587.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 589.95 | 587.86 | 587.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 592.40 | 589.46 | 588.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 589.95 | 590.87 | 589.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 589.95 | 590.87 | 589.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 589.95 | 590.87 | 589.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 590.20 | 590.87 | 589.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 586.85 | 590.06 | 589.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 586.85 | 590.06 | 589.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 590.80 | 590.21 | 589.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 592.50 | 590.44 | 589.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 592.20 | 591.87 | 591.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:30:00 | 592.35 | 592.11 | 591.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 613.85 | 615.97 | 616.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 613.85 | 615.97 | 616.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 609.70 | 614.41 | 615.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 618.00 | 615.13 | 615.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 618.00 | 615.13 | 615.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 618.00 | 615.13 | 615.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 618.00 | 615.13 | 615.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 617.45 | 615.59 | 615.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 615.40 | 615.55 | 615.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 617.75 | 612.92 | 612.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 617.75 | 612.92 | 612.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 623.40 | 617.31 | 615.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 13:15:00 | 608.55 | 615.96 | 615.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 13:15:00 | 608.55 | 615.96 | 615.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 608.55 | 615.96 | 615.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 608.55 | 615.96 | 615.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 592.35 | 611.24 | 613.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 590.00 | 606.99 | 610.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 12:15:00 | 606.25 | 605.27 | 608.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 13:00:00 | 606.25 | 605.27 | 608.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 607.05 | 605.22 | 608.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 607.05 | 605.22 | 608.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 607.90 | 605.75 | 608.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:15:00 | 604.65 | 606.89 | 607.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 14:00:00 | 605.00 | 606.51 | 607.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 15:00:00 | 605.20 | 606.25 | 607.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:15:00 | 574.42 | 583.12 | 585.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:15:00 | 574.75 | 583.12 | 585.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:15:00 | 574.94 | 583.12 | 585.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 578.75 | 576.96 | 580.17 | SL hit (close>ema200) qty=0.50 sl=576.96 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 600.75 | 581.88 | 581.86 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 587.25 | 589.39 | 589.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 585.35 | 587.92 | 588.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 580.90 | 580.81 | 583.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 13:00:00 | 580.90 | 580.81 | 583.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 566.60 | 565.28 | 568.20 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 573.55 | 568.80 | 568.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 576.10 | 572.27 | 570.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 570.90 | 572.46 | 570.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 570.90 | 572.46 | 570.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 570.90 | 572.46 | 570.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 570.90 | 572.46 | 570.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 574.10 | 572.79 | 571.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 570.50 | 572.79 | 571.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 571.80 | 572.59 | 571.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 571.00 | 572.59 | 571.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 571.65 | 572.40 | 571.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 569.45 | 572.40 | 571.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 569.65 | 571.85 | 571.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 569.65 | 571.85 | 571.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 567.25 | 570.93 | 570.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 567.25 | 570.93 | 570.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 569.10 | 570.57 | 570.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 563.50 | 568.81 | 569.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 567.95 | 567.76 | 568.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 567.95 | 567.76 | 568.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 570.15 | 568.02 | 568.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 570.15 | 568.02 | 568.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 569.65 | 568.34 | 568.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 568.95 | 568.34 | 568.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 573.50 | 569.38 | 569.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 573.50 | 569.38 | 569.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 576.70 | 570.84 | 569.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 570.25 | 571.34 | 570.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 570.25 | 571.34 | 570.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 570.25 | 571.34 | 570.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 570.25 | 571.34 | 570.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 572.10 | 571.49 | 570.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 573.80 | 571.49 | 570.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 568.90 | 571.07 | 570.58 | SL hit (close<static) qty=1.00 sl=569.65 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 567.35 | 569.92 | 570.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 15:15:00 | 566.70 | 567.83 | 568.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 561.00 | 560.96 | 564.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 561.00 | 560.96 | 564.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 566.40 | 561.93 | 564.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 566.40 | 561.93 | 564.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 566.60 | 562.86 | 564.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 566.50 | 562.86 | 564.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 564.70 | 563.09 | 564.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 566.05 | 563.09 | 564.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 561.50 | 562.77 | 564.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:30:00 | 560.40 | 562.30 | 563.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 15:00:00 | 560.40 | 562.30 | 563.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 568.20 | 563.27 | 563.90 | SL hit (close>static) qty=1.00 sl=565.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 568.60 | 564.34 | 564.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 569.60 | 566.06 | 565.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 578.75 | 579.82 | 576.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 578.75 | 579.82 | 576.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 577.25 | 579.30 | 576.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 576.80 | 579.30 | 576.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 581.70 | 579.78 | 577.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 584.00 | 579.83 | 577.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 584.20 | 580.78 | 578.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 583.20 | 581.27 | 579.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 582.80 | 581.27 | 579.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 588.00 | 589.93 | 586.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 580.90 | 589.93 | 586.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 586.75 | 589.29 | 586.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 589.60 | 588.93 | 586.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:45:00 | 591.80 | 590.66 | 588.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 582.30 | 588.54 | 588.15 | SL hit (close<static) qty=1.00 sl=585.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 583.00 | 587.43 | 587.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 581.15 | 584.48 | 586.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 11:15:00 | 566.30 | 565.94 | 569.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 12:00:00 | 566.30 | 565.94 | 569.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 569.70 | 567.07 | 569.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 570.45 | 567.07 | 569.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 571.50 | 567.96 | 569.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 568.05 | 568.44 | 569.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 573.15 | 569.74 | 569.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 573.15 | 569.74 | 569.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 577.90 | 571.38 | 570.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 573.80 | 574.12 | 572.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:45:00 | 575.35 | 574.12 | 572.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 573.40 | 573.98 | 572.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 572.90 | 573.98 | 572.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 572.25 | 573.63 | 572.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 572.55 | 573.63 | 572.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 572.75 | 573.46 | 572.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 572.80 | 573.46 | 572.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 571.95 | 573.15 | 572.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:00:00 | 571.95 | 573.15 | 572.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 574.05 | 573.33 | 572.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 575.55 | 573.37 | 572.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 569.70 | 572.39 | 572.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 569.70 | 572.39 | 572.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 566.15 | 570.09 | 571.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 567.85 | 566.21 | 568.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 567.85 | 566.21 | 568.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 567.85 | 566.21 | 568.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 566.90 | 566.21 | 568.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 566.20 | 566.21 | 567.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 567.30 | 566.21 | 567.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 571.20 | 566.82 | 567.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 570.85 | 566.82 | 567.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 570.70 | 567.59 | 567.63 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 570.60 | 568.19 | 567.90 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 566.00 | 567.94 | 568.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 563.20 | 566.41 | 567.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 563.75 | 562.69 | 564.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 563.75 | 562.69 | 564.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 563.75 | 562.69 | 564.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 564.30 | 562.69 | 564.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 565.80 | 563.35 | 564.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 565.10 | 563.35 | 564.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 565.85 | 563.85 | 564.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 565.85 | 563.85 | 564.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 566.05 | 564.39 | 564.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 566.20 | 564.39 | 564.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 565.60 | 564.64 | 564.91 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 567.50 | 565.21 | 565.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 571.35 | 567.09 | 566.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 565.20 | 568.38 | 567.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 565.20 | 568.38 | 567.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 565.20 | 568.38 | 567.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 564.90 | 568.38 | 567.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 567.50 | 568.21 | 567.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 564.45 | 568.21 | 567.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 565.90 | 567.75 | 567.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:45:00 | 565.60 | 567.75 | 567.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 563.50 | 566.90 | 566.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 561.40 | 564.65 | 565.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 565.20 | 564.76 | 565.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 565.20 | 564.76 | 565.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 565.20 | 564.76 | 565.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 566.75 | 564.76 | 565.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 566.90 | 565.19 | 565.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:00:00 | 566.90 | 565.19 | 565.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 567.60 | 565.67 | 565.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:45:00 | 567.30 | 565.67 | 565.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 567.05 | 566.07 | 566.12 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 568.65 | 566.59 | 566.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 569.50 | 567.24 | 566.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 568.00 | 568.16 | 567.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:45:00 | 568.05 | 568.16 | 567.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 563.15 | 567.16 | 567.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 563.15 | 567.16 | 567.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 564.05 | 566.54 | 566.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 563.00 | 565.83 | 566.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 558.25 | 557.21 | 560.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 13:45:00 | 558.10 | 557.21 | 560.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 560.15 | 557.80 | 560.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:30:00 | 559.45 | 557.80 | 560.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 561.60 | 558.56 | 560.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 562.45 | 558.56 | 560.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 559.10 | 558.66 | 560.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 559.00 | 558.36 | 559.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 565.40 | 559.08 | 559.16 | SL hit (close>static) qty=1.00 sl=563.25 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 562.20 | 559.71 | 559.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 570.30 | 564.04 | 561.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 567.25 | 567.29 | 564.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:00:00 | 567.25 | 567.29 | 564.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 566.00 | 566.81 | 565.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 566.00 | 566.81 | 565.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 564.05 | 566.26 | 565.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 564.20 | 566.26 | 565.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 564.95 | 566.00 | 565.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 570.90 | 566.66 | 565.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 569.45 | 566.20 | 565.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 561.80 | 568.41 | 569.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 561.80 | 568.41 | 569.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 560.75 | 564.96 | 567.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 560.80 | 560.79 | 564.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 560.80 | 560.79 | 564.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 562.05 | 560.75 | 562.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 561.75 | 560.75 | 562.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 560.30 | 560.75 | 562.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 559.90 | 560.75 | 562.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 558.50 | 557.56 | 559.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 566.10 | 559.41 | 559.69 | SL hit (close>static) qty=1.00 sl=563.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 563.45 | 560.22 | 560.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 568.35 | 562.26 | 561.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 561.05 | 563.29 | 562.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 561.05 | 563.29 | 562.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 561.05 | 563.29 | 562.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 561.05 | 563.29 | 562.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 559.90 | 562.61 | 562.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 559.90 | 562.61 | 562.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 558.60 | 561.81 | 561.83 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 562.60 | 561.90 | 561.87 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 560.95 | 562.05 | 562.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 12:15:00 | 559.70 | 561.58 | 561.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 13:15:00 | 560.85 | 560.21 | 560.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 13:15:00 | 560.85 | 560.21 | 560.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 560.85 | 560.21 | 560.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 560.85 | 560.21 | 560.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 557.50 | 559.67 | 560.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 556.10 | 558.97 | 560.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:45:00 | 556.95 | 556.99 | 558.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 557.15 | 557.38 | 558.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 551.55 | 548.78 | 548.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 551.55 | 548.78 | 548.68 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 547.80 | 548.71 | 548.79 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 551.20 | 549.26 | 549.03 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 546.50 | 549.11 | 549.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 545.70 | 548.42 | 548.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 545.90 | 545.30 | 546.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 11:00:00 | 545.90 | 545.30 | 546.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 539.15 | 543.24 | 545.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 535.75 | 541.74 | 544.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 536.65 | 538.87 | 540.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 536.25 | 535.14 | 536.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:45:00 | 537.95 | 535.40 | 536.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 532.70 | 534.86 | 536.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:45:00 | 532.30 | 534.40 | 536.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 13:15:00 | 530.60 | 534.40 | 536.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:30:00 | 532.15 | 531.67 | 533.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 532.25 | 531.67 | 533.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 532.25 | 531.79 | 532.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 534.50 | 531.79 | 532.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 533.85 | 532.20 | 533.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 532.20 | 532.44 | 533.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 532.05 | 532.60 | 533.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 532.10 | 532.50 | 532.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 11:45:00 | 531.25 | 530.92 | 531.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 533.40 | 531.42 | 531.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 534.70 | 531.42 | 531.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 534.90 | 532.11 | 532.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 535.25 | 532.11 | 532.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 535.25 | 532.74 | 532.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 535.25 | 532.74 | 532.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 546.55 | 536.02 | 534.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 547.45 | 550.87 | 546.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 547.45 | 550.87 | 546.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 547.40 | 550.18 | 546.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 546.80 | 550.18 | 546.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 546.50 | 549.44 | 546.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 546.10 | 549.44 | 546.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 545.95 | 548.74 | 546.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 545.80 | 548.74 | 546.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 547.95 | 548.58 | 546.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:30:00 | 546.80 | 548.58 | 546.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 546.45 | 548.19 | 546.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 546.45 | 548.19 | 546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 545.50 | 547.65 | 546.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 545.90 | 547.65 | 546.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 544.05 | 546.93 | 546.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 544.05 | 546.93 | 546.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 542.50 | 546.04 | 546.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 541.10 | 545.06 | 545.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 537.15 | 537.14 | 539.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 537.15 | 537.14 | 539.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 539.65 | 537.64 | 539.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 540.15 | 537.64 | 539.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 538.75 | 537.86 | 539.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 544.50 | 537.86 | 539.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 541.25 | 538.54 | 539.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 543.70 | 538.54 | 539.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 539.40 | 539.74 | 539.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:30:00 | 539.90 | 539.74 | 539.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 540.15 | 539.82 | 539.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 540.15 | 539.82 | 539.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 540.20 | 539.90 | 540.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 551.35 | 539.90 | 540.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 549.90 | 541.90 | 540.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 565.20 | 561.30 | 558.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 566.40 | 568.46 | 565.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 566.40 | 568.46 | 565.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 564.50 | 567.67 | 565.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 563.85 | 567.67 | 565.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 565.15 | 567.16 | 565.09 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 562.60 | 564.14 | 564.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 561.50 | 563.14 | 563.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 537.35 | 536.03 | 542.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 538.65 | 536.03 | 542.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 539.35 | 537.34 | 541.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 535.35 | 537.78 | 540.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 536.05 | 537.44 | 539.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 545.25 | 539.29 | 540.37 | SL hit (close>static) qty=1.00 sl=542.35 alert=retest2 |

### Cycle 200 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 549.60 | 542.58 | 541.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 557.00 | 548.04 | 544.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 14:15:00 | 552.40 | 552.85 | 550.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:30:00 | 552.35 | 552.85 | 550.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 551.30 | 552.54 | 550.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 552.75 | 552.54 | 550.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:00:00 | 552.85 | 552.60 | 550.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 548.30 | 551.74 | 550.62 | SL hit (close<static) qty=1.00 sl=549.65 alert=retest2 |

### Cycle 201 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 545.00 | 549.66 | 549.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 538.10 | 547.35 | 548.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 541.70 | 540.96 | 544.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 541.70 | 540.96 | 544.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 541.70 | 540.96 | 544.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 543.65 | 540.96 | 544.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 545.45 | 540.56 | 542.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 544.70 | 540.56 | 542.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 542.40 | 540.93 | 542.78 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 546.70 | 543.71 | 543.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 547.00 | 544.37 | 543.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 542.55 | 544.01 | 543.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 542.55 | 544.01 | 543.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 542.55 | 544.01 | 543.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 543.80 | 544.01 | 543.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 543.35 | 543.88 | 543.79 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 540.00 | 543.10 | 543.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 527.75 | 540.03 | 542.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 535.75 | 530.87 | 536.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 10:00:00 | 535.75 | 530.87 | 536.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 529.55 | 530.61 | 535.56 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 536.45 | 533.75 | 533.62 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 532.35 | 533.41 | 533.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 529.80 | 532.69 | 533.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 507.25 | 507.19 | 514.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 507.25 | 507.19 | 514.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 522.45 | 511.23 | 513.97 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 533.30 | 518.47 | 516.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 12:15:00 | 536.50 | 530.38 | 525.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 532.55 | 533.15 | 528.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 532.55 | 533.15 | 528.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 530.85 | 531.98 | 528.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:15:00 | 530.45 | 531.98 | 528.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 531.05 | 531.79 | 529.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 532.70 | 532.02 | 529.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 525.40 | 530.95 | 529.45 | SL hit (close<static) qty=1.00 sl=528.20 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 525.40 | 528.49 | 528.58 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 533.25 | 529.51 | 529.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 537.80 | 531.16 | 529.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 537.75 | 539.66 | 536.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 13:15:00 | 537.75 | 539.66 | 536.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 537.75 | 539.66 | 536.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 537.75 | 539.66 | 536.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 535.70 | 538.52 | 536.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 535.70 | 538.52 | 536.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 538.00 | 538.42 | 536.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 539.35 | 538.23 | 537.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 539.75 | 538.23 | 537.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 535.35 | 538.27 | 537.73 | SL hit (close<static) qty=1.00 sl=535.55 alert=retest2 |

### Cycle 209 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 533.40 | 537.29 | 537.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 532.60 | 535.76 | 536.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 522.50 | 521.98 | 526.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 522.50 | 521.98 | 526.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 524.45 | 523.09 | 525.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 524.45 | 523.09 | 525.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 527.95 | 524.12 | 525.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 527.95 | 524.12 | 525.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 525.55 | 524.41 | 525.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 526.25 | 524.41 | 525.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 521.15 | 523.75 | 525.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 520.35 | 521.96 | 523.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 494.33 | 500.79 | 505.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 491.90 | 491.86 | 498.11 | SL hit (close>ema200) qty=0.50 sl=491.86 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 442.10 | 435.99 | 435.80 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 424.95 | 434.65 | 435.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 424.55 | 432.63 | 434.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 425.80 | 425.67 | 429.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:00:00 | 425.80 | 425.67 | 429.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 410.50 | 403.35 | 410.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 410.50 | 403.35 | 410.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 411.70 | 405.02 | 410.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 411.70 | 405.02 | 410.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 406.75 | 405.36 | 409.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 411.25 | 405.36 | 409.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 410.40 | 406.37 | 410.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 426.35 | 406.37 | 410.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 424.60 | 410.02 | 411.33 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 424.25 | 412.86 | 412.51 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 411.85 | 414.29 | 414.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 408.85 | 413.21 | 414.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 415.95 | 407.39 | 409.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 415.95 | 407.39 | 409.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 415.95 | 407.39 | 409.56 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 421.50 | 412.77 | 411.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 422.00 | 416.62 | 413.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 408.85 | 415.07 | 413.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 408.85 | 415.07 | 413.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 408.85 | 415.07 | 413.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 408.85 | 415.07 | 413.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 408.60 | 413.77 | 412.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 410.95 | 413.14 | 412.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 411.45 | 412.98 | 412.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 10:15:00 | 452.05 | 431.79 | 425.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 450.15 | 456.67 | 457.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 445.50 | 451.79 | 454.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 449.40 | 447.63 | 451.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 449.40 | 447.63 | 451.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 450.00 | 448.10 | 451.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 456.05 | 448.10 | 451.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 451.40 | 448.76 | 451.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 458.10 | 448.76 | 451.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 456.80 | 450.37 | 451.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 459.00 | 450.37 | 451.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 458.15 | 453.37 | 452.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 467.20 | 456.14 | 454.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 459.25 | 459.64 | 456.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 459.25 | 459.64 | 456.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 457.25 | 458.94 | 457.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 457.85 | 458.94 | 457.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 456.75 | 458.50 | 457.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 456.70 | 458.50 | 457.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 458.85 | 458.57 | 457.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 462.60 | 458.76 | 457.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 462.50 | 459.25 | 457.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:00:00 | 462.60 | 459.92 | 458.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 454.35 | 458.63 | 458.01 | SL hit (close<static) qty=1.00 sl=456.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 455.00 | 457.23 | 457.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 442.80 | 454.34 | 456.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 449.65 | 446.68 | 449.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 10:15:00 | 449.65 | 446.68 | 449.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 449.65 | 446.68 | 449.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 449.65 | 446.68 | 449.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 452.35 | 447.81 | 450.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 454.90 | 447.81 | 450.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 448.85 | 448.02 | 450.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 446.65 | 448.02 | 450.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:15:00 | 441.15 | 448.45 | 450.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 447.65 | 443.11 | 442.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 447.65 | 443.11 | 442.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 456.45 | 449.83 | 446.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 449.05 | 449.93 | 447.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:15:00 | 447.75 | 449.93 | 447.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 446.65 | 449.28 | 447.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 446.65 | 449.28 | 447.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 446.00 | 448.62 | 447.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 445.65 | 448.62 | 447.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 444.40 | 447.78 | 446.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 444.40 | 447.78 | 446.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 443.90 | 447.00 | 446.69 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 13:30:00 | 402.65 | 2023-05-22 09:15:00 | 415.50 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2023-05-19 13:45:00 | 401.00 | 2023-05-22 09:15:00 | 415.50 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2023-05-19 15:00:00 | 402.50 | 2023-05-22 09:15:00 | 415.50 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest1 | 2023-06-06 09:15:00 | 451.05 | 2023-06-08 11:15:00 | 449.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2023-06-09 09:15:00 | 453.20 | 2023-06-16 09:15:00 | 457.90 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2023-06-22 10:45:00 | 445.30 | 2023-06-26 09:15:00 | 423.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-22 10:45:00 | 445.30 | 2023-06-26 14:15:00 | 432.25 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2023-07-06 09:30:00 | 431.00 | 2023-07-06 11:15:00 | 434.30 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-07-13 10:30:00 | 418.60 | 2023-07-19 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-07-13 13:45:00 | 418.00 | 2023-07-19 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-07-17 09:45:00 | 418.40 | 2023-07-19 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-07-18 11:15:00 | 417.95 | 2023-07-19 11:15:00 | 420.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-08-07 14:45:00 | 473.75 | 2023-08-08 09:15:00 | 467.45 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-08-24 09:15:00 | 457.55 | 2023-08-24 11:15:00 | 452.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2023-08-29 13:30:00 | 440.90 | 2023-08-30 12:15:00 | 444.80 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-08-29 14:45:00 | 441.10 | 2023-08-30 12:15:00 | 444.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-08-30 09:45:00 | 441.00 | 2023-08-30 12:15:00 | 444.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-09-08 14:45:00 | 439.05 | 2023-09-18 09:15:00 | 441.35 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2023-09-11 09:15:00 | 441.10 | 2023-09-18 09:15:00 | 441.35 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2023-09-29 14:45:00 | 426.05 | 2023-10-03 10:15:00 | 430.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-10-04 14:15:00 | 428.50 | 2023-10-09 14:15:00 | 430.45 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2023-10-20 10:15:00 | 434.90 | 2023-10-23 15:15:00 | 413.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 10:15:00 | 434.90 | 2023-10-26 14:15:00 | 414.80 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2023-11-13 10:45:00 | 419.85 | 2023-11-15 09:15:00 | 421.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-11-13 12:00:00 | 419.70 | 2023-11-15 09:15:00 | 421.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-11-13 12:45:00 | 419.25 | 2023-11-15 09:15:00 | 421.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-12-13 09:15:00 | 505.35 | 2023-12-20 10:15:00 | 516.90 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2023-12-22 12:00:00 | 503.15 | 2023-12-26 09:15:00 | 507.25 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-12-22 15:15:00 | 503.10 | 2023-12-26 09:15:00 | 507.25 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-12-29 11:15:00 | 520.80 | 2024-01-08 10:15:00 | 533.80 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2023-12-29 14:30:00 | 519.35 | 2024-01-08 10:15:00 | 533.80 | STOP_HIT | 1.00 | 2.78% |
| SELL | retest2 | 2024-01-11 10:15:00 | 525.65 | 2024-01-11 11:15:00 | 533.15 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-01-15 09:15:00 | 535.80 | 2024-01-15 09:15:00 | 528.10 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-01-16 13:00:00 | 526.40 | 2024-01-17 09:15:00 | 536.95 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-01-16 15:15:00 | 527.65 | 2024-01-17 09:15:00 | 536.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-01-17 12:15:00 | 526.00 | 2024-01-19 12:15:00 | 531.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-01-19 12:00:00 | 526.15 | 2024-01-19 12:15:00 | 531.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-02-05 12:15:00 | 558.30 | 2024-02-06 14:15:00 | 566.75 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-02-05 13:15:00 | 559.25 | 2024-02-06 14:15:00 | 566.75 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-02-05 14:15:00 | 558.70 | 2024-02-06 14:15:00 | 566.75 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-02-06 11:45:00 | 559.60 | 2024-02-06 14:15:00 | 566.75 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-02-20 15:00:00 | 588.10 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-02-21 15:00:00 | 589.55 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-02-22 09:30:00 | 588.40 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-02-22 10:30:00 | 588.95 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2024-02-23 09:15:00 | 593.00 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-02-23 11:30:00 | 593.60 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-02-23 12:00:00 | 594.00 | 2024-02-28 11:15:00 | 590.20 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-03-20 14:15:00 | 574.95 | 2024-03-21 09:15:00 | 587.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-03-26 11:15:00 | 599.25 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | 3.90% |
| BUY | retest2 | 2024-03-26 12:15:00 | 597.30 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | 4.24% |
| BUY | retest2 | 2024-03-26 13:00:00 | 597.10 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | 4.27% |
| BUY | retest2 | 2024-03-26 14:00:00 | 597.80 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | 4.15% |
| BUY | retest2 | 2024-04-05 11:15:00 | 631.40 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-04-05 12:15:00 | 631.85 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-04-08 11:00:00 | 632.45 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-04-08 13:30:00 | 631.35 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-04-09 09:30:00 | 632.65 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-04-09 10:00:00 | 632.25 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-04-09 11:15:00 | 632.45 | 2024-04-09 13:15:00 | 622.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-04-12 11:45:00 | 621.45 | 2024-04-18 09:15:00 | 627.85 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-04-25 14:00:00 | 636.25 | 2024-04-29 09:15:00 | 629.35 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-05-03 11:15:00 | 623.35 | 2024-05-07 12:15:00 | 592.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 12:45:00 | 620.05 | 2024-05-07 12:15:00 | 589.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:15:00 | 623.35 | 2024-05-08 12:15:00 | 598.30 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2024-05-03 12:45:00 | 620.05 | 2024-05-08 12:15:00 | 598.30 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2024-05-17 13:15:00 | 616.70 | 2024-05-27 12:15:00 | 628.95 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2024-05-30 09:15:00 | 626.85 | 2024-05-31 13:15:00 | 638.45 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-05-30 13:00:00 | 627.30 | 2024-05-31 13:15:00 | 638.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-05-31 09:45:00 | 627.00 | 2024-05-31 13:15:00 | 638.45 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-06-21 15:15:00 | 656.00 | 2024-06-26 13:15:00 | 658.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-06-25 09:30:00 | 653.00 | 2024-06-26 13:15:00 | 658.15 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-06-26 12:45:00 | 655.90 | 2024-06-26 13:15:00 | 658.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-07-04 13:15:00 | 683.45 | 2024-07-04 14:15:00 | 681.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-07-10 10:15:00 | 659.50 | 2024-07-11 14:15:00 | 679.70 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2024-07-10 11:00:00 | 660.50 | 2024-07-11 14:15:00 | 679.70 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-07-10 12:00:00 | 662.10 | 2024-07-11 14:15:00 | 679.70 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-07-10 12:45:00 | 661.70 | 2024-07-11 14:15:00 | 679.70 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-07-11 10:15:00 | 673.35 | 2024-07-11 14:15:00 | 679.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest1 | 2024-07-16 09:15:00 | 690.85 | 2024-07-18 10:15:00 | 680.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-07-18 12:15:00 | 682.55 | 2024-07-19 10:15:00 | 675.05 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest1 | 2024-08-06 12:30:00 | 637.55 | 2024-08-07 09:15:00 | 643.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-08 09:15:00 | 639.35 | 2024-08-16 11:15:00 | 630.40 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2024-08-08 10:45:00 | 641.00 | 2024-08-16 14:15:00 | 641.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-08 12:45:00 | 641.40 | 2024-08-16 14:15:00 | 641.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-08-09 09:45:00 | 640.85 | 2024-08-16 14:15:00 | 641.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-08-16 11:00:00 | 624.15 | 2024-08-16 14:15:00 | 641.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-08-21 11:30:00 | 627.30 | 2024-08-22 11:15:00 | 636.55 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-08-21 12:45:00 | 627.90 | 2024-08-22 11:15:00 | 636.55 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-22 09:15:00 | 627.45 | 2024-08-22 11:15:00 | 636.55 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-08-26 09:15:00 | 638.00 | 2024-08-26 09:15:00 | 633.25 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-09-06 11:45:00 | 626.50 | 2024-09-09 09:15:00 | 619.55 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-09-06 13:15:00 | 625.60 | 2024-09-09 09:15:00 | 619.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-16 09:15:00 | 632.35 | 2024-09-16 10:15:00 | 628.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-09-18 11:30:00 | 622.25 | 2024-09-23 10:15:00 | 622.05 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-09-26 09:15:00 | 612.25 | 2024-09-26 14:15:00 | 625.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-10-15 11:30:00 | 588.40 | 2024-10-18 09:15:00 | 558.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 13:00:00 | 587.35 | 2024-10-18 09:15:00 | 558.98 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2024-10-15 11:30:00 | 588.40 | 2024-10-18 14:15:00 | 572.85 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2024-10-15 13:00:00 | 587.35 | 2024-10-18 14:15:00 | 572.85 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2024-10-16 09:15:00 | 585.00 | 2024-10-22 14:15:00 | 557.98 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2024-10-17 09:15:00 | 588.40 | 2024-10-23 09:15:00 | 555.75 | PARTIAL | 0.50 | 5.55% |
| SELL | retest2 | 2024-10-16 09:15:00 | 585.00 | 2024-10-24 09:15:00 | 562.45 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2024-10-17 09:15:00 | 588.40 | 2024-10-24 09:15:00 | 562.45 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2024-10-17 11:00:00 | 580.15 | 2024-10-25 09:15:00 | 551.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 11:00:00 | 580.15 | 2024-10-28 09:15:00 | 554.40 | STOP_HIT | 0.50 | 4.44% |
| BUY | retest2 | 2024-10-31 10:15:00 | 584.60 | 2024-11-04 09:15:00 | 571.25 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-10-31 13:15:00 | 580.40 | 2024-11-04 09:15:00 | 571.25 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-10-31 14:30:00 | 580.90 | 2024-11-04 09:15:00 | 571.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-10-31 15:00:00 | 581.15 | 2024-11-04 09:15:00 | 571.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-12-09 12:00:00 | 572.90 | 2024-12-13 09:15:00 | 568.45 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-12-10 09:30:00 | 573.00 | 2024-12-13 09:15:00 | 568.45 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-12-10 14:30:00 | 573.60 | 2024-12-13 09:15:00 | 568.45 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-12 10:45:00 | 575.75 | 2024-12-13 09:15:00 | 568.45 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-12-24 10:45:00 | 545.65 | 2024-12-30 10:15:00 | 553.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-01-14 12:15:00 | 507.75 | 2025-01-14 15:15:00 | 516.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-01-14 12:45:00 | 507.65 | 2025-01-14 15:15:00 | 516.20 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-01-21 09:15:00 | 544.65 | 2025-01-21 14:15:00 | 531.85 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-01-21 10:45:00 | 536.40 | 2025-01-21 14:15:00 | 531.85 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-01-21 14:15:00 | 535.95 | 2025-01-21 14:15:00 | 531.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-02-07 11:15:00 | 521.35 | 2025-02-07 13:15:00 | 511.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-02-07 13:15:00 | 516.15 | 2025-02-07 13:15:00 | 511.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-02-18 10:15:00 | 476.25 | 2025-03-03 15:15:00 | 470.80 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-02-24 09:15:00 | 475.00 | 2025-03-03 15:15:00 | 470.80 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-02-24 11:45:00 | 475.70 | 2025-03-03 15:15:00 | 470.80 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-03-26 09:15:00 | 528.55 | 2025-04-01 14:15:00 | 529.60 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-04-02 13:45:00 | 532.90 | 2025-04-03 09:15:00 | 540.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-04-02 14:30:00 | 532.90 | 2025-04-03 09:15:00 | 540.90 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-04-04 11:15:00 | 535.90 | 2025-04-04 12:15:00 | 532.20 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-07 12:15:00 | 517.90 | 2025-04-08 09:15:00 | 532.05 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-04-23 11:45:00 | 572.60 | 2025-04-24 12:15:00 | 570.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-04-23 12:45:00 | 572.05 | 2025-04-24 12:15:00 | 570.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-05-08 09:45:00 | 533.25 | 2025-05-12 09:15:00 | 542.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-08 13:15:00 | 531.20 | 2025-05-12 09:15:00 | 542.30 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-05-21 09:30:00 | 565.50 | 2025-05-26 14:15:00 | 567.50 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-05-28 13:00:00 | 560.20 | 2025-06-06 14:15:00 | 555.75 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-05-28 15:15:00 | 559.90 | 2025-06-06 14:15:00 | 555.75 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-05-29 11:45:00 | 560.00 | 2025-06-06 14:15:00 | 555.75 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-05-30 09:30:00 | 560.15 | 2025-06-06 14:15:00 | 555.75 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-06-02 14:30:00 | 554.80 | 2025-06-06 14:15:00 | 555.75 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-06-02 15:00:00 | 554.75 | 2025-06-06 14:15:00 | 555.75 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-06-17 09:15:00 | 549.70 | 2025-06-17 10:15:00 | 554.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-11 15:15:00 | 584.15 | 2025-07-14 14:15:00 | 589.95 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-14 11:15:00 | 585.05 | 2025-07-14 14:15:00 | 589.95 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-07-16 13:15:00 | 592.50 | 2025-07-25 13:15:00 | 613.85 | STOP_HIT | 1.00 | 3.60% |
| BUY | retest2 | 2025-07-17 11:30:00 | 592.20 | 2025-07-25 13:15:00 | 613.85 | STOP_HIT | 1.00 | 3.66% |
| BUY | retest2 | 2025-07-17 12:30:00 | 592.35 | 2025-07-25 13:15:00 | 613.85 | STOP_HIT | 1.00 | 3.63% |
| SELL | retest2 | 2025-07-28 12:00:00 | 615.40 | 2025-07-30 11:15:00 | 617.75 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-08-04 13:15:00 | 604.65 | 2025-08-13 11:15:00 | 574.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 14:00:00 | 605.00 | 2025-08-13 11:15:00 | 574.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 15:00:00 | 605.20 | 2025-08-13 11:15:00 | 574.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-04 13:15:00 | 604.65 | 2025-08-14 14:15:00 | 578.75 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2025-08-04 14:00:00 | 605.00 | 2025-08-14 14:15:00 | 578.75 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2025-08-04 15:00:00 | 605.20 | 2025-08-14 14:15:00 | 578.75 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2025-09-08 11:15:00 | 568.95 | 2025-09-08 11:15:00 | 573.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-09 09:30:00 | 573.80 | 2025-09-09 11:15:00 | 568.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-12 14:30:00 | 560.40 | 2025-09-15 09:15:00 | 568.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-12 15:00:00 | 560.40 | 2025-09-15 09:15:00 | 568.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-19 09:15:00 | 584.00 | 2025-09-24 14:15:00 | 582.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-09-19 12:30:00 | 584.20 | 2025-09-24 14:15:00 | 582.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-09-19 14:30:00 | 583.20 | 2025-09-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-09-19 15:15:00 | 582.80 | 2025-09-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-09-23 13:45:00 | 589.60 | 2025-09-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-24 09:45:00 | 591.80 | 2025-09-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-01 10:15:00 | 568.05 | 2025-10-01 15:15:00 | 573.15 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-07 09:15:00 | 575.55 | 2025-10-07 10:15:00 | 569.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-28 10:45:00 | 559.00 | 2025-10-29 10:15:00 | 565.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-31 12:45:00 | 570.90 | 2025-11-06 09:15:00 | 561.80 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-11-03 09:15:00 | 569.45 | 2025-11-06 09:15:00 | 561.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-11-10 13:15:00 | 559.90 | 2025-11-12 09:15:00 | 566.10 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-11-11 14:00:00 | 558.50 | 2025-11-12 09:15:00 | 566.10 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-11-19 10:15:00 | 556.10 | 2025-11-26 14:15:00 | 551.55 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-11-20 09:45:00 | 556.95 | 2025-11-26 14:15:00 | 551.55 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-11-20 14:30:00 | 557.15 | 2025-11-26 14:15:00 | 551.55 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-12-03 10:30:00 | 535.75 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-12-04 13:15:00 | 536.65 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-12-08 09:45:00 | 536.25 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-12-08 10:45:00 | 537.95 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-12-08 12:45:00 | 532.30 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-12-08 13:15:00 | 530.60 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-09 14:30:00 | 532.15 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-09 15:15:00 | 532.25 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-10 10:45:00 | 532.20 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-10 12:30:00 | 532.05 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-10 14:00:00 | 532.10 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-11 11:45:00 | 531.25 | 2025-12-11 14:15:00 | 535.25 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-13 14:00:00 | 535.35 | 2026-01-14 10:15:00 | 545.25 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-01-14 09:15:00 | 536.05 | 2026-01-14 10:15:00 | 545.25 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-20 09:15:00 | 552.75 | 2026-01-20 10:15:00 | 548.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-20 10:00:00 | 552.85 | 2026-01-20 10:15:00 | 548.30 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-02-05 14:45:00 | 532.70 | 2026-02-06 09:15:00 | 525.40 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-11 12:45:00 | 539.35 | 2026-02-12 10:15:00 | 535.35 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-02-11 13:15:00 | 539.75 | 2026-02-12 10:15:00 | 535.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-19 11:30:00 | 520.35 | 2026-03-02 09:15:00 | 494.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 520.35 | 2026-03-02 15:15:00 | 491.90 | STOP_HIT | 0.50 | 5.47% |
| BUY | retest2 | 2026-04-02 11:45:00 | 410.95 | 2026-04-08 10:15:00 | 452.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:30:00 | 411.45 | 2026-04-08 10:15:00 | 452.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 462.60 | 2026-04-29 13:15:00 | 454.35 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-04-29 10:30:00 | 462.50 | 2026-04-29 13:15:00 | 454.35 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-04-29 12:00:00 | 462.60 | 2026-04-29 13:15:00 | 454.35 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-05-04 13:15:00 | 446.65 | 2026-05-07 09:15:00 | 447.65 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-05-04 14:15:00 | 441.15 | 2026-05-07 09:15:00 | 447.65 | STOP_HIT | 1.00 | -1.47% |
