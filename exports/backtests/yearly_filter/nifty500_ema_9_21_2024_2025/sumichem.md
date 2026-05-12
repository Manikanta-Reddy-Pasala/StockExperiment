# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 485.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 97 |
| ALERT2 | 93 |
| ALERT2_SKIP | 44 |
| ALERT3 | 284 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 144 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 150 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 161 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 119
- **Target hits / Stop hits / Partials:** 7 / 142 / 12
- **Avg / median % per leg:** -0.20% / -1.18%
- **Sum % (uncompounded):** -32.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 17 | 23.6% | 6 | 66 | 0 | -0.10% | -7.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.40% | -4.8% |
| BUY @ 3rd Alert (retest2) | 70 | 17 | 24.3% | 6 | 64 | 0 | -0.03% | -2.3% |
| SELL (all) | 89 | 25 | 28.1% | 1 | 76 | 12 | -0.29% | -25.7% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 3 | 1 | 2.19% | 8.8% |
| SELL @ 3rd Alert (retest2) | 85 | 21 | 24.7% | 1 | 73 | 11 | -0.41% | -34.4% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 5 | 1 | 0.66% | 4.0% |
| retest2 (combined) | 155 | 38 | 24.5% | 7 | 137 | 11 | -0.24% | -36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 400.40 | 390.63 | 389.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 403.20 | 394.77 | 391.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 395.90 | 396.92 | 394.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:45:00 | 395.85 | 396.92 | 394.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 398.40 | 397.21 | 394.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:15:00 | 401.00 | 394.57 | 394.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 09:15:00 | 441.10 | 419.34 | 412.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 09:15:00 | 461.80 | 467.84 | 468.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 11:15:00 | 454.55 | 464.09 | 466.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 14:15:00 | 470.80 | 462.62 | 465.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 14:15:00 | 470.80 | 462.62 | 465.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 470.80 | 462.62 | 465.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:00:00 | 470.80 | 462.62 | 465.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 464.40 | 462.98 | 465.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 452.20 | 462.98 | 465.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 456.75 | 458.82 | 462.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 433.91 | 454.22 | 460.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 429.59 | 453.84 | 459.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 13:15:00 | 457.55 | 454.58 | 459.41 | SL hit (close>ema200) qty=0.50 sl=454.58 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 485.05 | 461.98 | 461.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 15:15:00 | 500.15 | 486.85 | 482.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 11:15:00 | 488.00 | 488.50 | 484.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:00:00 | 488.00 | 488.50 | 484.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 486.30 | 488.22 | 485.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 490.50 | 488.22 | 485.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 09:15:00 | 482.00 | 486.98 | 485.43 | SL hit (close<static) qty=1.00 sl=485.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 481.00 | 483.82 | 484.20 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 14:15:00 | 487.75 | 485.03 | 484.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 499.00 | 489.17 | 487.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 483.55 | 492.40 | 490.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 483.55 | 492.40 | 490.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 483.55 | 492.40 | 490.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 483.55 | 492.40 | 490.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 483.25 | 490.57 | 489.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 483.25 | 490.57 | 489.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 484.90 | 489.44 | 489.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 14:15:00 | 479.15 | 485.79 | 487.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 485.65 | 482.28 | 485.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 11:15:00 | 485.65 | 482.28 | 485.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 485.65 | 482.28 | 485.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:00:00 | 485.65 | 482.28 | 485.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 478.65 | 481.56 | 484.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 14:00:00 | 477.15 | 480.67 | 483.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:15:00 | 470.00 | 480.24 | 483.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 13:15:00 | 489.55 | 481.74 | 482.40 | SL hit (close>static) qty=1.00 sl=489.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 14:15:00 | 494.50 | 484.29 | 483.50 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 485.65 | 488.37 | 488.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 483.50 | 487.40 | 488.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 485.75 | 484.51 | 486.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 485.75 | 484.51 | 486.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 485.75 | 484.51 | 486.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 486.95 | 484.51 | 486.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 485.70 | 484.75 | 486.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:30:00 | 486.05 | 484.75 | 486.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 486.00 | 485.00 | 486.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:30:00 | 486.25 | 485.00 | 486.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 485.35 | 485.07 | 485.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:30:00 | 485.80 | 485.07 | 485.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 484.10 | 484.87 | 485.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:00:00 | 477.80 | 482.93 | 484.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 481.25 | 481.26 | 483.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 481.25 | 481.26 | 483.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 481.00 | 481.21 | 482.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 479.00 | 479.64 | 481.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 481.15 | 479.64 | 481.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 493.50 | 482.41 | 482.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 493.50 | 482.41 | 482.47 | SL hit (close>static) qty=1.00 sl=486.25 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 487.00 | 483.33 | 482.88 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 477.90 | 482.08 | 482.59 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 490.00 | 483.66 | 483.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 10:15:00 | 495.00 | 485.93 | 484.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 11:15:00 | 503.10 | 504.05 | 496.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 12:00:00 | 503.10 | 504.05 | 496.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 494.55 | 501.50 | 496.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 494.55 | 501.50 | 496.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 493.65 | 499.93 | 496.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 494.00 | 499.93 | 496.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 499.60 | 498.78 | 496.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 498.35 | 498.78 | 496.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 496.80 | 498.38 | 496.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 496.80 | 498.38 | 496.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 497.75 | 498.25 | 496.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 498.45 | 498.25 | 496.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 493.40 | 496.91 | 496.38 | SL hit (close<static) qty=1.00 sl=495.10 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 489.95 | 494.85 | 495.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 489.15 | 493.19 | 494.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 492.60 | 491.47 | 493.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 492.60 | 491.47 | 493.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 492.60 | 491.47 | 493.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 492.60 | 491.47 | 493.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 496.60 | 492.50 | 493.49 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 12:15:00 | 499.25 | 494.89 | 494.47 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 490.35 | 494.77 | 495.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 486.20 | 493.06 | 494.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 13:15:00 | 497.55 | 492.10 | 493.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 13:15:00 | 497.55 | 492.10 | 493.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 497.55 | 492.10 | 493.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:45:00 | 497.60 | 492.10 | 493.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 496.50 | 492.98 | 493.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:30:00 | 500.75 | 492.98 | 493.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 498.00 | 494.46 | 494.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 15:15:00 | 505.00 | 500.15 | 498.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 499.60 | 500.04 | 498.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 499.60 | 500.04 | 498.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 499.60 | 500.04 | 498.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 499.60 | 500.04 | 498.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 498.95 | 499.82 | 498.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 498.45 | 499.82 | 498.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 495.45 | 498.95 | 498.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 495.45 | 498.95 | 498.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 496.00 | 498.36 | 497.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 14:45:00 | 502.30 | 498.72 | 498.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 504.80 | 499.96 | 499.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:45:00 | 502.85 | 500.84 | 499.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 504.50 | 501.46 | 500.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 500.35 | 501.64 | 500.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 500.35 | 501.64 | 500.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 506.00 | 502.51 | 501.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-19 15:15:00 | 496.90 | 501.16 | 501.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 496.90 | 501.16 | 501.47 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 503.85 | 501.70 | 501.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 11:15:00 | 504.00 | 502.50 | 502.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 507.30 | 511.61 | 508.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 507.30 | 511.61 | 508.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 507.30 | 511.61 | 508.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 504.25 | 511.61 | 508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 513.90 | 512.07 | 508.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 511.55 | 512.07 | 508.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 510.20 | 512.91 | 510.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:15:00 | 509.05 | 512.91 | 510.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 509.70 | 512.27 | 510.13 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 13:15:00 | 501.25 | 508.54 | 508.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 14:15:00 | 498.85 | 506.60 | 507.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 509.45 | 502.92 | 504.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 509.45 | 502.92 | 504.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 509.45 | 502.92 | 504.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 509.45 | 502.92 | 504.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 508.30 | 504.00 | 504.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 508.30 | 504.00 | 504.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 514.35 | 506.73 | 506.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 531.90 | 511.51 | 508.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 524.00 | 527.25 | 520.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:00:00 | 524.00 | 527.25 | 520.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 520.60 | 525.45 | 521.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 520.60 | 525.45 | 521.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 517.40 | 523.84 | 520.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 517.40 | 523.84 | 520.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 520.00 | 523.07 | 520.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 524.75 | 523.07 | 520.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 521.45 | 523.12 | 522.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 519.25 | 521.74 | 522.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 519.25 | 521.74 | 522.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 514.60 | 520.35 | 521.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 507.00 | 501.38 | 507.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 507.00 | 501.38 | 507.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 507.00 | 501.38 | 507.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 492.20 | 498.85 | 503.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:45:00 | 494.90 | 496.53 | 501.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 516.00 | 503.41 | 502.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 516.00 | 503.41 | 502.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 522.65 | 512.19 | 508.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 516.85 | 517.59 | 512.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:45:00 | 516.00 | 517.59 | 512.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 525.85 | 519.47 | 514.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:45:00 | 533.00 | 521.88 | 516.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:15:00 | 532.50 | 523.51 | 517.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 13:00:00 | 533.40 | 532.32 | 526.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:45:00 | 531.60 | 530.88 | 527.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 531.90 | 535.77 | 532.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 531.90 | 535.77 | 532.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 531.00 | 534.82 | 532.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 531.00 | 534.82 | 532.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 533.40 | 534.53 | 532.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 12:30:00 | 537.95 | 535.20 | 532.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 13:30:00 | 540.70 | 536.84 | 533.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 13:15:00 | 530.55 | 534.11 | 533.91 | SL hit (close<static) qty=1.00 sl=530.90 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 531.30 | 533.55 | 533.67 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 538.60 | 534.28 | 533.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 543.60 | 538.54 | 536.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 538.50 | 539.11 | 537.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 13:15:00 | 538.50 | 539.11 | 537.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 538.50 | 539.11 | 537.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:30:00 | 537.50 | 539.11 | 537.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 544.65 | 540.22 | 538.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 538.05 | 540.22 | 538.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 543.05 | 545.82 | 542.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 544.85 | 545.82 | 542.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 540.90 | 544.83 | 542.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 540.60 | 544.83 | 542.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 538.00 | 543.47 | 542.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 539.20 | 543.47 | 542.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 540.00 | 542.77 | 541.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:30:00 | 542.00 | 542.22 | 541.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 12:30:00 | 540.45 | 541.71 | 541.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:15:00 | 542.00 | 541.71 | 541.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 14:15:00 | 532.70 | 539.79 | 540.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 532.70 | 539.79 | 540.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 15:15:00 | 531.90 | 538.21 | 539.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 539.35 | 530.34 | 533.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 539.35 | 530.34 | 533.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 539.35 | 530.34 | 533.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 539.35 | 530.34 | 533.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 539.75 | 532.22 | 534.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:15:00 | 533.40 | 532.22 | 534.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:00:00 | 537.35 | 532.95 | 533.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 539.00 | 534.16 | 533.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 539.00 | 534.16 | 533.80 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 530.25 | 533.17 | 533.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 523.90 | 530.24 | 531.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 521.95 | 518.47 | 523.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 09:15:00 | 521.95 | 518.47 | 523.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 521.95 | 518.47 | 523.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 522.10 | 518.47 | 523.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 517.25 | 517.64 | 521.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 516.70 | 517.64 | 521.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 523.35 | 513.29 | 515.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:45:00 | 524.40 | 513.29 | 515.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 523.10 | 515.25 | 516.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:00:00 | 523.10 | 515.25 | 516.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 518.70 | 516.21 | 516.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:45:00 | 519.10 | 516.21 | 516.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 13:15:00 | 520.90 | 517.15 | 517.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 523.20 | 519.29 | 518.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 12:15:00 | 519.10 | 519.25 | 518.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 12:15:00 | 519.10 | 519.25 | 518.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 519.10 | 519.25 | 518.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:45:00 | 519.60 | 519.25 | 518.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 518.00 | 519.00 | 518.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:00:00 | 518.00 | 519.00 | 518.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 516.65 | 518.53 | 518.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:30:00 | 513.95 | 518.53 | 518.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 531.75 | 521.07 | 519.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 10:30:00 | 537.55 | 524.26 | 520.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:45:00 | 536.80 | 532.45 | 525.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 533.90 | 533.20 | 528.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 13:30:00 | 536.15 | 533.28 | 529.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-10 14:15:00 | 591.30 | 566.79 | 549.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 584.40 | 597.14 | 597.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 576.65 | 593.04 | 595.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 565.50 | 564.72 | 574.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:30:00 | 570.05 | 564.72 | 574.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 565.30 | 566.03 | 571.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:45:00 | 561.05 | 567.11 | 571.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 15:15:00 | 560.50 | 566.72 | 570.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:45:00 | 559.65 | 564.16 | 568.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 560.85 | 554.98 | 556.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 557.85 | 556.33 | 556.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 559.85 | 556.33 | 556.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 556.85 | 556.58 | 556.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 551.90 | 556.57 | 556.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:00:00 | 553.20 | 555.23 | 555.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:30:00 | 553.75 | 555.64 | 556.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 569.85 | 558.54 | 557.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 569.85 | 558.54 | 557.31 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 552.35 | 559.13 | 559.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 550.95 | 557.50 | 558.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 555.90 | 553.83 | 556.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 555.90 | 553.83 | 556.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 555.90 | 553.83 | 556.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:30:00 | 538.50 | 548.84 | 552.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 548.50 | 541.25 | 541.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 548.50 | 541.25 | 541.00 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 11:15:00 | 541.60 | 543.87 | 544.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 13:15:00 | 540.85 | 542.97 | 543.68 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 555.35 | 544.70 | 544.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 557.55 | 547.27 | 545.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 13:15:00 | 548.40 | 548.60 | 546.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 14:00:00 | 548.40 | 548.60 | 546.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 559.55 | 565.35 | 561.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 559.55 | 565.35 | 561.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 560.20 | 564.32 | 561.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 559.25 | 564.32 | 561.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 549.20 | 561.30 | 560.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 549.20 | 561.30 | 560.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 549.35 | 558.91 | 559.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 15:15:00 | 547.40 | 553.88 | 556.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 514.05 | 513.03 | 522.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 13:00:00 | 514.05 | 513.03 | 522.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 518.80 | 514.19 | 522.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 518.80 | 514.19 | 522.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 515.70 | 515.18 | 521.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:45:00 | 506.55 | 514.91 | 518.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 519.70 | 510.56 | 510.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 519.70 | 510.56 | 510.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 537.10 | 515.87 | 512.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 569.90 | 570.48 | 561.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 569.90 | 570.48 | 561.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 557.95 | 567.97 | 560.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 557.95 | 567.97 | 560.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 559.65 | 566.31 | 560.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 558.60 | 566.31 | 560.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 560.65 | 565.18 | 560.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 564.85 | 563.20 | 561.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 564.90 | 563.21 | 561.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:00:00 | 567.10 | 563.99 | 561.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:00:00 | 564.65 | 563.72 | 562.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 567.25 | 566.19 | 563.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:45:00 | 563.10 | 566.19 | 563.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 562.80 | 566.71 | 564.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 562.80 | 566.71 | 564.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 564.10 | 566.19 | 564.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 564.10 | 566.19 | 564.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 563.70 | 565.69 | 564.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 575.45 | 565.69 | 564.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 574.30 | 567.41 | 565.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 11:00:00 | 592.05 | 572.34 | 567.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 560.00 | 567.82 | 567.68 | SL hit (close<static) qty=1.00 sl=561.65 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 554.40 | 565.13 | 566.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 549.00 | 558.58 | 562.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 524.05 | 519.14 | 527.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 524.05 | 519.14 | 527.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 525.50 | 520.13 | 526.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 524.15 | 520.13 | 526.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 526.90 | 521.49 | 526.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 526.90 | 521.49 | 526.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 524.10 | 522.01 | 526.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:30:00 | 521.25 | 521.52 | 525.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:00:00 | 519.55 | 521.52 | 525.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 515.60 | 522.00 | 525.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:30:00 | 521.50 | 519.40 | 521.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 526.75 | 520.87 | 522.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 526.75 | 520.87 | 522.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 525.00 | 521.70 | 522.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:45:00 | 526.95 | 521.70 | 522.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 519.20 | 521.56 | 522.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 515.00 | 521.56 | 522.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 10:30:00 | 515.70 | 518.44 | 520.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 540.50 | 519.30 | 517.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 540.50 | 519.30 | 517.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 544.80 | 524.40 | 520.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 544.30 | 545.45 | 536.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 544.30 | 545.45 | 536.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 547.50 | 545.21 | 539.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 542.05 | 545.21 | 539.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 539.25 | 544.40 | 541.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 539.85 | 544.40 | 541.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 541.65 | 543.85 | 541.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 545.95 | 543.51 | 541.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 534.50 | 540.33 | 540.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 534.50 | 540.33 | 540.73 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 548.05 | 540.47 | 540.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 555.90 | 548.58 | 546.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 551.50 | 552.45 | 549.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:00:00 | 551.50 | 552.45 | 549.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 549.70 | 552.62 | 550.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 549.70 | 552.62 | 550.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 550.00 | 552.10 | 550.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 546.75 | 551.03 | 550.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 547.50 | 550.32 | 550.15 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 11:15:00 | 548.35 | 549.93 | 549.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 12:15:00 | 546.50 | 549.24 | 549.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 545.35 | 542.91 | 545.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 545.35 | 542.91 | 545.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 545.35 | 542.91 | 545.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 545.35 | 542.91 | 545.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 542.90 | 542.91 | 545.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 544.80 | 542.91 | 545.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 538.90 | 542.10 | 544.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 11:45:00 | 536.45 | 540.69 | 543.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 541.00 | 531.08 | 530.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 541.00 | 531.08 | 530.55 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 526.50 | 530.86 | 531.27 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 15:15:00 | 532.05 | 531.34 | 531.32 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 519.30 | 528.93 | 530.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 513.25 | 519.85 | 524.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 14:15:00 | 524.40 | 517.94 | 521.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 524.40 | 517.94 | 521.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 524.40 | 517.94 | 521.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 524.40 | 517.94 | 521.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 529.10 | 520.17 | 522.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 518.35 | 520.17 | 522.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 526.10 | 517.91 | 517.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 526.10 | 517.91 | 517.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 542.30 | 526.02 | 523.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 531.00 | 533.03 | 528.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 09:30:00 | 530.00 | 533.03 | 528.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 532.30 | 534.41 | 531.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 526.15 | 534.41 | 531.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 529.50 | 533.20 | 531.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 529.65 | 533.20 | 531.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 529.10 | 532.38 | 530.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:30:00 | 528.80 | 532.38 | 530.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 529.30 | 531.76 | 530.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:45:00 | 528.50 | 531.76 | 530.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 530.30 | 531.05 | 530.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:30:00 | 528.35 | 531.05 | 530.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 528.50 | 530.25 | 530.27 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 536.80 | 531.15 | 530.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 542.55 | 533.52 | 531.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 537.05 | 538.13 | 535.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 537.05 | 538.13 | 535.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 537.05 | 538.13 | 535.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 537.05 | 538.13 | 535.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 535.20 | 537.54 | 535.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 530.00 | 537.54 | 535.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 531.30 | 536.29 | 535.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 530.05 | 536.29 | 535.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 522.70 | 533.57 | 534.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 517.00 | 528.61 | 531.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 523.05 | 522.07 | 527.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 523.05 | 522.07 | 527.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 524.00 | 522.54 | 525.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:15:00 | 524.00 | 522.54 | 525.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 524.00 | 522.83 | 525.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 515.95 | 522.83 | 525.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 490.15 | 501.99 | 510.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 11:15:00 | 464.36 | 475.54 | 489.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 484.95 | 478.74 | 478.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 490.90 | 483.66 | 481.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 492.90 | 494.54 | 491.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 492.90 | 494.54 | 491.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 492.90 | 494.54 | 491.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 492.90 | 494.54 | 491.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 493.05 | 494.24 | 491.41 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 482.85 | 489.33 | 490.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 479.65 | 487.39 | 489.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 489.00 | 483.46 | 486.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 489.00 | 483.46 | 486.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 489.00 | 483.46 | 486.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 489.00 | 483.46 | 486.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 487.60 | 484.29 | 486.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 487.60 | 484.29 | 486.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 483.00 | 483.85 | 485.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:30:00 | 482.50 | 483.58 | 485.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 481.00 | 481.78 | 484.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 12:15:00 | 490.50 | 483.99 | 484.53 | SL hit (close>static) qty=1.00 sl=486.70 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-27 14:15:00 | 503.45 | 483.64 | 483.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 515.90 | 504.19 | 498.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 11:15:00 | 516.80 | 516.94 | 510.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 12:00:00 | 516.80 | 516.94 | 510.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 510.25 | 515.09 | 510.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 510.25 | 515.09 | 510.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 511.15 | 514.30 | 510.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 513.40 | 513.80 | 510.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:15:00 | 518.05 | 513.59 | 511.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 513.50 | 514.28 | 512.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:45:00 | 513.40 | 514.00 | 512.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 512.15 | 513.63 | 512.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 511.95 | 513.63 | 512.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 508.10 | 512.53 | 511.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-01 15:15:00 | 508.10 | 512.53 | 511.83 | SL hit (close<static) qty=1.00 sl=509.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 513.50 | 528.08 | 528.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 511.85 | 524.83 | 527.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 487.15 | 485.17 | 493.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:15:00 | 486.80 | 485.17 | 493.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 488.30 | 486.11 | 491.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:30:00 | 494.80 | 486.11 | 491.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 475.35 | 471.88 | 476.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 476.50 | 471.88 | 476.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 473.90 | 472.29 | 476.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 473.90 | 472.29 | 476.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 475.90 | 473.01 | 476.42 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 478.90 | 475.78 | 475.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 483.90 | 479.17 | 477.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 492.90 | 493.41 | 489.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 13:45:00 | 493.55 | 493.41 | 489.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 484.35 | 492.48 | 490.07 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 484.90 | 488.53 | 488.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 480.75 | 485.38 | 487.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 479.85 | 479.48 | 482.85 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 14:15:00 | 472.15 | 476.29 | 480.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:15:00 | 468.40 | 475.74 | 479.18 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 448.54 | 459.53 | 467.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 458.50 | 457.25 | 463.48 | SL hit (close>ema200) qty=0.50 sl=457.25 alert=retest1 |

### Cycle 55 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 470.45 | 465.38 | 465.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 472.25 | 466.76 | 465.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 12:15:00 | 469.25 | 471.09 | 468.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 12:15:00 | 469.25 | 471.09 | 468.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 469.25 | 471.09 | 468.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:45:00 | 469.95 | 471.09 | 468.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 475.95 | 472.06 | 469.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 14:30:00 | 478.25 | 473.87 | 470.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 13:15:00 | 488.25 | 489.82 | 489.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 488.25 | 489.82 | 489.96 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 494.80 | 490.81 | 490.40 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 484.50 | 489.51 | 489.87 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 493.00 | 490.43 | 490.14 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 486.30 | 489.41 | 489.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 485.10 | 487.06 | 488.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 12:15:00 | 486.50 | 486.45 | 487.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 13:15:00 | 486.20 | 486.45 | 487.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 488.50 | 486.86 | 487.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 488.50 | 486.86 | 487.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 488.50 | 487.19 | 487.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:30:00 | 488.50 | 487.19 | 487.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 489.00 | 487.55 | 487.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 489.70 | 487.55 | 487.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 493.80 | 488.80 | 488.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 494.30 | 489.90 | 488.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 13:15:00 | 483.90 | 489.32 | 489.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 13:15:00 | 483.90 | 489.32 | 489.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 483.90 | 489.32 | 489.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 483.90 | 489.32 | 489.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 489.35 | 489.33 | 489.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 494.20 | 489.38 | 489.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-28 12:15:00 | 543.62 | 537.52 | 534.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 558.60 | 561.78 | 561.84 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 562.60 | 561.95 | 561.91 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 532.15 | 555.99 | 559.20 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 561.55 | 552.06 | 552.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 565.90 | 560.72 | 558.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 15:15:00 | 560.10 | 562.52 | 560.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 557.60 | 561.54 | 559.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 557.60 | 561.54 | 559.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 557.60 | 561.54 | 559.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 552.80 | 559.79 | 559.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:00:00 | 552.80 | 559.79 | 559.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 11:15:00 | 553.10 | 558.45 | 558.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 09:15:00 | 550.90 | 553.10 | 554.31 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 567.35 | 555.95 | 555.49 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 551.35 | 555.33 | 555.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 540.50 | 550.10 | 552.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 521.15 | 520.37 | 529.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 521.15 | 520.37 | 529.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 509.40 | 512.13 | 519.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:30:00 | 508.20 | 510.88 | 518.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:30:00 | 507.00 | 506.24 | 510.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 15:00:00 | 508.35 | 506.24 | 510.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 521.50 | 509.22 | 510.93 | SL hit (close>static) qty=1.00 sl=520.30 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 524.15 | 514.14 | 512.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 13:15:00 | 527.00 | 518.45 | 515.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 11:15:00 | 519.75 | 521.27 | 518.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 11:15:00 | 519.75 | 521.27 | 518.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 519.75 | 521.27 | 518.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 518.90 | 521.27 | 518.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 517.50 | 520.52 | 518.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 517.15 | 520.52 | 518.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 515.55 | 519.53 | 517.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:30:00 | 516.20 | 519.53 | 517.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 514.70 | 518.56 | 517.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 514.70 | 518.56 | 517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 518.35 | 517.76 | 517.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:15:00 | 516.50 | 517.76 | 517.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 515.85 | 517.38 | 517.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 515.30 | 517.38 | 517.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 510.65 | 516.03 | 516.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 509.15 | 514.66 | 515.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 513.90 | 510.02 | 512.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 513.90 | 510.02 | 512.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 513.90 | 510.02 | 512.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 513.90 | 510.02 | 512.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 512.85 | 510.59 | 512.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 515.00 | 510.59 | 512.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 514.15 | 511.30 | 512.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 514.15 | 511.30 | 512.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 518.10 | 512.66 | 513.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 518.10 | 512.66 | 513.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 512.90 | 512.71 | 513.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 518.10 | 512.71 | 513.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 518.10 | 513.79 | 513.64 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 510.60 | 513.20 | 513.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 504.80 | 511.52 | 512.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 519.15 | 506.54 | 507.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 519.15 | 506.54 | 507.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 519.15 | 506.54 | 507.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 521.70 | 506.54 | 507.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 518.80 | 508.99 | 508.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 522.20 | 516.68 | 513.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 525.15 | 525.27 | 521.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 525.40 | 525.27 | 521.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 526.60 | 525.84 | 522.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 526.60 | 525.84 | 522.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 520.45 | 524.46 | 523.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 520.45 | 524.46 | 523.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 522.25 | 524.02 | 522.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 525.75 | 524.02 | 522.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 523.90 | 524.99 | 523.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 533.35 | 534.63 | 534.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 533.35 | 534.63 | 534.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 531.60 | 534.02 | 534.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 534.35 | 533.24 | 533.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 534.35 | 533.24 | 533.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 534.35 | 533.24 | 533.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 534.35 | 533.24 | 533.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 532.75 | 533.14 | 533.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 525.30 | 530.94 | 532.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 527.85 | 530.32 | 531.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 514.10 | 531.31 | 531.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 501.46 | 510.32 | 518.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 513.00 | 510.65 | 517.41 | SL hit (close>ema200) qty=0.50 sl=510.65 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 513.25 | 506.13 | 505.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 516.15 | 508.13 | 506.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 515.45 | 517.01 | 513.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 515.45 | 517.01 | 513.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 519.05 | 517.16 | 514.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:30:00 | 514.65 | 517.16 | 514.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 515.55 | 516.97 | 514.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 515.85 | 516.97 | 514.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 513.10 | 516.19 | 514.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 513.10 | 516.19 | 514.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 512.10 | 515.38 | 514.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 512.20 | 515.38 | 514.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 510.45 | 513.36 | 513.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 509.45 | 512.32 | 513.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 495.00 | 494.92 | 499.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 497.05 | 494.92 | 499.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 495.00 | 494.94 | 499.00 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 500.20 | 498.15 | 498.14 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 495.45 | 497.61 | 497.90 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 13:15:00 | 499.00 | 498.18 | 498.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 500.35 | 498.62 | 498.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 497.85 | 500.85 | 500.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 497.85 | 500.85 | 500.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 497.85 | 500.85 | 500.09 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 495.45 | 499.22 | 499.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 492.10 | 497.79 | 498.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 500.05 | 496.21 | 497.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 500.05 | 496.21 | 497.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 500.05 | 496.21 | 497.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 499.70 | 496.21 | 497.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 505.50 | 498.06 | 498.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 505.95 | 498.06 | 498.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 509.05 | 500.26 | 499.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 512.00 | 502.61 | 500.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 514.80 | 515.31 | 510.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:00:00 | 514.80 | 515.31 | 510.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 512.30 | 513.66 | 511.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 512.30 | 513.66 | 511.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 511.00 | 513.13 | 511.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 511.75 | 513.13 | 511.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 513.65 | 513.23 | 511.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 516.15 | 512.85 | 511.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 516.25 | 514.17 | 512.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 516.40 | 514.69 | 512.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 516.15 | 514.89 | 513.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 527.40 | 522.50 | 519.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 523.50 | 522.50 | 519.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 520.00 | 524.59 | 522.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 520.00 | 524.59 | 522.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 517.40 | 523.15 | 521.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 517.40 | 523.15 | 521.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 516.00 | 520.61 | 520.86 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 525.65 | 521.18 | 520.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 532.30 | 525.32 | 523.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 532.80 | 533.47 | 530.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:45:00 | 533.10 | 533.47 | 530.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 542.45 | 535.27 | 531.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 545.10 | 535.27 | 531.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:45:00 | 543.85 | 547.89 | 545.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 545.25 | 547.38 | 545.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 537.45 | 544.86 | 545.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 537.45 | 544.86 | 545.00 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 555.35 | 544.29 | 542.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 559.85 | 547.40 | 544.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 601.45 | 602.53 | 593.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 601.45 | 602.53 | 593.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 594.95 | 600.82 | 594.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 594.95 | 600.82 | 594.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 585.45 | 597.74 | 593.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 585.45 | 597.74 | 593.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 585.00 | 595.20 | 592.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 585.00 | 595.20 | 592.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 592.45 | 593.03 | 592.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:30:00 | 595.60 | 593.18 | 592.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 595.00 | 593.52 | 592.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 12:15:00 | 589.25 | 592.40 | 592.38 | SL hit (close<static) qty=1.00 sl=590.40 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 586.75 | 591.27 | 591.87 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 595.85 | 592.14 | 592.01 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 588.10 | 591.91 | 592.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 584.75 | 590.48 | 591.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 586.55 | 586.05 | 588.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 10:15:00 | 586.55 | 586.05 | 588.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 586.55 | 586.05 | 588.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 586.55 | 586.05 | 588.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 588.75 | 586.59 | 588.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 13:15:00 | 586.00 | 586.54 | 587.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 605.30 | 590.81 | 588.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 605.30 | 590.81 | 588.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 632.65 | 603.55 | 595.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 642.60 | 644.50 | 631.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 09:15:00 | 652.70 | 644.50 | 631.96 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 10:45:00 | 652.60 | 647.63 | 635.64 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 637.00 | 645.28 | 637.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 637.00 | 645.28 | 637.55 | SL hit (close<ema400) qty=1.00 sl=637.55 alert=retest1 |

### Cycle 90 — SELL (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 10:15:00 | 630.85 | 635.22 | 635.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 628.95 | 632.41 | 633.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 14:15:00 | 633.30 | 632.58 | 633.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 14:15:00 | 633.30 | 632.58 | 633.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 633.30 | 632.58 | 633.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 633.30 | 632.58 | 633.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 630.85 | 632.24 | 633.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 624.00 | 632.24 | 633.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 616.00 | 628.99 | 631.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 613.40 | 628.99 | 631.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 582.73 | 593.71 | 605.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 583.40 | 581.89 | 591.35 | SL hit (close>ema200) qty=0.50 sl=581.89 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 596.00 | 589.46 | 589.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 598.25 | 593.81 | 591.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 592.80 | 594.28 | 592.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 12:00:00 | 592.80 | 594.28 | 592.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 590.55 | 593.54 | 592.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 590.55 | 593.54 | 592.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 590.95 | 593.02 | 592.04 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 585.30 | 590.38 | 590.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 582.95 | 589.01 | 590.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 583.30 | 582.87 | 586.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:00:00 | 583.30 | 582.87 | 586.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 585.60 | 583.41 | 586.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 585.60 | 583.41 | 586.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 586.40 | 584.01 | 586.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 586.40 | 584.01 | 586.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 584.60 | 584.13 | 586.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 582.85 | 584.13 | 586.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 14:15:00 | 586.80 | 584.35 | 585.78 | SL hit (close>static) qty=1.00 sl=586.40 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 594.65 | 586.66 | 586.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 598.65 | 589.06 | 587.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 588.15 | 590.85 | 589.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 15:15:00 | 588.15 | 590.85 | 589.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 588.15 | 590.85 | 589.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 588.25 | 590.85 | 589.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 592.90 | 591.26 | 589.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 588.95 | 591.26 | 589.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 588.55 | 590.72 | 589.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:30:00 | 585.85 | 590.72 | 589.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 589.80 | 590.53 | 589.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 589.00 | 590.53 | 589.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 594.85 | 591.40 | 589.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 600.70 | 594.54 | 592.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 599.60 | 595.59 | 592.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 599.75 | 597.47 | 595.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:45:00 | 600.00 | 597.85 | 595.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 594.75 | 597.23 | 595.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 594.75 | 597.23 | 595.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 595.00 | 596.78 | 595.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 595.65 | 596.78 | 595.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 594.40 | 596.31 | 595.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 595.00 | 596.31 | 595.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 587.55 | 594.56 | 594.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 586.00 | 592.84 | 593.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 564.80 | 564.03 | 569.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 564.80 | 564.03 | 569.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 566.85 | 565.02 | 568.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 567.50 | 565.02 | 568.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 568.95 | 565.80 | 568.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 563.60 | 566.93 | 568.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 576.95 | 569.38 | 569.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 576.95 | 569.38 | 569.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 586.00 | 573.70 | 571.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 583.60 | 585.32 | 579.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 583.60 | 585.32 | 579.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 586.00 | 585.46 | 580.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 581.55 | 585.46 | 580.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 581.30 | 583.71 | 581.76 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 574.40 | 580.50 | 580.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 13:15:00 | 572.05 | 578.81 | 579.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 572.70 | 572.11 | 575.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 572.70 | 572.11 | 575.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 572.75 | 572.23 | 575.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 572.75 | 572.23 | 575.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 568.30 | 570.14 | 573.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 565.55 | 567.96 | 571.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 565.90 | 567.96 | 571.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:45:00 | 565.65 | 567.59 | 570.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 573.00 | 571.17 | 571.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 573.00 | 571.17 | 571.16 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 570.35 | 571.11 | 571.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 564.65 | 569.29 | 570.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 556.35 | 555.25 | 559.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:30:00 | 555.15 | 555.25 | 559.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 557.55 | 555.71 | 558.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 557.00 | 555.71 | 558.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 558.50 | 556.27 | 558.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 558.50 | 556.27 | 558.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 556.75 | 556.36 | 558.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 555.70 | 556.36 | 558.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 555.45 | 556.34 | 558.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 556.35 | 555.53 | 557.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 561.20 | 557.06 | 557.86 | SL hit (close>static) qty=1.00 sl=558.95 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 561.15 | 558.60 | 558.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 14:15:00 | 570.10 | 560.90 | 559.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 563.30 | 564.63 | 563.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 563.30 | 564.63 | 563.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 563.30 | 564.63 | 563.05 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 559.95 | 561.94 | 562.14 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 599.95 | 569.54 | 565.57 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 569.35 | 573.77 | 574.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 564.10 | 571.83 | 573.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 534.95 | 534.58 | 540.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 12:45:00 | 534.10 | 534.58 | 540.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 536.00 | 534.02 | 538.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 537.80 | 534.02 | 538.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 538.30 | 535.53 | 537.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 538.30 | 535.53 | 537.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 544.50 | 537.33 | 538.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 544.50 | 537.33 | 538.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 544.00 | 538.66 | 538.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 540.15 | 538.66 | 538.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 535.55 | 538.07 | 538.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 535.00 | 538.07 | 538.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 534.75 | 537.56 | 538.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 541.80 | 538.34 | 538.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 541.80 | 538.34 | 538.25 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 535.55 | 538.51 | 538.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 530.85 | 536.98 | 537.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 534.60 | 534.54 | 536.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:15:00 | 530.80 | 534.54 | 536.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 524.05 | 528.27 | 531.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 523.50 | 527.62 | 530.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 523.20 | 525.36 | 527.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 530.25 | 526.34 | 528.07 | SL hit (close>ema400) qty=1.00 sl=528.07 alert=retest1 |

### Cycle 105 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 530.40 | 523.20 | 522.55 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 518.10 | 521.82 | 522.18 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 525.05 | 522.33 | 522.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 528.80 | 523.71 | 522.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 524.10 | 524.72 | 523.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 524.10 | 524.72 | 523.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 524.10 | 524.72 | 523.59 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 520.00 | 522.91 | 523.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 518.60 | 522.05 | 522.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 519.20 | 518.68 | 520.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:45:00 | 518.80 | 518.68 | 520.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 525.80 | 520.12 | 520.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 514.55 | 519.59 | 520.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 516.80 | 519.59 | 520.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 511.15 | 518.08 | 519.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:15:00 | 490.96 | 499.82 | 502.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 488.82 | 495.87 | 499.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 485.59 | 495.87 | 499.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 477.55 | 477.52 | 482.96 | SL hit (close>ema200) qty=0.50 sl=477.52 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 455.00 | 451.60 | 451.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 455.50 | 452.38 | 451.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 452.65 | 453.81 | 452.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 452.65 | 453.81 | 452.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 452.65 | 453.81 | 452.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 452.70 | 453.81 | 452.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 451.80 | 453.41 | 452.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 451.80 | 453.41 | 452.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 456.45 | 454.02 | 453.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 452.10 | 454.02 | 453.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 455.00 | 454.21 | 453.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 453.65 | 454.21 | 453.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 456.80 | 454.73 | 453.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 453.90 | 454.73 | 453.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 461.00 | 468.40 | 465.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 461.00 | 468.40 | 465.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 459.70 | 466.66 | 464.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 459.70 | 466.66 | 464.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 460.40 | 463.10 | 463.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 458.10 | 461.60 | 462.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 463.25 | 461.65 | 462.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 463.25 | 461.65 | 462.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 463.25 | 461.65 | 462.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 463.25 | 461.65 | 462.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 463.65 | 462.05 | 462.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 464.80 | 462.05 | 462.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 462.40 | 462.30 | 462.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:30:00 | 463.50 | 462.30 | 462.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 462.00 | 462.24 | 462.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 458.20 | 462.24 | 462.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 466.25 | 460.76 | 461.32 | SL hit (close>static) qty=1.00 sl=463.15 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 466.00 | 461.85 | 461.65 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 457.05 | 461.46 | 461.79 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 462.35 | 458.40 | 457.90 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 453.60 | 457.19 | 457.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 453.50 | 456.46 | 457.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 455.30 | 455.26 | 456.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 14:15:00 | 455.30 | 455.26 | 456.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 455.30 | 455.26 | 456.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 455.10 | 455.26 | 456.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 456.20 | 455.44 | 456.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 456.40 | 455.44 | 456.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 455.80 | 455.52 | 456.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 458.50 | 455.52 | 456.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 453.55 | 455.12 | 455.91 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 458.10 | 455.50 | 455.36 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 451.90 | 454.81 | 455.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 450.25 | 453.89 | 454.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 442.75 | 441.22 | 445.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 442.75 | 441.22 | 445.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 450.40 | 443.33 | 445.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 449.30 | 443.33 | 445.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 451.85 | 445.04 | 446.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 451.85 | 445.04 | 446.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 450.70 | 447.08 | 446.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 453.90 | 448.44 | 447.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 456.00 | 456.22 | 453.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 456.00 | 456.22 | 453.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 459.35 | 459.05 | 456.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 463.30 | 460.26 | 457.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 463.80 | 461.79 | 459.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 464.50 | 462.03 | 459.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 463.75 | 463.57 | 461.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 472.85 | 466.01 | 463.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:15:00 | 473.25 | 466.01 | 463.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 462.70 | 465.35 | 463.29 | SL hit (close<static) qty=1.00 sl=463.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 464.75 | 471.62 | 471.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 462.20 | 467.43 | 469.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 439.60 | 439.49 | 444.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 436.15 | 439.49 | 444.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 442.00 | 439.21 | 442.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 444.00 | 439.21 | 442.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 444.90 | 440.35 | 442.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 438.45 | 439.24 | 441.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 416.53 | 422.34 | 426.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 419.00 | 417.59 | 421.72 | SL hit (close>ema200) qty=0.50 sl=417.59 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 434.85 | 424.09 | 423.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 440.65 | 427.40 | 425.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 429.80 | 430.43 | 427.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 429.80 | 430.43 | 427.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 428.90 | 430.95 | 428.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 427.90 | 430.95 | 428.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 420.00 | 428.76 | 427.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 420.00 | 428.76 | 427.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 416.80 | 426.37 | 426.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 410.00 | 423.09 | 425.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 405.55 | 403.68 | 408.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-29 13:00:00 | 405.55 | 403.68 | 408.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 407.30 | 404.15 | 407.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 407.65 | 404.15 | 407.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 408.60 | 405.04 | 407.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 404.10 | 405.04 | 407.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 408.70 | 405.77 | 407.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 408.70 | 405.77 | 407.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 409.00 | 406.42 | 407.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 409.00 | 406.42 | 407.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 408.55 | 406.84 | 408.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 408.30 | 406.84 | 408.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 408.25 | 407.13 | 408.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 408.25 | 407.13 | 408.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 410.00 | 407.70 | 408.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 409.10 | 407.70 | 408.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 408.40 | 407.84 | 408.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:30:00 | 409.90 | 407.84 | 408.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 409.20 | 408.11 | 408.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 411.05 | 408.11 | 408.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 414.45 | 409.38 | 408.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 416.15 | 410.73 | 409.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 408.70 | 412.98 | 411.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 408.70 | 412.98 | 411.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 408.70 | 412.98 | 411.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 408.70 | 412.98 | 411.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 411.40 | 412.67 | 411.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:15:00 | 413.05 | 412.67 | 411.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 413.50 | 412.43 | 411.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:45:00 | 412.45 | 412.61 | 411.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 418.35 | 422.01 | 422.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 418.35 | 422.01 | 422.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 412.55 | 419.34 | 420.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 11:15:00 | 417.00 | 413.02 | 415.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 11:15:00 | 417.00 | 413.02 | 415.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 417.00 | 413.02 | 415.75 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 422.70 | 417.29 | 417.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 15:15:00 | 425.00 | 421.35 | 419.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 13:15:00 | 427.85 | 428.58 | 424.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 14:00:00 | 427.85 | 428.58 | 424.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 425.50 | 427.77 | 424.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 419.60 | 427.77 | 424.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 419.95 | 426.21 | 424.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 419.05 | 426.21 | 424.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 418.50 | 424.66 | 423.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 418.85 | 424.66 | 423.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 417.20 | 423.17 | 423.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 416.15 | 421.77 | 422.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 408.05 | 407.76 | 411.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:45:00 | 409.00 | 407.76 | 411.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 406.95 | 407.81 | 410.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 405.15 | 407.81 | 410.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 413.00 | 408.84 | 410.97 | SL hit (close>static) qty=1.00 sl=410.95 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 413.10 | 410.69 | 410.64 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 409.85 | 410.69 | 410.80 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 412.15 | 410.98 | 410.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 15:15:00 | 412.90 | 411.36 | 411.10 | Break + close above crossover candle high |

### Cycle 128 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 407.30 | 410.55 | 410.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 405.05 | 408.49 | 409.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 409.00 | 407.90 | 409.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 409.00 | 407.90 | 409.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 409.00 | 407.90 | 409.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 407.60 | 407.90 | 409.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 410.85 | 408.49 | 409.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 410.85 | 408.49 | 409.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 411.50 | 409.09 | 409.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 411.50 | 409.09 | 409.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 413.60 | 409.99 | 409.84 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 407.00 | 409.67 | 409.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 405.85 | 408.91 | 409.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 406.60 | 406.26 | 407.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 406.60 | 406.26 | 407.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 400.40 | 405.09 | 406.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:15:00 | 398.20 | 405.09 | 406.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 407.50 | 404.63 | 405.68 | SL hit (close>static) qty=1.00 sl=406.85 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 400.75 | 397.90 | 397.82 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 395.45 | 397.74 | 397.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 387.20 | 395.64 | 396.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 387.45 | 386.75 | 390.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 387.45 | 386.75 | 390.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 390.05 | 387.41 | 390.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 390.05 | 387.41 | 390.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 391.75 | 388.28 | 390.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 391.75 | 388.28 | 390.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 393.10 | 389.24 | 390.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:30:00 | 392.40 | 389.24 | 390.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 397.10 | 391.98 | 391.78 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 390.85 | 391.59 | 391.65 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 393.15 | 391.90 | 391.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 394.05 | 392.33 | 391.99 | Break + close above crossover candle high |

### Cycle 136 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 386.85 | 391.34 | 391.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 381.05 | 388.50 | 390.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 382.50 | 380.22 | 383.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:00:00 | 382.50 | 380.22 | 383.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 386.00 | 381.38 | 383.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 386.00 | 381.38 | 383.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 386.00 | 382.30 | 383.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 387.85 | 382.30 | 383.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 393.10 | 385.89 | 385.25 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 381.40 | 386.75 | 387.37 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 387.65 | 385.84 | 385.77 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 380.80 | 385.02 | 385.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 375.80 | 381.98 | 383.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 377.90 | 376.65 | 379.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 377.90 | 376.65 | 379.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 380.65 | 377.45 | 379.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 380.65 | 377.45 | 379.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 379.45 | 377.85 | 379.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 377.75 | 378.15 | 379.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 387.05 | 380.16 | 380.34 | SL hit (close>static) qty=1.00 sl=381.40 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 384.55 | 381.04 | 380.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 387.85 | 384.51 | 382.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 373.85 | 382.79 | 382.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 373.85 | 382.79 | 382.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 373.85 | 382.79 | 382.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 373.85 | 382.79 | 382.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 375.15 | 381.26 | 381.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 368.90 | 375.48 | 378.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 377.85 | 371.03 | 374.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 377.85 | 371.03 | 374.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 377.85 | 371.03 | 374.03 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 380.40 | 375.67 | 375.49 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 373.25 | 375.49 | 375.58 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 377.60 | 375.91 | 375.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 383.15 | 377.36 | 376.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 424.15 | 426.68 | 423.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 11:45:00 | 424.25 | 426.68 | 423.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 423.65 | 426.07 | 423.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:45:00 | 421.25 | 426.07 | 423.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 425.85 | 426.03 | 423.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 426.50 | 425.95 | 423.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 426.00 | 425.95 | 423.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 439.20 | 443.35 | 443.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 439.20 | 443.35 | 443.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 432.70 | 441.22 | 442.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 444.45 | 439.95 | 441.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 444.45 | 439.95 | 441.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 444.45 | 439.95 | 441.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 443.40 | 439.95 | 441.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 444.90 | 440.94 | 441.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 445.00 | 440.94 | 441.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 446.80 | 443.11 | 442.77 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 438.95 | 442.26 | 442.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 435.15 | 439.99 | 441.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 11:15:00 | 442.15 | 437.09 | 439.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 11:15:00 | 442.15 | 437.09 | 439.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 442.15 | 437.09 | 439.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 442.15 | 437.09 | 439.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 440.25 | 437.72 | 439.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 438.30 | 437.62 | 438.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 15:15:00 | 416.38 | 426.02 | 431.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 429.40 | 426.70 | 431.06 | SL hit (close>ema200) qty=0.50 sl=426.70 alert=retest2 |

### Cycle 149 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 449.65 | 433.96 | 433.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 455.00 | 440.75 | 436.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 491.40 | 496.02 | 486.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:15:00 | 489.50 | 496.02 | 486.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 487.00 | 493.01 | 486.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 487.15 | 493.01 | 486.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 486.25 | 490.90 | 486.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 486.25 | 490.90 | 486.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 485.10 | 489.74 | 486.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 485.10 | 489.74 | 486.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 485.90 | 488.97 | 486.47 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 10:15:00 | 401.00 | 2024-05-23 09:15:00 | 441.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 452.20 | 2024-06-04 11:15:00 | 433.91 | PARTIAL | 0.50 | 4.04% |
| SELL | retest2 | 2024-06-04 10:30:00 | 456.75 | 2024-06-04 12:15:00 | 429.59 | PARTIAL | 0.50 | 5.95% |
| SELL | retest2 | 2024-06-04 09:15:00 | 452.20 | 2024-06-04 13:15:00 | 457.55 | STOP_HIT | 0.50 | -1.18% |
| SELL | retest2 | 2024-06-04 10:30:00 | 456.75 | 2024-06-04 13:15:00 | 457.55 | STOP_HIT | 0.50 | -0.18% |
| BUY | retest2 | 2024-06-12 09:15:00 | 490.50 | 2024-06-12 09:15:00 | 482.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-06-12 09:45:00 | 489.90 | 2024-06-12 10:15:00 | 481.20 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-06-19 14:00:00 | 477.15 | 2024-06-20 13:15:00 | 489.55 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-06-19 15:15:00 | 470.00 | 2024-06-20 13:15:00 | 489.55 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2024-06-27 14:00:00 | 477.80 | 2024-07-01 09:15:00 | 493.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-06-28 09:30:00 | 481.25 | 2024-07-01 09:15:00 | 493.50 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-06-28 10:15:00 | 481.25 | 2024-07-01 09:15:00 | 493.50 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-06-28 11:00:00 | 481.00 | 2024-07-01 09:15:00 | 493.50 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-07-04 13:15:00 | 498.45 | 2024-07-04 14:15:00 | 493.40 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-07-15 14:45:00 | 502.30 | 2024-07-19 15:15:00 | 496.90 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-07-18 09:15:00 | 504.80 | 2024-07-19 15:15:00 | 496.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-07-18 09:45:00 | 502.85 | 2024-07-19 15:15:00 | 496.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-07-18 10:30:00 | 504.50 | 2024-07-19 15:15:00 | 496.90 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-31 09:15:00 | 524.75 | 2024-08-01 11:15:00 | 519.25 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-08-01 09:15:00 | 521.45 | 2024-08-01 11:15:00 | 519.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-08-06 15:00:00 | 492.20 | 2024-08-08 09:15:00 | 516.00 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2024-08-07 09:45:00 | 494.90 | 2024-08-08 09:15:00 | 516.00 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2024-08-12 10:45:00 | 533.00 | 2024-08-19 13:15:00 | 530.55 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-08-12 12:15:00 | 532.50 | 2024-08-19 13:15:00 | 530.55 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-08-13 13:00:00 | 533.40 | 2024-08-19 14:15:00 | 531.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-08-14 09:45:00 | 531.60 | 2024-08-19 14:15:00 | 531.30 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-08-16 12:30:00 | 537.95 | 2024-08-19 14:15:00 | 531.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-08-16 13:30:00 | 540.70 | 2024-08-19 14:15:00 | 531.30 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-08-23 11:30:00 | 542.00 | 2024-08-23 14:15:00 | 532.70 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-08-23 12:30:00 | 540.45 | 2024-08-23 14:15:00 | 532.70 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-08-23 13:15:00 | 542.00 | 2024-08-23 14:15:00 | 532.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-08-27 11:15:00 | 533.40 | 2024-08-28 11:15:00 | 539.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-28 11:00:00 | 537.35 | 2024-08-28 11:15:00 | 539.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-09-06 10:30:00 | 537.55 | 2024-09-10 14:15:00 | 591.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-06 13:45:00 | 536.80 | 2024-09-10 14:15:00 | 590.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 10:30:00 | 533.90 | 2024-09-10 14:15:00 | 587.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 13:30:00 | 536.15 | 2024-09-10 14:15:00 | 589.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-16 13:15:00 | 590.05 | 2024-09-17 10:15:00 | 581.85 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-09-17 11:45:00 | 608.95 | 2024-09-19 10:15:00 | 584.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-09-24 11:45:00 | 561.05 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-09-24 15:15:00 | 560.50 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-09-25 09:45:00 | 559.65 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-09-27 14:00:00 | 560.85 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-09-30 12:15:00 | 551.90 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2024-09-30 14:00:00 | 553.20 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-09-30 14:30:00 | 553.75 | 2024-10-01 09:15:00 | 569.85 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-10-07 09:30:00 | 538.50 | 2024-10-09 09:15:00 | 548.50 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-10-25 09:45:00 | 506.55 | 2024-10-28 15:15:00 | 519.70 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-11-05 09:45:00 | 564.85 | 2024-11-08 09:15:00 | 560.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-11-05 10:30:00 | 564.90 | 2024-11-08 10:15:00 | 554.40 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-11-05 12:00:00 | 567.10 | 2024-11-08 10:15:00 | 554.40 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-11-05 14:00:00 | 564.65 | 2024-11-08 10:15:00 | 554.40 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-11-07 11:00:00 | 592.05 | 2024-11-08 10:15:00 | 554.40 | STOP_HIT | 1.00 | -6.36% |
| SELL | retest2 | 2024-11-14 14:30:00 | 521.25 | 2024-11-25 09:15:00 | 540.50 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-11-14 15:00:00 | 519.55 | 2024-11-25 09:15:00 | 540.50 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-11-18 09:15:00 | 515.60 | 2024-11-25 09:15:00 | 540.50 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2024-11-19 10:30:00 | 521.50 | 2024-11-25 09:15:00 | 540.50 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-11-19 15:15:00 | 515.00 | 2024-11-25 09:15:00 | 540.50 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2024-11-21 10:30:00 | 515.70 | 2024-11-25 09:15:00 | 540.50 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-11-28 09:15:00 | 545.95 | 2024-11-28 14:15:00 | 534.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-12-11 11:45:00 | 536.45 | 2024-12-17 09:15:00 | 541.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-23 09:15:00 | 518.35 | 2024-12-24 14:15:00 | 526.10 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-01-08 09:15:00 | 515.95 | 2025-01-10 09:15:00 | 490.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 515.95 | 2025-01-13 11:15:00 | 464.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 14:30:00 | 482.50 | 2025-01-24 12:15:00 | 490.50 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-01-24 09:30:00 | 481.00 | 2025-01-24 12:15:00 | 490.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-01-27 09:15:00 | 474.45 | 2025-01-27 14:15:00 | 503.45 | STOP_HIT | 1.00 | -6.11% |
| BUY | retest2 | 2025-02-01 09:15:00 | 513.40 | 2025-02-01 15:15:00 | 508.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-02-01 10:15:00 | 518.05 | 2025-02-01 15:15:00 | 508.10 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-02-01 13:00:00 | 513.50 | 2025-02-01 15:15:00 | 508.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-02-01 13:45:00 | 513.40 | 2025-02-01 15:15:00 | 508.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-02-04 10:30:00 | 524.45 | 2025-02-10 09:15:00 | 513.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-02-05 15:00:00 | 525.00 | 2025-02-10 09:15:00 | 513.50 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest1 | 2025-02-27 14:15:00 | 472.15 | 2025-03-03 09:15:00 | 448.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-27 14:15:00 | 472.15 | 2025-03-03 13:15:00 | 458.50 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest1 | 2025-02-28 09:15:00 | 468.40 | 2025-03-04 09:15:00 | 464.85 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-03-05 14:30:00 | 478.25 | 2025-03-11 13:15:00 | 488.25 | STOP_HIT | 1.00 | 2.09% |
| BUY | retest2 | 2025-03-19 09:15:00 | 494.20 | 2025-03-28 12:15:00 | 543.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-29 11:30:00 | 508.20 | 2025-05-02 09:15:00 | 521.50 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-04-30 14:30:00 | 507.00 | 2025-05-02 09:15:00 | 521.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-04-30 15:00:00 | 508.35 | 2025-05-02 09:15:00 | 521.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-05-16 09:15:00 | 525.75 | 2025-05-22 12:15:00 | 533.35 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-05-16 11:30:00 | 523.90 | 2025-05-22 12:15:00 | 533.35 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-05-26 09:30:00 | 525.30 | 2025-05-27 15:15:00 | 501.46 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2025-05-26 09:30:00 | 525.30 | 2025-05-28 10:15:00 | 513.00 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2025-05-26 11:00:00 | 527.85 | 2025-05-29 09:15:00 | 499.03 | PARTIAL | 0.50 | 5.46% |
| SELL | retest2 | 2025-05-26 11:00:00 | 527.85 | 2025-05-30 09:15:00 | 510.65 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2025-05-27 09:15:00 | 514.10 | 2025-06-05 13:15:00 | 513.25 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-06-26 11:15:00 | 516.15 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-06-26 13:00:00 | 516.25 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-06-26 14:30:00 | 516.40 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-06-27 09:15:00 | 516.15 | 2025-07-01 13:15:00 | 516.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-07-07 12:15:00 | 545.10 | 2025-07-10 09:15:00 | 537.45 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-09 13:45:00 | 543.85 | 2025-07-10 09:15:00 | 537.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-09 14:30:00 | 545.25 | 2025-07-10 09:15:00 | 537.45 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-22 11:30:00 | 595.60 | 2025-07-23 12:15:00 | 589.25 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-23 09:45:00 | 595.00 | 2025-07-23 12:15:00 | 589.25 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-28 13:15:00 | 586.00 | 2025-07-29 13:15:00 | 605.30 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest1 | 2025-08-01 09:15:00 | 652.70 | 2025-08-01 13:15:00 | 637.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest1 | 2025-08-01 10:45:00 | 652.60 | 2025-08-01 13:15:00 | 637.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-08-04 09:15:00 | 649.40 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-08-04 12:30:00 | 637.30 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-08-04 13:15:00 | 646.00 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-05 10:15:00 | 635.75 | 2025-08-05 10:15:00 | 630.85 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-08-06 10:15:00 | 613.40 | 2025-08-08 09:15:00 | 582.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 10:15:00 | 613.40 | 2025-08-11 10:15:00 | 583.40 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-08-19 13:15:00 | 582.85 | 2025-08-19 14:15:00 | 586.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-08-19 15:15:00 | 582.50 | 2025-08-20 09:15:00 | 587.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-22 09:45:00 | 600.70 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-08-22 12:15:00 | 599.60 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-08-25 09:30:00 | 599.75 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-08-25 10:45:00 | 600.00 | 2025-08-25 14:15:00 | 587.55 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-02 09:15:00 | 563.60 | 2025-09-02 10:15:00 | 576.95 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-09-09 13:30:00 | 565.55 | 2025-09-10 15:15:00 | 573.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-09 14:00:00 | 565.90 | 2025-09-10 15:15:00 | 573.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-09 14:45:00 | 565.65 | 2025-09-10 15:15:00 | 573.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-16 13:15:00 | 555.70 | 2025-09-17 11:15:00 | 561.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-16 13:45:00 | 555.45 | 2025-09-17 11:15:00 | 561.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-17 09:45:00 | 556.35 | 2025-09-17 11:15:00 | 561.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-03 11:15:00 | 535.00 | 2025-10-06 09:15:00 | 541.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-03 12:15:00 | 534.75 | 2025-10-06 09:15:00 | 541.80 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest1 | 2025-10-08 09:15:00 | 530.80 | 2025-10-10 11:15:00 | 530.25 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-10-09 10:30:00 | 523.50 | 2025-10-17 11:15:00 | 525.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-10 11:00:00 | 523.20 | 2025-10-17 11:15:00 | 525.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-13 15:15:00 | 523.00 | 2025-10-17 11:15:00 | 525.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-14 09:45:00 | 523.25 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-16 15:15:00 | 516.00 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-10-17 09:30:00 | 517.15 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-10-17 10:00:00 | 516.30 | 2025-10-17 13:15:00 | 530.40 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-10-27 12:45:00 | 514.55 | 2025-11-06 09:15:00 | 490.96 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2025-10-27 13:15:00 | 516.80 | 2025-11-06 11:15:00 | 488.82 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-10-28 09:15:00 | 511.15 | 2025-11-06 11:15:00 | 485.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 12:45:00 | 514.55 | 2025-11-10 11:15:00 | 477.55 | STOP_HIT | 0.50 | 7.19% |
| SELL | retest2 | 2025-10-27 13:15:00 | 516.80 | 2025-11-10 11:15:00 | 477.55 | STOP_HIT | 0.50 | 7.59% |
| SELL | retest2 | 2025-10-28 09:15:00 | 511.15 | 2025-11-10 11:15:00 | 477.55 | STOP_HIT | 0.50 | 6.57% |
| SELL | retest2 | 2025-12-04 09:15:00 | 458.20 | 2025-12-04 14:15:00 | 466.25 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-05 09:15:00 | 458.05 | 2025-12-05 11:15:00 | 466.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-26 13:30:00 | 463.30 | 2025-12-30 14:15:00 | 462.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-12-29 09:30:00 | 463.80 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-12-29 11:15:00 | 464.50 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-12-30 10:30:00 | 463.75 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-12-30 14:15:00 | 473.25 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-01-05 10:15:00 | 475.45 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-01-06 14:45:00 | 474.20 | 2026-01-07 09:15:00 | 464.75 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-01-14 12:30:00 | 438.45 | 2026-01-21 09:15:00 | 416.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:30:00 | 438.45 | 2026-01-21 15:15:00 | 419.00 | STOP_HIT | 0.50 | 4.44% |
| BUY | retest2 | 2026-02-02 11:15:00 | 413.05 | 2026-02-05 15:15:00 | 418.35 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2026-02-02 13:30:00 | 413.50 | 2026-02-05 15:15:00 | 418.35 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2026-02-02 14:45:00 | 412.45 | 2026-02-05 15:15:00 | 418.35 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2026-02-17 09:15:00 | 405.15 | 2026-02-17 09:15:00 | 413.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-25 13:15:00 | 398.20 | 2026-02-26 10:15:00 | 407.50 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-02-27 10:45:00 | 399.75 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-02-27 12:30:00 | 399.30 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-27 14:00:00 | 399.60 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-03-05 10:15:00 | 393.40 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-05 14:15:00 | 393.70 | 2026-03-06 11:15:00 | 400.75 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-03-24 14:30:00 | 377.75 | 2026-03-25 09:15:00 | 387.05 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-04-16 14:30:00 | 426.50 | 2026-04-24 12:15:00 | 439.20 | STOP_HIT | 1.00 | 2.98% |
| BUY | retest2 | 2026-04-16 15:15:00 | 426.00 | 2026-04-24 12:15:00 | 439.20 | STOP_HIT | 1.00 | 3.10% |
| SELL | retest2 | 2026-04-29 13:30:00 | 438.30 | 2026-04-30 15:15:00 | 416.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 13:30:00 | 438.30 | 2026-05-04 09:15:00 | 429.40 | STOP_HIT | 0.50 | 2.03% |
