# Saregama India Ltd (SAREGAMA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 156 |
| ALERT1 | 100 |
| ALERT2 | 98 |
| ALERT2_SKIP | 59 |
| ALERT3 | 258 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 113 |
| PARTIAL | 16 |
| TARGET_HIT | 6 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 50 / 83
- **Target hits / Stop hits / Partials:** 6 / 111 / 16
- **Avg / median % per leg:** 0.40% / -0.86%
- **Sum % (uncompounded):** 53.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 14 | 31.8% | 2 | 42 | 0 | -0.74% | -32.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.51% | -5.0% |
| BUY @ 3rd Alert (retest2) | 42 | 14 | 33.3% | 2 | 40 | 0 | -0.66% | -27.6% |
| SELL (all) | 89 | 36 | 40.4% | 4 | 69 | 16 | 0.97% | 85.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.05% | -3.1% |
| SELL @ 3rd Alert (retest2) | 86 | 36 | 41.9% | 4 | 66 | 16 | 1.04% | 89.1% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.63% | -8.2% |
| retest2 (combined) | 128 | 50 | 39.1% | 6 | 106 | 16 | 0.48% | 61.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 426.65 | 430.58 | 430.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 12:15:00 | 424.35 | 427.57 | 428.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 10:15:00 | 426.90 | 426.11 | 427.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 426.90 | 426.11 | 427.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 426.90 | 426.11 | 427.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 426.90 | 426.11 | 427.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 426.90 | 426.27 | 427.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:45:00 | 429.65 | 426.27 | 427.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 426.85 | 426.38 | 427.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:45:00 | 427.70 | 426.38 | 427.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 425.35 | 426.18 | 427.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 426.75 | 426.18 | 427.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 425.05 | 425.95 | 426.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 425.05 | 425.95 | 426.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 427.50 | 426.26 | 426.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 428.35 | 426.26 | 426.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 434.25 | 427.86 | 427.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 11:15:00 | 436.45 | 430.15 | 428.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 432.10 | 434.20 | 431.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 12:15:00 | 432.10 | 434.20 | 431.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 432.10 | 434.20 | 431.99 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 425.00 | 429.75 | 430.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 421.95 | 426.33 | 428.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 431.00 | 417.23 | 421.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 11:15:00 | 431.00 | 417.23 | 421.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 431.00 | 417.23 | 421.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 431.00 | 417.23 | 421.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 460.15 | 425.82 | 424.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 465.25 | 438.45 | 430.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 480.00 | 483.32 | 470.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 480.00 | 483.32 | 470.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 474.90 | 479.14 | 473.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 474.40 | 479.14 | 473.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 471.95 | 477.70 | 472.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 471.95 | 477.70 | 472.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 474.10 | 476.98 | 473.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:30:00 | 469.55 | 476.98 | 473.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 473.00 | 476.19 | 473.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 483.30 | 476.19 | 473.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-03 09:15:00 | 531.63 | 505.59 | 499.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 491.65 | 504.67 | 505.99 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 520.00 | 506.94 | 505.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 528.30 | 513.34 | 508.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 10:15:00 | 557.15 | 557.36 | 547.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 10:30:00 | 557.80 | 557.36 | 547.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 559.00 | 560.87 | 555.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 559.75 | 560.87 | 555.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 545.30 | 557.56 | 554.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 545.30 | 557.56 | 554.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 550.85 | 556.22 | 554.34 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 09:15:00 | 545.55 | 553.08 | 553.18 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 554.65 | 551.16 | 551.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 10:15:00 | 568.30 | 558.63 | 556.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 555.15 | 560.51 | 558.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 14:15:00 | 555.15 | 560.51 | 558.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 555.15 | 560.51 | 558.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 555.15 | 560.51 | 558.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 557.00 | 559.81 | 558.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 552.70 | 559.81 | 558.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 552.20 | 556.63 | 556.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 13:15:00 | 547.95 | 554.27 | 555.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 558.20 | 552.44 | 554.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 558.20 | 552.44 | 554.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 558.20 | 552.44 | 554.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:45:00 | 561.55 | 552.44 | 554.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 556.25 | 553.20 | 554.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 556.25 | 553.20 | 554.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 555.75 | 553.71 | 554.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:00:00 | 550.50 | 553.07 | 554.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:30:00 | 552.80 | 552.13 | 553.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 557.80 | 550.94 | 552.05 | SL hit (close>static) qty=1.00 sl=556.70 alert=retest2 |

### Cycle 10 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 556.70 | 553.05 | 552.87 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 553.75 | 555.65 | 555.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 547.85 | 553.58 | 554.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 10:15:00 | 557.60 | 549.86 | 551.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 10:15:00 | 557.60 | 549.86 | 551.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 557.60 | 549.86 | 551.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 557.60 | 549.86 | 551.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 558.10 | 551.51 | 552.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:45:00 | 554.50 | 551.89 | 552.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:15:00 | 526.77 | 538.96 | 542.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 524.20 | 521.87 | 526.54 | SL hit (close>ema200) qty=0.50 sl=521.87 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 530.35 | 527.33 | 526.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 15:15:00 | 534.00 | 529.12 | 527.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 12:15:00 | 534.35 | 537.18 | 534.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 534.35 | 537.18 | 534.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 534.35 | 537.18 | 534.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 534.35 | 537.18 | 534.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 532.15 | 536.18 | 534.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:30:00 | 542.70 | 537.42 | 535.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 529.75 | 534.40 | 534.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 529.75 | 534.40 | 534.53 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 545.55 | 532.84 | 531.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 563.35 | 543.18 | 539.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 12:15:00 | 545.25 | 545.98 | 542.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 13:00:00 | 545.25 | 545.98 | 542.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 545.00 | 545.77 | 542.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:30:00 | 542.80 | 545.77 | 542.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 554.40 | 549.64 | 545.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 566.40 | 557.04 | 553.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 562.45 | 558.04 | 553.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 547.00 | 552.71 | 553.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 547.00 | 552.71 | 553.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 542.00 | 548.68 | 551.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 09:15:00 | 547.15 | 545.44 | 548.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 547.15 | 545.44 | 548.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 547.15 | 545.44 | 548.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 547.15 | 545.44 | 548.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 554.50 | 547.25 | 549.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 12:00:00 | 535.00 | 544.80 | 548.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 508.25 | 526.26 | 536.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 504.40 | 502.03 | 516.48 | SL hit (close>ema200) qty=0.50 sl=502.03 alert=retest2 |

### Cycle 16 — BUY (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 13:15:00 | 497.50 | 492.12 | 491.84 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 486.35 | 490.96 | 491.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 15:15:00 | 486.00 | 489.97 | 490.86 | Break + close below crossover candle low |

### Cycle 18 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 499.85 | 491.95 | 491.68 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 488.20 | 493.34 | 493.50 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 496.35 | 493.89 | 493.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 13:15:00 | 497.40 | 494.59 | 494.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 14:15:00 | 493.85 | 494.44 | 494.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 14:15:00 | 493.85 | 494.44 | 494.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 493.85 | 494.44 | 494.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 493.10 | 494.44 | 494.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 491.00 | 493.75 | 493.76 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 498.75 | 494.75 | 494.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 11:15:00 | 508.75 | 499.38 | 496.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 524.05 | 527.69 | 522.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:00:00 | 524.05 | 527.69 | 522.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 532.00 | 528.55 | 523.66 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 517.30 | 524.37 | 524.42 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 528.00 | 525.01 | 524.63 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 509.15 | 522.34 | 523.73 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 526.45 | 519.17 | 518.75 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 514.35 | 519.32 | 519.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 12:15:00 | 512.60 | 517.36 | 518.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 10:15:00 | 511.10 | 508.84 | 511.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 10:15:00 | 511.10 | 508.84 | 511.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 511.10 | 508.84 | 511.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 511.10 | 508.84 | 511.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 508.70 | 508.81 | 511.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:00:00 | 506.85 | 508.42 | 510.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 11:15:00 | 507.80 | 502.53 | 502.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 11:15:00 | 507.80 | 502.53 | 502.46 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 493.55 | 502.02 | 502.50 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 507.75 | 501.64 | 501.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 510.95 | 503.50 | 502.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 519.60 | 520.18 | 514.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:45:00 | 518.85 | 520.18 | 514.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 518.15 | 520.06 | 515.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 518.15 | 520.06 | 515.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 514.60 | 518.97 | 515.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:00:00 | 514.60 | 518.97 | 515.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 513.05 | 517.79 | 515.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 513.65 | 517.79 | 515.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 511.30 | 516.08 | 514.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:45:00 | 516.00 | 516.21 | 514.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 15:15:00 | 517.70 | 520.49 | 520.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 15:15:00 | 517.70 | 520.49 | 520.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 511.80 | 518.17 | 519.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 516.45 | 515.73 | 517.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 516.45 | 515.73 | 517.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 516.45 | 515.73 | 517.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 510.50 | 515.73 | 517.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 507.40 | 514.97 | 517.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:15:00 | 510.15 | 514.76 | 516.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 14:00:00 | 510.60 | 513.92 | 516.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 514.60 | 513.97 | 515.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:45:00 | 509.65 | 512.42 | 514.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 507.40 | 512.11 | 514.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 520.35 | 515.85 | 515.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 520.35 | 515.85 | 515.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 525.55 | 519.43 | 517.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 15:15:00 | 585.00 | 586.87 | 569.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 580.35 | 586.87 | 569.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 572.45 | 578.38 | 571.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:45:00 | 571.45 | 578.38 | 571.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 566.05 | 575.91 | 570.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 566.05 | 575.91 | 570.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 564.70 | 573.67 | 570.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 569.80 | 573.67 | 570.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 632.30 | 645.08 | 633.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 627.75 | 645.08 | 633.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 628.60 | 641.79 | 632.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 636.00 | 639.89 | 632.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 13:15:00 | 621.80 | 633.61 | 631.00 | SL hit (close<static) qty=1.00 sl=622.10 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 608.90 | 628.67 | 628.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 595.00 | 618.71 | 624.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 636.50 | 603.37 | 610.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 636.50 | 603.37 | 610.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 636.50 | 603.37 | 610.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 636.50 | 603.37 | 610.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 633.75 | 609.45 | 613.01 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 642.40 | 619.98 | 617.42 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 606.60 | 619.82 | 621.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 605.40 | 616.94 | 620.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 13:15:00 | 547.75 | 547.28 | 556.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 14:00:00 | 547.75 | 547.28 | 556.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 565.30 | 550.88 | 557.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 566.20 | 550.88 | 557.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 568.00 | 554.31 | 558.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 557.35 | 554.31 | 558.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 502.30 | 506.49 | 515.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:30:00 | 505.50 | 506.49 | 515.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 485.95 | 485.73 | 493.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 474.10 | 485.73 | 493.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 500.00 | 486.02 | 488.86 | SL hit (close>static) qty=1.00 sl=493.70 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 498.15 | 491.24 | 490.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 503.75 | 497.05 | 493.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 525.90 | 531.25 | 519.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 525.90 | 531.25 | 519.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 510.65 | 527.13 | 519.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 508.75 | 527.13 | 519.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 511.25 | 523.96 | 518.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 510.75 | 523.96 | 518.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 523.65 | 527.26 | 522.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:00:00 | 523.65 | 527.26 | 522.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 508.55 | 523.52 | 521.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:00:00 | 508.55 | 523.52 | 521.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 510.90 | 520.99 | 520.54 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 14:15:00 | 515.30 | 519.86 | 520.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 09:15:00 | 505.25 | 515.28 | 517.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 510.00 | 504.51 | 509.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 510.00 | 504.51 | 509.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 510.00 | 504.51 | 509.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:45:00 | 511.05 | 504.51 | 509.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 515.15 | 506.64 | 510.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 515.15 | 506.64 | 510.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 512.35 | 507.78 | 510.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 513.10 | 507.78 | 510.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 514.10 | 510.74 | 511.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 15:15:00 | 510.00 | 510.74 | 511.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 484.50 | 495.72 | 502.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 09:15:00 | 459.00 | 470.64 | 481.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 13:15:00 | 489.00 | 478.09 | 477.45 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 11:15:00 | 474.45 | 477.78 | 477.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 467.90 | 474.35 | 476.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 473.70 | 471.87 | 474.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 10:15:00 | 473.70 | 471.87 | 474.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 473.70 | 471.87 | 474.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 469.20 | 473.61 | 474.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 474.45 | 463.06 | 462.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 474.45 | 463.06 | 462.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 12:15:00 | 478.90 | 466.23 | 463.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 482.90 | 484.64 | 477.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 15:15:00 | 482.90 | 484.64 | 477.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 482.90 | 484.64 | 477.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:15:00 | 493.55 | 485.41 | 478.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 493.25 | 495.64 | 491.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:30:00 | 492.25 | 494.77 | 491.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 513.00 | 517.98 | 518.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 513.00 | 517.98 | 518.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 11:15:00 | 509.15 | 515.31 | 517.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 508.45 | 505.73 | 509.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 508.45 | 505.73 | 509.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 508.45 | 505.73 | 509.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 508.45 | 505.73 | 509.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 510.90 | 506.77 | 509.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 513.80 | 506.77 | 509.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 507.55 | 506.92 | 509.41 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 516.05 | 511.44 | 510.90 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 501.70 | 510.47 | 510.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 500.40 | 508.46 | 509.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 498.55 | 498.29 | 501.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 498.55 | 498.29 | 501.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 498.55 | 498.29 | 501.37 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 12:15:00 | 502.50 | 500.28 | 500.13 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 497.55 | 500.02 | 500.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 492.55 | 498.24 | 499.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 12:15:00 | 500.70 | 498.26 | 498.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 12:15:00 | 500.70 | 498.26 | 498.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 500.70 | 498.26 | 498.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:00:00 | 500.70 | 498.26 | 498.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 499.40 | 498.49 | 498.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:30:00 | 497.15 | 498.63 | 499.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 15:15:00 | 501.80 | 499.26 | 499.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 15:15:00 | 501.80 | 499.26 | 499.25 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 495.70 | 498.55 | 498.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 490.80 | 496.21 | 497.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 14:15:00 | 497.00 | 495.68 | 496.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 497.00 | 495.68 | 496.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 497.00 | 495.68 | 496.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 497.00 | 495.68 | 496.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 497.50 | 496.04 | 497.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 488.10 | 496.04 | 497.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 14:15:00 | 463.69 | 470.41 | 476.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 470.90 | 470.16 | 475.13 | SL hit (close>ema200) qty=0.50 sl=470.16 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 479.75 | 468.63 | 467.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 493.50 | 480.79 | 476.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 15:15:00 | 570.40 | 572.41 | 554.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-08 09:15:00 | 560.00 | 572.41 | 554.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 552.50 | 566.63 | 555.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 554.05 | 566.63 | 555.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 547.30 | 562.76 | 554.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 549.15 | 562.76 | 554.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 549.90 | 554.01 | 552.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 586.80 | 554.01 | 552.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 14:00:00 | 554.75 | 555.93 | 554.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 14:45:00 | 553.95 | 555.04 | 554.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 531.70 | 550.64 | 552.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 531.70 | 550.64 | 552.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 522.40 | 532.92 | 540.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 514.50 | 514.41 | 524.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:00:00 | 514.50 | 514.41 | 524.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 542.00 | 519.93 | 525.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 542.00 | 519.93 | 525.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 539.15 | 523.77 | 527.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 535.15 | 523.77 | 527.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 532.45 | 529.48 | 529.13 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 519.70 | 527.69 | 528.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 10:15:00 | 517.05 | 525.56 | 527.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 526.85 | 520.66 | 523.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 526.85 | 520.66 | 523.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 526.85 | 520.66 | 523.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 526.85 | 520.66 | 523.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 524.85 | 521.50 | 523.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 11:45:00 | 518.35 | 521.14 | 523.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 541.55 | 524.40 | 523.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 541.55 | 524.40 | 523.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 546.70 | 533.59 | 528.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 540.85 | 541.04 | 534.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 540.85 | 541.04 | 534.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 534.85 | 539.80 | 534.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 534.85 | 539.80 | 534.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 537.00 | 539.24 | 534.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 537.35 | 539.24 | 534.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 534.80 | 538.35 | 534.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 535.65 | 538.35 | 534.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 536.65 | 538.01 | 534.76 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 525.50 | 532.62 | 533.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 523.20 | 530.73 | 532.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 528.25 | 528.22 | 530.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 528.25 | 528.22 | 530.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 528.25 | 528.22 | 530.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 528.25 | 528.22 | 530.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 530.30 | 528.64 | 530.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 524.00 | 528.64 | 530.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:45:00 | 524.20 | 528.24 | 530.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 525.25 | 528.24 | 530.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 497.80 | 506.92 | 515.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 497.99 | 506.92 | 515.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 498.99 | 506.92 | 515.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 10:15:00 | 471.60 | 499.51 | 511.78 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 54 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 489.40 | 481.19 | 480.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 502.20 | 485.85 | 482.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 14:15:00 | 522.80 | 523.82 | 513.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:15:00 | 523.60 | 523.82 | 513.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 537.20 | 528.38 | 521.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 539.20 | 528.38 | 521.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 529.00 | 530.85 | 525.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 526.55 | 530.85 | 525.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 525.70 | 529.23 | 526.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 525.70 | 529.23 | 526.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 528.00 | 528.98 | 526.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 523.70 | 528.98 | 526.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 527.10 | 528.61 | 526.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 527.10 | 528.61 | 526.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 531.15 | 529.11 | 526.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 541.85 | 528.85 | 527.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 514.00 | 525.88 | 525.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 514.00 | 525.88 | 525.96 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 14:15:00 | 540.45 | 525.02 | 524.58 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 509.50 | 521.85 | 523.21 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 525.70 | 519.98 | 519.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 530.00 | 521.92 | 520.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 524.05 | 524.97 | 522.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 13:15:00 | 524.05 | 524.97 | 522.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 524.05 | 524.97 | 522.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:00:00 | 524.05 | 524.97 | 522.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 519.40 | 523.85 | 522.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:30:00 | 520.65 | 523.85 | 522.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 514.90 | 522.06 | 521.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:15:00 | 509.15 | 522.06 | 521.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 511.00 | 519.85 | 520.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 499.55 | 512.33 | 516.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 494.65 | 477.44 | 483.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 494.65 | 477.44 | 483.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 494.65 | 477.44 | 483.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 495.65 | 477.44 | 483.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 495.40 | 481.03 | 484.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 495.50 | 481.03 | 484.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 499.00 | 488.61 | 487.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 501.95 | 492.78 | 489.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 499.20 | 499.22 | 494.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 499.20 | 499.22 | 494.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 497.70 | 498.23 | 495.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:00:00 | 500.25 | 498.63 | 495.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:00:00 | 500.80 | 498.68 | 496.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 10:00:00 | 500.55 | 497.81 | 497.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 11:00:00 | 500.50 | 498.35 | 497.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 497.95 | 498.27 | 497.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 497.95 | 498.27 | 497.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 498.05 | 498.23 | 497.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:30:00 | 497.50 | 498.23 | 497.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 498.90 | 498.36 | 497.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:15:00 | 496.95 | 498.36 | 497.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 498.05 | 498.30 | 497.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:15:00 | 495.50 | 498.30 | 497.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 495.50 | 497.74 | 497.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 492.15 | 497.74 | 497.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 488.80 | 495.95 | 496.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 488.80 | 495.95 | 496.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 481.15 | 491.21 | 494.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 445.10 | 440.68 | 455.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 445.10 | 440.68 | 455.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 451.20 | 443.52 | 454.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 464.95 | 447.81 | 455.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 471.10 | 452.47 | 456.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 471.10 | 452.47 | 456.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 471.15 | 459.17 | 459.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 476.85 | 468.74 | 465.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 486.20 | 488.24 | 481.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 486.20 | 488.24 | 481.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 486.20 | 488.24 | 481.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 486.20 | 488.24 | 481.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 481.00 | 486.59 | 483.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 477.45 | 486.59 | 483.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 484.80 | 486.23 | 484.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 491.20 | 487.41 | 484.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:15:00 | 487.05 | 487.84 | 485.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:15:00 | 488.00 | 486.93 | 485.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 477.50 | 483.91 | 484.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 477.50 | 483.91 | 484.42 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 489.00 | 484.02 | 483.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 14:15:00 | 496.20 | 488.50 | 486.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 14:15:00 | 487.65 | 492.86 | 490.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 487.65 | 492.86 | 490.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 487.65 | 492.86 | 490.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 487.65 | 492.86 | 490.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 487.00 | 491.69 | 489.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 492.15 | 491.69 | 489.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 507.00 | 511.59 | 511.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 507.00 | 511.59 | 511.72 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 516.75 | 512.19 | 511.71 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 500.00 | 510.63 | 511.29 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 518.10 | 512.13 | 511.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 521.90 | 517.16 | 515.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 12:15:00 | 516.55 | 519.03 | 516.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 516.55 | 519.03 | 516.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 516.55 | 519.03 | 516.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 516.55 | 519.03 | 516.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 514.85 | 518.19 | 516.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:45:00 | 513.95 | 518.19 | 516.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 514.65 | 517.48 | 516.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 514.65 | 517.48 | 516.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 513.00 | 516.59 | 516.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 514.30 | 516.59 | 516.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 513.05 | 515.88 | 515.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 506.00 | 512.21 | 514.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 502.00 | 501.01 | 505.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-04 15:00:00 | 502.00 | 501.01 | 505.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 505.00 | 488.53 | 494.23 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 499.95 | 496.70 | 496.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 517.80 | 502.42 | 499.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 15:15:00 | 521.35 | 522.64 | 514.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 09:15:00 | 524.90 | 522.64 | 514.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 526.80 | 523.47 | 515.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 13:00:00 | 533.95 | 525.68 | 518.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 546.20 | 555.97 | 557.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 546.20 | 555.97 | 557.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 14:15:00 | 544.55 | 552.55 | 555.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 560.55 | 548.46 | 551.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 560.55 | 548.46 | 551.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 560.55 | 548.46 | 551.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 560.55 | 548.46 | 551.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 545.70 | 547.91 | 550.75 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 556.85 | 551.59 | 551.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 15:15:00 | 560.90 | 555.80 | 553.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 10:15:00 | 548.95 | 554.47 | 553.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 10:15:00 | 548.95 | 554.47 | 553.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 548.95 | 554.47 | 553.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 548.95 | 554.47 | 553.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 545.05 | 552.59 | 552.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 12:15:00 | 540.85 | 550.24 | 551.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 532.30 | 530.78 | 537.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 15:00:00 | 532.30 | 530.78 | 537.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 535.45 | 532.32 | 537.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 537.50 | 532.32 | 537.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 529.00 | 531.66 | 536.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:30:00 | 526.60 | 530.31 | 535.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 537.45 | 532.54 | 535.52 | SL hit (close>static) qty=1.00 sl=537.05 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 536.20 | 533.17 | 532.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 13:15:00 | 545.00 | 536.03 | 534.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 533.05 | 538.33 | 536.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 533.05 | 538.33 | 536.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 533.05 | 538.33 | 536.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 533.05 | 538.33 | 536.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 530.00 | 536.67 | 535.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:00:00 | 530.00 | 536.67 | 535.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 536.10 | 535.50 | 535.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:30:00 | 533.65 | 535.50 | 535.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 535.05 | 535.41 | 535.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:30:00 | 533.00 | 535.41 | 535.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 531.80 | 534.69 | 534.85 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 548.05 | 537.36 | 536.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 553.80 | 547.86 | 544.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 551.15 | 553.63 | 549.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:15:00 | 549.00 | 553.63 | 549.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 536.50 | 550.21 | 548.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 536.50 | 550.21 | 548.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 538.80 | 547.93 | 547.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 534.35 | 547.93 | 547.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 541.00 | 546.54 | 546.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 530.80 | 542.07 | 544.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 14:15:00 | 537.75 | 534.33 | 536.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 14:15:00 | 537.75 | 534.33 | 536.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 537.75 | 534.33 | 536.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 537.75 | 534.33 | 536.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 536.00 | 534.67 | 536.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 533.50 | 534.67 | 536.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 532.60 | 535.15 | 535.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:00:00 | 534.80 | 534.84 | 535.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 541.00 | 536.71 | 536.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 541.00 | 536.71 | 536.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 549.10 | 543.01 | 540.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 543.05 | 543.37 | 540.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 539.10 | 542.51 | 540.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 539.10 | 542.51 | 540.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 539.10 | 542.51 | 540.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 538.00 | 541.61 | 540.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:30:00 | 539.45 | 541.61 | 540.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 544.15 | 545.76 | 542.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 544.15 | 545.76 | 542.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 548.10 | 546.23 | 543.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 545.30 | 546.23 | 543.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 543.40 | 546.07 | 543.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 543.40 | 546.07 | 543.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 545.45 | 545.95 | 544.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:45:00 | 542.50 | 545.95 | 544.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 559.45 | 548.65 | 545.40 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 540.65 | 548.24 | 549.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 538.90 | 544.05 | 546.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 547.40 | 544.08 | 546.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 547.40 | 544.08 | 546.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 547.40 | 544.08 | 546.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 547.40 | 544.08 | 546.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 545.20 | 544.30 | 545.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 543.65 | 544.30 | 545.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 543.00 | 544.53 | 545.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 548.50 | 545.40 | 546.05 | SL hit (close>static) qty=1.00 sl=547.40 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 551.55 | 546.84 | 546.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 15:15:00 | 554.95 | 551.18 | 549.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 565.90 | 570.09 | 561.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 565.90 | 570.09 | 561.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 565.80 | 569.23 | 562.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 579.60 | 569.23 | 562.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:45:00 | 574.95 | 575.02 | 571.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 568.85 | 572.82 | 571.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:45:00 | 568.80 | 572.07 | 571.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 567.00 | 571.06 | 571.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 567.00 | 571.06 | 571.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 561.80 | 568.46 | 569.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 539.95 | 539.52 | 544.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 537.60 | 539.52 | 544.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 544.80 | 540.61 | 544.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 544.80 | 540.61 | 544.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 540.15 | 540.51 | 543.74 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 548.85 | 545.29 | 544.83 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 535.75 | 543.10 | 544.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 533.50 | 541.18 | 543.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 520.00 | 509.44 | 516.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 520.00 | 509.44 | 516.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 520.00 | 509.44 | 516.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 520.00 | 509.44 | 516.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 513.15 | 510.18 | 516.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 509.40 | 510.03 | 515.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:00:00 | 509.35 | 506.90 | 511.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 13:15:00 | 499.25 | 497.89 | 497.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 499.25 | 497.89 | 497.71 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 494.10 | 497.11 | 497.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 489.00 | 494.56 | 496.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 490.70 | 490.65 | 492.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:00:00 | 487.75 | 490.07 | 492.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 12:00:00 | 487.30 | 489.52 | 491.98 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 14:00:00 | 488.00 | 488.39 | 490.99 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 492.80 | 489.27 | 491.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 492.80 | 489.27 | 491.15 | SL hit (close>ema400) qty=1.00 sl=491.15 alert=retest1 |

### Cycle 86 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 499.90 | 492.34 | 492.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 502.55 | 495.71 | 493.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 499.80 | 500.10 | 497.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:30:00 | 500.25 | 500.10 | 497.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 491.60 | 498.43 | 497.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 491.60 | 498.43 | 497.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 490.60 | 496.86 | 497.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 490.00 | 495.49 | 496.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 486.80 | 485.55 | 489.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 486.80 | 485.55 | 489.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 486.80 | 485.55 | 489.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 482.30 | 485.16 | 487.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 482.95 | 484.03 | 486.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 15:15:00 | 487.50 | 485.42 | 485.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 15:15:00 | 487.50 | 485.42 | 485.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 518.20 | 491.98 | 488.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 13:15:00 | 506.05 | 507.81 | 502.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:45:00 | 505.45 | 507.81 | 502.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 502.25 | 506.60 | 503.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 502.25 | 506.60 | 503.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 503.05 | 505.89 | 503.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 503.05 | 505.89 | 503.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 500.55 | 504.46 | 503.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 500.55 | 504.46 | 503.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 501.80 | 503.93 | 503.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 497.40 | 503.93 | 503.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 497.90 | 502.73 | 502.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 492.55 | 499.73 | 501.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 496.20 | 494.53 | 496.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 496.20 | 494.53 | 496.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 496.20 | 494.53 | 496.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 496.20 | 494.53 | 496.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 494.95 | 494.61 | 496.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:00:00 | 491.00 | 493.89 | 496.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 466.45 | 477.73 | 483.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 477.70 | 476.97 | 481.51 | SL hit (close>ema200) qty=0.50 sl=476.97 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 491.10 | 482.06 | 482.02 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 481.10 | 482.00 | 482.08 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 494.25 | 484.37 | 483.13 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 478.80 | 482.97 | 483.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 472.50 | 480.88 | 482.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 480.05 | 474.45 | 476.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 480.05 | 474.45 | 476.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 480.05 | 474.45 | 476.76 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 483.00 | 478.63 | 478.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 492.95 | 481.49 | 479.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 484.20 | 489.26 | 484.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 484.20 | 489.26 | 484.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 484.20 | 489.26 | 484.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 484.20 | 489.26 | 484.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 484.00 | 488.20 | 484.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 480.80 | 486.72 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 483.00 | 485.98 | 484.30 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 478.85 | 483.05 | 483.25 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 495.45 | 485.46 | 484.29 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 485.15 | 488.14 | 488.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 483.20 | 487.15 | 487.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 11:15:00 | 481.10 | 480.50 | 483.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 481.10 | 480.50 | 483.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 480.95 | 480.42 | 482.05 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 486.70 | 482.00 | 481.93 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 482.35 | 482.87 | 482.90 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 484.50 | 483.20 | 483.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 486.60 | 483.88 | 483.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 485.75 | 486.61 | 485.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 485.75 | 486.61 | 485.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 485.75 | 486.61 | 485.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 485.75 | 486.61 | 485.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 485.15 | 486.32 | 485.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:15:00 | 484.65 | 486.32 | 485.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 482.15 | 485.49 | 485.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 482.15 | 485.49 | 485.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 482.05 | 484.80 | 484.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 486.35 | 484.80 | 484.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 497.30 | 487.30 | 485.93 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 483.40 | 488.57 | 488.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 476.00 | 483.17 | 485.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 480.65 | 480.01 | 482.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 482.75 | 480.01 | 482.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 481.80 | 480.37 | 482.22 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 482.85 | 482.56 | 482.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 485.10 | 483.27 | 482.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 492.25 | 492.56 | 488.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:30:00 | 492.80 | 492.56 | 488.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 495.25 | 492.78 | 489.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 499.65 | 494.16 | 492.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 488.45 | 491.88 | 491.74 | SL hit (close<static) qty=1.00 sl=489.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 487.30 | 490.97 | 491.34 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 506.55 | 494.00 | 492.49 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 495.65 | 498.59 | 498.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 15:15:00 | 495.40 | 497.12 | 498.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 499.45 | 497.56 | 498.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 499.45 | 497.56 | 498.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 499.45 | 497.56 | 498.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 499.45 | 497.56 | 498.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 494.70 | 496.99 | 497.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 493.35 | 496.57 | 497.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 493.45 | 495.02 | 496.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 491.80 | 493.88 | 495.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 493.95 | 493.31 | 495.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 493.20 | 492.79 | 494.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 494.15 | 492.79 | 494.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 493.60 | 492.95 | 494.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 493.20 | 492.95 | 494.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 494.70 | 493.43 | 494.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 494.70 | 493.43 | 494.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 493.75 | 493.49 | 494.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:15:00 | 494.25 | 493.49 | 494.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 496.00 | 493.99 | 494.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 496.00 | 493.99 | 494.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 495.00 | 494.20 | 494.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 496.00 | 494.20 | 494.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 495.00 | 494.36 | 494.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 495.30 | 494.54 | 494.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 495.30 | 494.54 | 494.50 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 493.05 | 494.29 | 494.39 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 500.00 | 494.77 | 494.52 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 490.60 | 494.23 | 494.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 489.40 | 492.26 | 493.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 481.80 | 481.71 | 485.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 481.80 | 481.71 | 485.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 474.50 | 474.16 | 477.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 474.90 | 474.16 | 477.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 472.50 | 472.74 | 474.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 463.45 | 470.13 | 472.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 472.50 | 470.33 | 470.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 472.50 | 470.33 | 470.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 473.00 | 470.87 | 470.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 471.00 | 471.41 | 470.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 471.00 | 471.41 | 470.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 471.00 | 471.41 | 470.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 470.75 | 471.41 | 470.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 470.10 | 471.15 | 470.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:30:00 | 471.10 | 471.15 | 470.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 470.45 | 471.01 | 470.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:00:00 | 472.35 | 471.28 | 470.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 468.60 | 471.16 | 471.02 | SL hit (close<static) qty=1.00 sl=469.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 467.95 | 470.49 | 470.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 14:15:00 | 465.95 | 469.07 | 470.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 11:15:00 | 470.25 | 467.65 | 468.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 470.25 | 467.65 | 468.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 470.25 | 467.65 | 468.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 470.25 | 467.65 | 468.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 469.15 | 467.95 | 468.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 469.80 | 467.95 | 468.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 471.80 | 468.91 | 469.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 471.80 | 468.91 | 469.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 469.50 | 469.03 | 469.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 468.50 | 468.52 | 468.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 464.60 | 458.08 | 457.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 464.60 | 458.08 | 457.70 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 451.55 | 457.68 | 457.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 450.70 | 455.32 | 456.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 457.55 | 455.76 | 456.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 457.55 | 455.76 | 456.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 457.55 | 455.76 | 456.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 457.55 | 455.76 | 456.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 457.40 | 456.09 | 456.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 459.10 | 456.09 | 456.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 460.60 | 457.83 | 457.57 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 456.30 | 457.99 | 458.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 12:15:00 | 453.60 | 457.11 | 457.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 457.85 | 457.25 | 457.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 14:15:00 | 457.85 | 457.25 | 457.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 457.85 | 457.25 | 457.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 455.85 | 457.40 | 457.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:30:00 | 455.90 | 453.14 | 453.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 455.85 | 453.14 | 453.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 457.80 | 454.64 | 454.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 457.80 | 454.64 | 454.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 460.55 | 455.82 | 454.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 456.65 | 457.18 | 456.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 456.65 | 457.18 | 456.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 456.65 | 457.18 | 456.03 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 452.75 | 455.11 | 455.34 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 456.90 | 455.05 | 454.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 459.00 | 456.25 | 455.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 458.05 | 458.73 | 457.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 458.05 | 458.73 | 457.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 458.05 | 458.73 | 457.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 458.05 | 458.73 | 457.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 458.25 | 458.63 | 457.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 459.20 | 458.71 | 457.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 458.75 | 458.66 | 457.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 455.25 | 457.98 | 457.58 | SL hit (close<static) qty=1.00 sl=457.40 alert=retest2 |

### Cycle 119 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 453.00 | 456.99 | 457.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 451.90 | 454.63 | 455.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 417.35 | 415.87 | 426.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:45:00 | 419.90 | 415.87 | 426.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 420.00 | 417.67 | 424.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 417.80 | 418.51 | 423.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 14:15:00 | 396.91 | 403.72 | 407.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 09:15:00 | 404.00 | 402.57 | 406.59 | SL hit (close>ema200) qty=0.50 sl=402.57 alert=retest2 |

### Cycle 120 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 378.80 | 376.59 | 376.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 384.00 | 379.26 | 377.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 15:15:00 | 397.15 | 397.54 | 391.94 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 406.45 | 397.54 | 391.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 13:15:00 | 400.55 | 400.01 | 395.12 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 395.90 | 399.02 | 395.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 394.00 | 399.02 | 395.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 393.35 | 397.89 | 395.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 393.35 | 397.89 | 395.31 | SL hit (close<ema400) qty=1.00 sl=395.31 alert=retest1 |

### Cycle 121 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 382.70 | 392.62 | 393.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 380.60 | 390.22 | 392.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 369.25 | 367.84 | 371.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:00:00 | 369.25 | 367.84 | 371.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 371.80 | 368.83 | 371.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 362.85 | 368.83 | 371.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 344.71 | 351.78 | 359.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 359.20 | 351.94 | 358.43 | SL hit (close>ema200) qty=0.50 sl=351.94 alert=retest2 |

### Cycle 122 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 365.15 | 360.77 | 360.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 376.50 | 367.30 | 365.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 373.15 | 377.33 | 374.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 373.15 | 377.33 | 374.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 373.15 | 377.33 | 374.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 373.15 | 377.33 | 374.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 368.00 | 375.46 | 374.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 368.00 | 375.46 | 374.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 361.65 | 371.32 | 372.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 350.85 | 367.23 | 370.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 15:15:00 | 349.10 | 348.46 | 353.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 09:15:00 | 352.45 | 348.46 | 353.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 351.90 | 349.15 | 353.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 347.45 | 349.01 | 352.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 346.85 | 348.71 | 351.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 347.00 | 348.51 | 351.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:45:00 | 347.10 | 347.62 | 349.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 349.70 | 347.93 | 349.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 351.30 | 347.93 | 349.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 350.00 | 348.35 | 349.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:15:00 | 351.55 | 348.35 | 349.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 351.10 | 348.90 | 349.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 351.50 | 348.90 | 349.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 350.85 | 349.40 | 349.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 350.85 | 349.40 | 349.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 349.95 | 349.51 | 349.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:15:00 | 349.00 | 349.51 | 349.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 349.00 | 349.69 | 349.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 349.20 | 349.18 | 349.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 351.90 | 349.72 | 349.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 15:15:00 | 351.90 | 349.72 | 349.63 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 346.05 | 348.99 | 349.31 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 11:15:00 | 361.00 | 351.64 | 350.47 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 15:15:00 | 351.95 | 352.21 | 352.22 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 354.50 | 352.20 | 352.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 358.05 | 354.27 | 353.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 354.90 | 365.00 | 361.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 354.90 | 365.00 | 361.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 354.90 | 365.00 | 361.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 354.90 | 365.00 | 361.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 352.95 | 362.59 | 360.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 352.95 | 362.59 | 360.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 353.40 | 359.06 | 359.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 351.55 | 355.33 | 357.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 12:15:00 | 367.60 | 357.13 | 357.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 12:15:00 | 367.60 | 357.13 | 357.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 367.60 | 357.13 | 357.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 370.95 | 357.13 | 357.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 13:15:00 | 373.35 | 360.37 | 359.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 377.55 | 363.81 | 360.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 366.35 | 368.85 | 365.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:00:00 | 366.35 | 368.85 | 365.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 365.50 | 367.69 | 365.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 11:45:00 | 370.40 | 367.78 | 366.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 369.35 | 368.03 | 366.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 369.55 | 368.03 | 366.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 356.30 | 364.71 | 365.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 356.30 | 364.71 | 365.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 351.85 | 362.14 | 364.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 352.60 | 352.52 | 355.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 349.20 | 352.52 | 355.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 349.00 | 348.74 | 351.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 345.35 | 348.81 | 350.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 346.70 | 348.53 | 350.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:15:00 | 347.05 | 348.30 | 349.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 346.00 | 347.84 | 349.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 349.75 | 348.22 | 349.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:00:00 | 349.75 | 348.22 | 349.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 348.15 | 348.21 | 349.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 346.00 | 347.77 | 349.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 328.08 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 329.36 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 329.70 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 328.70 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 328.70 | 335.38 | 340.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 330.55 | 328.05 | 333.07 | SL hit (close>ema200) qty=0.50 sl=328.05 alert=retest2 |

### Cycle 132 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 332.70 | 330.49 | 330.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 335.15 | 331.83 | 330.93 | Break + close above crossover candle high |

### Cycle 133 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 323.70 | 330.21 | 330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 321.75 | 328.51 | 329.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 328.70 | 327.56 | 328.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 328.70 | 327.56 | 328.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 328.70 | 327.56 | 328.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 328.70 | 327.56 | 328.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 329.30 | 327.91 | 328.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 329.00 | 327.91 | 328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 327.50 | 327.83 | 328.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 325.00 | 327.83 | 328.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 332.70 | 328.80 | 329.02 | SL hit (close>static) qty=1.00 sl=329.80 alert=retest2 |

### Cycle 134 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 335.00 | 330.04 | 329.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 337.10 | 333.57 | 331.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 336.10 | 337.81 | 335.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 336.10 | 337.81 | 335.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 336.10 | 337.81 | 335.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 331.00 | 337.81 | 335.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 329.30 | 336.11 | 334.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 329.30 | 336.11 | 334.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 323.55 | 333.59 | 333.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 13:15:00 | 322.40 | 327.45 | 329.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 15:15:00 | 328.95 | 327.48 | 329.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 15:15:00 | 328.95 | 327.48 | 329.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 328.95 | 327.48 | 329.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 320.70 | 327.48 | 329.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 331.25 | 327.31 | 327.54 | SL hit (close>static) qty=1.00 sl=329.65 alert=retest2 |

### Cycle 136 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 333.30 | 328.51 | 328.07 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 327.75 | 328.65 | 328.73 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 349.05 | 332.59 | 330.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 354.25 | 347.36 | 340.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 348.50 | 350.53 | 345.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 352.15 | 350.53 | 345.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 355.65 | 357.92 | 355.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 355.80 | 357.92 | 355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 354.05 | 357.14 | 355.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 347.35 | 357.14 | 355.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 346.35 | 354.99 | 354.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:15:00 | 345.05 | 354.99 | 354.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 346.50 | 353.29 | 353.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 343.45 | 348.93 | 351.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 342.25 | 340.91 | 344.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 342.25 | 340.91 | 344.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 341.05 | 340.94 | 344.30 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 345.85 | 344.69 | 344.68 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 343.05 | 344.66 | 344.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 341.05 | 343.94 | 344.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 341.75 | 340.59 | 342.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 341.75 | 340.59 | 342.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 341.75 | 340.59 | 342.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 341.75 | 340.59 | 342.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 341.95 | 340.86 | 342.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 343.50 | 340.86 | 342.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 342.25 | 341.14 | 342.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 342.50 | 341.14 | 342.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 343.00 | 341.51 | 342.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 342.25 | 341.51 | 342.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 343.00 | 341.81 | 342.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 342.85 | 341.81 | 342.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 343.15 | 342.20 | 342.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 343.15 | 342.20 | 342.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 343.50 | 342.46 | 342.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 343.50 | 342.46 | 342.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 343.90 | 342.74 | 342.64 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 15:15:00 | 340.05 | 342.13 | 342.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 335.65 | 340.83 | 341.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 333.55 | 331.19 | 334.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 333.55 | 331.19 | 334.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 333.55 | 331.19 | 334.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 333.55 | 331.19 | 334.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 333.85 | 332.15 | 333.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 334.00 | 332.15 | 333.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 332.50 | 332.22 | 333.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:30:00 | 334.10 | 332.22 | 333.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 333.00 | 329.22 | 330.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 333.00 | 329.22 | 330.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 333.30 | 330.03 | 330.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 333.15 | 330.03 | 330.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 328.00 | 328.17 | 329.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 320.85 | 328.17 | 329.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 329.05 | 325.14 | 324.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 329.05 | 325.14 | 324.68 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 309.50 | 321.82 | 323.44 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 325.05 | 322.85 | 322.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 326.80 | 324.39 | 323.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 321.30 | 327.55 | 326.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 321.30 | 327.55 | 326.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 321.30 | 327.55 | 326.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:45:00 | 322.00 | 327.55 | 326.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 326.10 | 327.26 | 326.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 327.35 | 327.44 | 326.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-16 09:15:00 | 360.09 | 344.97 | 339.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 342.10 | 348.95 | 348.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 339.55 | 345.85 | 347.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 343.80 | 341.84 | 344.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 343.80 | 341.84 | 344.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 343.80 | 341.84 | 344.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 343.80 | 341.84 | 344.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 344.00 | 342.27 | 344.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 343.70 | 342.27 | 344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 344.45 | 342.71 | 344.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 344.45 | 342.71 | 344.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 344.00 | 342.97 | 344.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 344.00 | 342.97 | 344.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 344.35 | 343.24 | 344.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:30:00 | 344.10 | 343.24 | 344.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 345.30 | 343.66 | 344.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 333.15 | 343.66 | 344.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 343.50 | 339.82 | 339.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 343.50 | 339.82 | 339.72 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 11:15:00 | 337.25 | 339.68 | 339.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 12:15:00 | 336.00 | 338.94 | 339.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 329.90 | 328.04 | 332.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 329.90 | 328.04 | 332.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 332.70 | 328.98 | 332.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 332.70 | 328.98 | 332.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 330.00 | 329.18 | 332.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:45:00 | 328.00 | 329.36 | 332.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:15:00 | 328.20 | 329.36 | 332.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:45:00 | 328.85 | 329.53 | 331.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 14:15:00 | 325.85 | 329.53 | 331.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 337.75 | 327.98 | 329.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 337.75 | 327.98 | 329.81 | SL hit (close>static) qty=1.00 sl=333.00 alert=retest2 |

### Cycle 150 — BUY (started 2026-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 13:15:00 | 321.00 | 319.05 | 319.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 323.00 | 320.16 | 319.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 315.50 | 319.22 | 319.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 315.50 | 319.22 | 319.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 315.50 | 319.22 | 319.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 315.50 | 319.22 | 319.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 10:15:00 | 316.35 | 318.65 | 318.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 12:15:00 | 313.80 | 317.39 | 318.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 14:15:00 | 317.90 | 316.93 | 317.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 317.90 | 316.93 | 317.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 317.90 | 316.93 | 317.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 317.90 | 316.93 | 317.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 317.25 | 316.99 | 317.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 325.70 | 316.99 | 317.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 328.60 | 319.31 | 318.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 15:15:00 | 332.20 | 326.23 | 323.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 339.55 | 340.14 | 335.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 10:00:00 | 339.55 | 340.14 | 335.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 343.00 | 343.57 | 341.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:45:00 | 346.75 | 343.07 | 341.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 13:15:00 | 339.75 | 342.37 | 341.97 | SL hit (close<static) qty=1.00 sl=340.05 alert=retest2 |

### Cycle 153 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 339.70 | 341.58 | 341.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 339.35 | 341.13 | 341.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 341.30 | 340.85 | 341.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 12:15:00 | 341.30 | 340.85 | 341.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 341.30 | 340.85 | 341.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 341.30 | 340.85 | 341.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 342.95 | 341.27 | 341.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:30:00 | 343.45 | 341.27 | 341.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 343.30 | 341.68 | 341.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 15:15:00 | 344.80 | 342.30 | 341.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 10:15:00 | 340.20 | 342.03 | 341.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 340.20 | 342.03 | 341.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 340.20 | 342.03 | 341.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:30:00 | 340.35 | 342.03 | 341.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 342.60 | 342.15 | 341.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 342.30 | 342.15 | 341.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 343.65 | 342.45 | 342.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:15:00 | 347.40 | 342.45 | 342.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:00:00 | 345.35 | 345.24 | 343.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:45:00 | 345.65 | 345.22 | 343.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 345.15 | 345.08 | 344.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 346.00 | 345.26 | 344.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 15:00:00 | 346.65 | 345.54 | 344.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 349.15 | 345.75 | 344.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 350.60 | 354.82 | 355.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 350.60 | 354.82 | 355.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 342.70 | 352.39 | 353.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 10:15:00 | 348.30 | 346.14 | 348.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 11:00:00 | 348.30 | 346.14 | 348.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 346.60 | 346.23 | 348.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:30:00 | 349.60 | 346.23 | 348.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 346.75 | 345.82 | 347.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 350.05 | 345.82 | 347.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 345.50 | 344.44 | 345.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 343.30 | 344.24 | 345.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 343.90 | 344.79 | 345.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 347.50 | 345.33 | 345.76 | SL hit (close>static) qty=1.00 sl=347.10 alert=retest2 |

### Cycle 156 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 350.70 | 346.41 | 346.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 353.70 | 348.61 | 347.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 349.60 | 350.57 | 348.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 349.60 | 350.57 | 348.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 350.90 | 350.54 | 349.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:30:00 | 347.95 | 350.54 | 349.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 350.95 | 350.64 | 349.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:45:00 | 359.10 | 352.12 | 350.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 09:15:00 | 483.30 | 2024-06-03 09:15:00 | 531.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-25 13:00:00 | 550.50 | 2024-06-26 14:15:00 | 557.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-06-26 09:30:00 | 552.80 | 2024-06-26 14:15:00 | 557.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-06-27 09:15:00 | 552.10 | 2024-06-27 09:15:00 | 556.70 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-07-03 12:45:00 | 554.50 | 2024-07-10 09:15:00 | 526.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-03 12:45:00 | 554.50 | 2024-07-12 09:15:00 | 524.20 | STOP_HIT | 0.50 | 5.46% |
| BUY | retest2 | 2024-07-18 14:30:00 | 542.70 | 2024-07-19 12:15:00 | 529.75 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-07-31 09:15:00 | 566.40 | 2024-08-01 10:15:00 | 547.00 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-07-31 09:45:00 | 562.45 | 2024-08-01 10:15:00 | 547.00 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-08-02 12:00:00 | 535.00 | 2024-08-05 09:15:00 | 508.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 12:00:00 | 535.00 | 2024-08-06 09:15:00 | 504.40 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2024-09-03 13:00:00 | 506.85 | 2024-09-06 11:15:00 | 507.80 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-09-12 14:45:00 | 516.00 | 2024-09-17 15:15:00 | 517.70 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2024-09-19 10:15:00 | 510.50 | 2024-09-23 10:15:00 | 520.35 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-09-19 11:15:00 | 507.40 | 2024-09-23 10:15:00 | 520.35 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-09-19 13:15:00 | 510.15 | 2024-09-23 10:15:00 | 520.35 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-09-19 14:00:00 | 510.60 | 2024-09-23 10:15:00 | 520.35 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-09-20 12:45:00 | 509.65 | 2024-09-23 10:15:00 | 520.35 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-09-20 13:30:00 | 507.40 | 2024-09-23 10:15:00 | 520.35 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-10-04 11:30:00 | 636.00 | 2024-10-04 13:15:00 | 621.80 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-10-28 09:15:00 | 474.10 | 2024-10-29 09:15:00 | 500.00 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2024-11-07 15:15:00 | 510.00 | 2024-11-11 09:15:00 | 484.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 15:15:00 | 510.00 | 2024-11-13 09:15:00 | 459.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-19 15:00:00 | 469.20 | 2024-11-25 11:15:00 | 474.45 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-11-27 10:15:00 | 493.55 | 2024-12-09 09:15:00 | 513.00 | STOP_HIT | 1.00 | 3.94% |
| BUY | retest2 | 2024-11-29 09:45:00 | 493.25 | 2024-12-09 09:15:00 | 513.00 | STOP_HIT | 1.00 | 4.00% |
| BUY | retest2 | 2024-11-29 10:30:00 | 492.25 | 2024-12-09 09:15:00 | 513.00 | STOP_HIT | 1.00 | 4.22% |
| SELL | retest2 | 2024-12-18 14:30:00 | 497.15 | 2024-12-18 15:15:00 | 501.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-23 09:15:00 | 488.10 | 2024-12-26 14:15:00 | 463.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-23 09:15:00 | 488.10 | 2024-12-27 09:15:00 | 470.90 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest2 | 2025-01-09 09:15:00 | 586.80 | 2025-01-10 09:15:00 | 531.70 | STOP_HIT | 1.00 | -9.39% |
| BUY | retest2 | 2025-01-09 14:00:00 | 554.75 | 2025-01-10 09:15:00 | 531.70 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2025-01-09 14:45:00 | 553.95 | 2025-01-10 09:15:00 | 531.70 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-01-17 11:45:00 | 518.35 | 2025-01-20 09:15:00 | 541.55 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-01-23 09:15:00 | 524.00 | 2025-01-27 09:15:00 | 497.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:45:00 | 524.20 | 2025-01-27 09:15:00 | 497.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 10:15:00 | 525.25 | 2025-01-27 09:15:00 | 498.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 524.00 | 2025-01-27 10:15:00 | 471.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 09:45:00 | 524.20 | 2025-01-27 10:15:00 | 471.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 10:15:00 | 525.25 | 2025-01-27 10:15:00 | 472.73 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-10 09:15:00 | 541.85 | 2025-02-10 09:15:00 | 514.00 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2025-02-21 12:00:00 | 500.25 | 2025-02-27 09:15:00 | 488.80 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-02-21 15:00:00 | 500.80 | 2025-02-27 09:15:00 | 488.80 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-02-25 10:00:00 | 500.55 | 2025-02-27 09:15:00 | 488.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-02-25 11:00:00 | 500.50 | 2025-02-27 09:15:00 | 488.80 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-03-11 10:30:00 | 491.20 | 2025-03-12 10:15:00 | 477.50 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-03-11 14:15:00 | 487.05 | 2025-03-12 10:15:00 | 477.50 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-03-11 15:15:00 | 488.00 | 2025-03-12 10:15:00 | 477.50 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-03-18 09:15:00 | 492.15 | 2025-03-26 12:15:00 | 507.00 | STOP_HIT | 1.00 | 3.02% |
| BUY | retest2 | 2025-04-15 13:00:00 | 533.95 | 2025-04-24 12:15:00 | 546.20 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2025-05-05 11:30:00 | 526.60 | 2025-05-05 13:15:00 | 537.45 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-05-06 11:15:00 | 527.55 | 2025-05-08 11:15:00 | 536.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-07 09:30:00 | 527.55 | 2025-05-08 11:15:00 | 536.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-07 10:00:00 | 526.50 | 2025-05-08 11:15:00 | 536.20 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-05-21 09:15:00 | 533.50 | 2025-05-23 09:15:00 | 541.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-22 09:15:00 | 532.60 | 2025-05-23 09:15:00 | 541.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-22 11:00:00 | 534.80 | 2025-05-23 09:15:00 | 541.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-02 12:15:00 | 543.65 | 2025-06-02 14:15:00 | 548.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-02 13:15:00 | 543.00 | 2025-06-02 14:15:00 | 548.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-05 09:15:00 | 579.60 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-06-06 11:45:00 | 574.95 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-06-09 10:45:00 | 568.85 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-06-09 11:45:00 | 568.80 | 2025-06-09 12:15:00 | 567.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-06-23 12:00:00 | 509.40 | 2025-07-03 13:15:00 | 499.25 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2025-06-24 10:00:00 | 509.35 | 2025-07-03 13:15:00 | 499.25 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest1 | 2025-07-08 11:00:00 | 487.75 | 2025-07-08 14:15:00 | 492.80 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest1 | 2025-07-08 12:00:00 | 487.30 | 2025-07-08 14:15:00 | 492.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest1 | 2025-07-08 14:00:00 | 488.00 | 2025-07-08 14:15:00 | 492.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-15 15:00:00 | 482.30 | 2025-07-17 15:15:00 | 487.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-16 12:30:00 | 482.95 | 2025-07-17 15:15:00 | 487.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-25 10:00:00 | 491.00 | 2025-07-29 09:15:00 | 466.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 10:00:00 | 491.00 | 2025-07-29 12:15:00 | 477.70 | STOP_HIT | 0.50 | 2.71% |
| BUY | retest2 | 2025-09-04 09:30:00 | 499.65 | 2025-09-05 09:15:00 | 488.45 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-12 13:15:00 | 493.35 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-09-12 14:45:00 | 493.45 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-15 09:45:00 | 491.80 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-15 12:30:00 | 493.95 | 2025-09-17 10:15:00 | 495.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-29 15:00:00 | 463.45 | 2025-10-01 13:15:00 | 472.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-03 14:00:00 | 472.35 | 2025-10-06 09:15:00 | 468.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-06 11:15:00 | 472.10 | 2025-10-06 11:15:00 | 467.95 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-08 09:30:00 | 468.50 | 2025-10-13 11:15:00 | 464.60 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-10-17 09:15:00 | 455.85 | 2025-10-24 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-20 13:30:00 | 455.90 | 2025-10-24 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-20 14:15:00 | 455.85 | 2025-10-24 10:15:00 | 457.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-31 12:30:00 | 459.20 | 2025-10-31 14:15:00 | 455.25 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-31 13:45:00 | 458.75 | 2025-10-31 14:15:00 | 455.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-10 15:15:00 | 417.80 | 2025-11-13 14:15:00 | 396.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 15:15:00 | 417.80 | 2025-11-14 09:15:00 | 404.00 | STOP_HIT | 0.50 | 3.30% |
| BUY | retest1 | 2025-12-01 09:15:00 | 406.45 | 2025-12-01 15:15:00 | 393.35 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest1 | 2025-12-01 13:15:00 | 400.55 | 2025-12-01 15:15:00 | 393.35 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-12-08 09:15:00 | 362.85 | 2025-12-09 09:15:00 | 344.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 362.85 | 2025-12-09 11:15:00 | 359.20 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest2 | 2025-12-10 10:00:00 | 367.40 | 2025-12-10 10:15:00 | 365.15 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-12-22 12:30:00 | 347.45 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-22 13:45:00 | 346.85 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-12-22 15:15:00 | 347.00 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-23 12:45:00 | 347.10 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-24 15:15:00 | 349.00 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-26 11:15:00 | 349.00 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-12-26 15:00:00 | 349.20 | 2025-12-26 15:15:00 | 351.90 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-09 11:45:00 | 370.40 | 2026-01-12 09:15:00 | 356.30 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-01-09 12:45:00 | 369.35 | 2026-01-12 09:15:00 | 356.30 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2026-01-09 13:15:00 | 369.55 | 2026-01-12 09:15:00 | 356.30 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-01-19 09:15:00 | 345.35 | 2026-01-21 09:15:00 | 328.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 346.70 | 2026-01-21 09:15:00 | 329.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:15:00 | 347.05 | 2026-01-21 09:15:00 | 329.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 12:00:00 | 346.00 | 2026-01-21 09:15:00 | 328.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:00:00 | 346.00 | 2026-01-21 09:15:00 | 328.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 345.35 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2026-01-19 10:15:00 | 346.70 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-19 11:15:00 | 347.05 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-19 12:00:00 | 346.00 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2026-01-19 15:00:00 | 346.00 | 2026-01-22 10:15:00 | 330.55 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2026-01-30 09:15:00 | 325.00 | 2026-01-30 09:15:00 | 332.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-04 09:15:00 | 320.70 | 2026-02-05 10:15:00 | 331.25 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-03-04 09:15:00 | 320.85 | 2026-03-06 09:15:00 | 329.05 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-03-12 11:45:00 | 327.35 | 2026-03-16 09:15:00 | 360.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 333.15 | 2026-03-25 09:15:00 | 343.50 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-03-30 09:45:00 | 328.00 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-03-30 10:15:00 | 328.20 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-03-30 13:45:00 | 328.85 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-03-30 14:15:00 | 325.85 | 2026-04-01 09:15:00 | 337.75 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-04-01 15:15:00 | 326.00 | 2026-04-07 15:15:00 | 309.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 15:15:00 | 326.00 | 2026-04-08 09:15:00 | 320.80 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest2 | 2026-04-21 09:45:00 | 346.75 | 2026-04-21 13:15:00 | 339.75 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-23 13:15:00 | 347.40 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2026-04-24 10:00:00 | 345.35 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-04-24 10:45:00 | 345.65 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2026-04-24 12:30:00 | 345.15 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.58% |
| BUY | retest2 | 2026-04-24 15:00:00 | 346.65 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2026-04-27 09:15:00 | 349.15 | 2026-04-29 15:15:00 | 350.60 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-05-06 10:45:00 | 343.30 | 2026-05-06 13:15:00 | 347.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-05-06 12:30:00 | 343.90 | 2026-05-06 13:15:00 | 347.50 | STOP_HIT | 1.00 | -1.05% |
