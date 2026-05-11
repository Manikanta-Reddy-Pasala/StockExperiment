# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1100.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 10 |
| TARGET_HIT | 10 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 25
- **Target hits / Stop hits / Partials:** 10 / 27 / 10
- **Avg / median % per leg:** 0.52% / -1.34%
- **Sum % (uncompounded):** 24.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 1 | 14.3% | 1 | 6 | 0 | -0.56% | -3.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 1 | 6 | 0 | -0.56% | -3.9% |
| SELL (all) | 40 | 21 | 52.5% | 9 | 21 | 10 | 0.71% | 28.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 21 | 52.5% | 9 | 21 | 10 | 0.71% | 28.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 22 | 46.8% | 10 | 27 | 10 | 0.52% | 24.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 527.50 | 601.85 | 602.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 517.95 | 597.94 | 600.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 487.70 | 486.54 | 506.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 499.70 | 487.21 | 505.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 499.70 | 487.21 | 505.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:30:00 | 494.95 | 487.37 | 505.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 13:15:00 | 470.20 | 485.75 | 502.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-05-13 09:15:00 | 445.45 | 483.86 | 500.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 502.30 | 480.61 | 480.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 507.40 | 481.09 | 480.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 10:15:00 | 484.85 | 485.10 | 483.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:30:00 | 485.00 | 485.10 | 483.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 483.00 | 485.07 | 483.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 483.00 | 485.07 | 483.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 482.75 | 485.05 | 483.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 487.90 | 485.05 | 483.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:15:00 | 485.00 | 485.15 | 483.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 11:15:00 | 481.35 | 485.13 | 483.29 | SL hit (close<static) qty=1.00 sl=482.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 452.00 | 481.51 | 481.58 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 13:15:00 | 501.75 | 481.10 | 481.07 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 10:15:00 | 459.00 | 481.02 | 481.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 11:15:00 | 451.60 | 480.73 | 480.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 457.40 | 447.41 | 460.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 11:15:00 | 457.40 | 447.41 | 460.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 457.40 | 447.41 | 460.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 457.40 | 447.41 | 460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 445.00 | 447.67 | 460.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 444.00 | 447.67 | 460.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:30:00 | 443.55 | 445.32 | 457.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 421.80 | 441.93 | 453.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 421.37 | 441.93 | 453.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 455.55 | 437.78 | 449.62 | SL hit (close>ema200) qty=0.50 sl=437.78 alert=retest2 |

### Cycle 6 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 511.75 | 439.72 | 439.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 534.25 | 446.02 | 442.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 564.60 | 588.96 | 557.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 564.60 | 588.96 | 557.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 564.60 | 588.96 | 557.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 564.60 | 588.96 | 557.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 557.50 | 587.64 | 557.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 557.50 | 587.64 | 557.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 547.90 | 587.25 | 557.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 547.90 | 587.25 | 557.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 546.50 | 586.84 | 557.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:30:00 | 548.90 | 586.84 | 557.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 550.95 | 580.52 | 556.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 550.95 | 580.52 | 556.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 539.85 | 580.12 | 556.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:30:00 | 542.40 | 580.12 | 556.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 412.50 | 539.30 | 539.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 10:15:00 | 405.30 | 537.97 | 538.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 531.80 | 530.49 | 534.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 531.80 | 530.49 | 534.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 531.80 | 530.49 | 534.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:45:00 | 510.30 | 530.14 | 534.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:30:00 | 509.30 | 529.98 | 534.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 10:15:00 | 560.60 | 531.26 | 534.74 | SL hit (close>static) qty=1.00 sl=554.75 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 538.00 | 478.72 | 478.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 545.05 | 481.17 | 479.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 518.90 | 526.54 | 511.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:45:00 | 522.10 | 526.51 | 511.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 524.50 | 526.42 | 512.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 522.75 | 525.97 | 512.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 524.40 | 525.92 | 512.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 512.80 | 525.66 | 513.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 512.80 | 525.66 | 513.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 513.95 | 525.54 | 513.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 511.15 | 525.54 | 513.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 512.45 | 525.41 | 513.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 512.45 | 525.41 | 513.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 508.00 | 525.24 | 513.50 | SL hit (close<static) qty=1.00 sl=511.25 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 733.60 | 795.88 | 795.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 728.20 | 792.81 | 794.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 11:15:00 | 753.45 | 753.04 | 769.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:00:00 | 753.45 | 753.04 | 769.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 769.95 | 713.24 | 737.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 780.15 | 713.24 | 737.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 776.50 | 719.33 | 739.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 764.00 | 719.84 | 739.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 794.35 | 721.02 | 740.02 | SL hit (close>static) qty=1.00 sl=787.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 882.70 | 755.38 | 755.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 890.60 | 775.75 | 765.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 795.90 | 816.50 | 794.82 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 09:15:00 | 743.25 | 780.05 | 780.12 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 09:15:00 | 816.00 | 779.89 | 779.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 837.55 | 787.18 | 783.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-03 11:30:00 | 494.95 | 2024-05-09 13:15:00 | 470.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:30:00 | 494.95 | 2024-05-13 09:15:00 | 445.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 496.50 | 2024-06-04 09:15:00 | 471.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 496.50 | 2024-06-04 09:15:00 | 481.45 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2024-06-03 15:15:00 | 495.00 | 2024-06-04 09:15:00 | 470.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 15:15:00 | 495.00 | 2024-06-04 09:15:00 | 481.45 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2024-06-25 13:00:00 | 493.35 | 2024-06-27 09:15:00 | 511.30 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2024-07-12 09:15:00 | 487.90 | 2024-07-16 11:15:00 | 481.35 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-07-15 12:15:00 | 485.00 | 2024-07-16 11:15:00 | 481.35 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-08-26 10:15:00 | 444.00 | 2024-09-09 09:15:00 | 421.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 09:30:00 | 443.55 | 2024-09-09 09:15:00 | 421.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 10:15:00 | 444.00 | 2024-09-12 13:15:00 | 455.55 | STOP_HIT | 0.50 | -2.60% |
| SELL | retest2 | 2024-08-30 09:30:00 | 443.55 | 2024-09-12 13:15:00 | 455.55 | STOP_HIT | 0.50 | -2.71% |
| SELL | retest2 | 2024-09-17 10:15:00 | 443.70 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -4.90% |
| SELL | retest2 | 2024-09-17 11:00:00 | 443.85 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2024-09-19 09:45:00 | 439.65 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -5.87% |
| SELL | retest2 | 2024-09-19 11:15:00 | 435.85 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -6.79% |
| SELL | retest2 | 2024-09-19 14:00:00 | 440.70 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -5.62% |
| SELL | retest2 | 2024-09-20 09:15:00 | 439.15 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -5.99% |
| SELL | retest2 | 2024-09-24 10:15:00 | 441.15 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2024-09-24 15:00:00 | 439.70 | 2024-09-25 09:15:00 | 465.45 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2024-09-27 14:45:00 | 440.70 | 2024-10-03 09:15:00 | 418.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 15:15:00 | 441.00 | 2024-10-03 09:15:00 | 418.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:45:00 | 440.70 | 2024-10-22 09:15:00 | 396.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-27 15:15:00 | 441.00 | 2024-10-22 09:15:00 | 396.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-30 11:45:00 | 510.30 | 2025-02-04 10:15:00 | 560.60 | STOP_HIT | 1.00 | -9.86% |
| SELL | retest2 | 2025-01-30 12:30:00 | 509.30 | 2025-02-04 10:15:00 | 560.60 | STOP_HIT | 1.00 | -10.07% |
| SELL | retest2 | 2025-02-10 09:30:00 | 509.00 | 2025-02-11 10:15:00 | 483.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:30:00 | 509.00 | 2025-02-12 09:15:00 | 458.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 496.10 | 2025-02-12 09:15:00 | 471.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 496.10 | 2025-02-14 11:15:00 | 446.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 15:00:00 | 460.55 | 2025-04-07 09:15:00 | 414.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 09:15:00 | 458.80 | 2025-04-07 09:15:00 | 412.92 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 14:30:00 | 458.95 | 2025-04-07 09:15:00 | 413.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 456.40 | 2025-04-07 09:15:00 | 410.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-30 12:15:00 | 473.45 | 2025-05-07 09:15:00 | 449.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 12:15:00 | 473.45 | 2025-05-07 10:15:00 | 483.95 | STOP_HIT | 0.50 | -2.22% |
| SELL | retest2 | 2025-05-09 09:15:00 | 469.95 | 2025-05-12 09:15:00 | 517.35 | STOP_HIT | 1.00 | -10.09% |
| SELL | retest2 | 2025-05-09 09:45:00 | 470.45 | 2025-05-12 09:15:00 | 517.35 | STOP_HIT | 1.00 | -9.97% |
| SELL | retest2 | 2025-05-09 10:30:00 | 471.80 | 2025-05-12 09:15:00 | 517.35 | STOP_HIT | 1.00 | -9.65% |
| BUY | retest2 | 2025-06-13 10:45:00 | 522.10 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-06-13 15:15:00 | 524.50 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-06-16 14:45:00 | 522.75 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-06-17 09:15:00 | 524.40 | 2025-06-19 15:15:00 | 508.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-24 13:45:00 | 524.00 | 2025-07-01 13:15:00 | 576.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-01 15:15:00 | 764.00 | 2026-02-02 09:15:00 | 794.35 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-02-02 12:00:00 | 765.55 | 2026-02-03 09:15:00 | 819.75 | STOP_HIT | 1.00 | -7.08% |
