# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 727.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 49 |
| ALERT2 | 49 |
| ALERT2_SKIP | 22 |
| ALERT3 | 139 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 79 |
| PARTIAL | 10 |
| TARGET_HIT | 12 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 55
- **Target hits / Stop hits / Partials:** 12 / 69 / 10
- **Avg / median % per leg:** 1.15% / -0.80%
- **Sum % (uncompounded):** 104.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 12 | 31.6% | 9 | 29 | 0 | 1.29% | 49.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 38 | 12 | 31.6% | 9 | 29 | 0 | 1.29% | 49.0% |
| SELL (all) | 53 | 24 | 45.3% | 3 | 40 | 10 | 1.06% | 55.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.58% | -3.2% |
| SELL @ 3rd Alert (retest2) | 51 | 24 | 47.1% | 3 | 38 | 10 | 1.16% | 59.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.58% | -3.2% |
| retest2 (combined) | 89 | 36 | 40.4% | 12 | 67 | 10 | 1.21% | 108.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 459.40 | 440.19 | 439.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 462.65 | 449.60 | 444.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 11:15:00 | 466.40 | 466.90 | 460.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:45:00 | 466.10 | 466.90 | 460.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 471.95 | 474.81 | 470.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:00:00 | 471.95 | 474.81 | 470.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 460.00 | 471.73 | 470.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:45:00 | 454.60 | 471.73 | 470.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 463.85 | 470.15 | 469.60 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 462.60 | 468.64 | 468.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 12:15:00 | 461.90 | 467.29 | 468.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 462.80 | 458.23 | 461.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 462.80 | 458.23 | 461.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 462.80 | 458.23 | 461.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 462.80 | 458.23 | 461.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 460.65 | 458.71 | 461.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 456.05 | 458.18 | 460.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 458.45 | 460.43 | 460.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 458.80 | 460.57 | 460.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 455.30 | 459.89 | 460.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 459.80 | 457.94 | 458.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 455.85 | 457.65 | 458.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 453.85 | 457.65 | 458.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 455.60 | 457.52 | 458.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 454.55 | 457.32 | 458.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 456.60 | 455.52 | 456.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 457.25 | 455.52 | 456.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 453.95 | 455.21 | 456.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 457.00 | 455.21 | 456.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 455.15 | 454.62 | 455.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 11:15:00 | 454.00 | 454.57 | 455.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 459.40 | 455.51 | 455.71 | SL hit (close>static) qty=1.00 sl=458.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 459.50 | 456.31 | 456.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 475.00 | 460.05 | 457.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 12:15:00 | 489.20 | 489.47 | 481.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:15:00 | 488.35 | 489.47 | 481.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 483.00 | 487.75 | 482.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 483.85 | 487.75 | 482.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 485.15 | 487.23 | 483.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:15:00 | 491.10 | 487.23 | 483.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-04 10:15:00 | 540.21 | 507.22 | 495.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 520.95 | 524.83 | 524.87 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 533.40 | 526.54 | 525.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 570.60 | 537.38 | 531.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 552.55 | 555.24 | 546.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 552.55 | 555.24 | 546.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 552.55 | 555.24 | 546.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 591.05 | 552.66 | 548.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 562.05 | 564.40 | 562.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 559.95 | 563.62 | 562.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 560.50 | 563.62 | 562.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 556.60 | 561.58 | 561.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 554.00 | 560.07 | 561.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 562.50 | 558.28 | 559.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 562.50 | 558.28 | 559.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 562.50 | 558.28 | 559.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 568.60 | 558.28 | 559.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 555.80 | 557.78 | 559.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 554.10 | 556.75 | 558.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 563.35 | 552.49 | 555.23 | SL hit (close>static) qty=1.00 sl=562.70 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 567.50 | 557.94 | 557.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 581.80 | 567.15 | 562.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 580.70 | 582.68 | 576.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 12:30:00 | 580.45 | 582.68 | 576.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 576.90 | 581.52 | 576.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 577.40 | 581.52 | 576.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 571.60 | 579.54 | 575.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 571.60 | 579.54 | 575.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 571.00 | 577.83 | 575.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 589.10 | 577.83 | 575.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 592.50 | 595.73 | 595.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 592.50 | 595.73 | 595.99 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 601.75 | 596.03 | 595.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 610.80 | 602.30 | 599.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 609.70 | 609.76 | 605.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 609.70 | 609.76 | 605.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 602.30 | 608.58 | 606.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 602.30 | 608.58 | 606.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 603.45 | 607.55 | 606.27 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 596.70 | 603.94 | 604.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 579.30 | 596.14 | 600.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 589.40 | 588.91 | 594.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 608.25 | 588.91 | 594.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 596.60 | 590.45 | 594.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 603.45 | 590.45 | 594.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 600.35 | 592.43 | 594.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 600.35 | 592.43 | 594.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 594.50 | 595.49 | 595.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 593.80 | 595.49 | 595.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 593.00 | 594.99 | 595.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 590.00 | 594.08 | 595.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:00:00 | 589.50 | 593.16 | 594.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 590.30 | 592.92 | 593.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 590.00 | 591.88 | 593.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 592.10 | 590.84 | 592.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 592.10 | 590.84 | 592.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 589.75 | 590.62 | 591.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 589.25 | 590.62 | 591.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 593.10 | 591.11 | 592.07 | SL hit (close>static) qty=1.00 sl=592.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 589.50 | 591.47 | 592.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 605.90 | 594.24 | 593.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 612.60 | 599.63 | 595.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 613.25 | 613.61 | 609.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 13:00:00 | 613.25 | 613.61 | 609.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 608.40 | 613.48 | 610.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 608.40 | 613.48 | 610.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 609.95 | 612.77 | 610.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 605.75 | 612.77 | 610.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 608.00 | 611.82 | 610.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 607.95 | 611.82 | 610.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 605.95 | 610.64 | 609.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 605.75 | 610.64 | 609.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 608.50 | 609.44 | 609.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 603.35 | 608.22 | 608.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 607.70 | 607.52 | 608.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 607.70 | 607.52 | 608.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 607.70 | 607.52 | 608.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 609.30 | 607.52 | 608.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 606.85 | 607.08 | 608.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 608.15 | 607.08 | 608.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 605.15 | 606.70 | 607.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 613.20 | 606.70 | 607.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 613.50 | 608.06 | 608.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 616.80 | 608.06 | 608.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 610.60 | 608.57 | 608.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 616.85 | 610.50 | 609.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 610.10 | 611.10 | 609.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 610.10 | 611.10 | 609.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 633.50 | 615.42 | 612.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 637.90 | 628.80 | 621.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 616.35 | 623.84 | 624.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 616.35 | 623.84 | 624.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 610.60 | 621.19 | 622.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 601.35 | 600.46 | 606.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 601.35 | 600.46 | 606.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 605.95 | 601.75 | 606.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 612.00 | 601.75 | 606.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 605.80 | 602.56 | 606.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 604.25 | 602.56 | 606.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 611.60 | 608.24 | 607.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 611.60 | 608.24 | 607.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 615.60 | 609.98 | 608.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 624.90 | 628.89 | 623.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 624.90 | 628.89 | 623.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 617.00 | 626.51 | 622.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 624.45 | 626.51 | 622.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 620.75 | 625.36 | 622.54 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 619.70 | 621.20 | 621.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 615.50 | 620.06 | 620.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 606.95 | 603.66 | 607.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 606.95 | 603.66 | 607.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 606.95 | 603.66 | 607.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 606.95 | 603.66 | 607.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 608.10 | 604.55 | 607.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 607.60 | 604.55 | 607.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 605.20 | 604.68 | 607.61 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 621.60 | 609.32 | 608.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 626.15 | 617.01 | 613.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 619.05 | 619.49 | 615.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:45:00 | 619.00 | 619.49 | 615.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 618.10 | 619.21 | 615.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 618.10 | 619.21 | 615.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 616.20 | 619.99 | 617.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 615.30 | 619.99 | 617.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 616.65 | 619.32 | 617.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 617.15 | 619.32 | 617.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 616.05 | 618.67 | 617.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:30:00 | 614.45 | 618.67 | 617.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 616.30 | 618.31 | 617.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 616.30 | 618.31 | 617.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 615.65 | 617.78 | 617.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 615.65 | 617.78 | 617.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 614.50 | 617.12 | 617.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 618.70 | 617.12 | 617.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 582.20 | 610.80 | 614.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 582.20 | 610.80 | 614.24 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 596.30 | 592.39 | 591.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 605.65 | 596.79 | 594.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 669.90 | 671.20 | 656.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 669.90 | 671.20 | 656.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 658.40 | 665.60 | 659.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 658.40 | 665.60 | 659.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 665.00 | 665.48 | 659.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 667.10 | 664.40 | 659.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 673.85 | 665.78 | 661.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 657.00 | 662.82 | 661.43 | SL hit (close<static) qty=1.00 sl=657.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 657.00 | 662.82 | 661.43 | SL hit (close<static) qty=1.00 sl=657.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 10:15:00 | 667.80 | 661.36 | 660.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 693.70 | 698.72 | 698.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 693.70 | 698.72 | 698.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 679.55 | 693.94 | 696.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 687.65 | 684.92 | 689.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 687.65 | 684.92 | 689.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 687.65 | 684.92 | 689.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 687.65 | 684.92 | 689.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 690.40 | 686.02 | 689.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 690.40 | 686.02 | 689.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 688.65 | 686.54 | 689.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 688.65 | 686.54 | 689.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 691.00 | 687.43 | 689.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 690.10 | 687.43 | 689.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 688.35 | 687.62 | 689.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 685.95 | 687.28 | 688.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 683.30 | 686.42 | 688.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:15:00 | 685.70 | 687.18 | 688.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 702.80 | 690.30 | 689.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 702.80 | 690.30 | 689.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 11:15:00 | 702.80 | 690.30 | 689.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 702.80 | 690.30 | 689.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 704.20 | 701.00 | 698.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 703.15 | 703.96 | 700.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 703.15 | 703.96 | 700.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 703.20 | 704.70 | 702.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 712.70 | 704.70 | 702.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 707.70 | 709.19 | 705.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 709.00 | 707.68 | 705.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 712.25 | 706.94 | 705.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 714.05 | 708.36 | 706.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 702.20 | 708.37 | 708.65 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 715.40 | 709.74 | 709.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 718.60 | 714.15 | 712.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 13:15:00 | 715.20 | 716.63 | 713.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 715.20 | 716.63 | 713.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 717.00 | 716.71 | 714.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 724.40 | 717.49 | 714.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 709.35 | 715.71 | 715.18 | SL hit (close<static) qty=1.00 sl=714.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 712.65 | 714.47 | 714.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 705.85 | 710.71 | 712.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 710.00 | 707.42 | 709.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 710.00 | 707.42 | 709.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 710.00 | 707.42 | 709.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 711.25 | 707.42 | 709.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 708.50 | 707.63 | 709.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 708.80 | 707.63 | 709.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 695.25 | 701.42 | 705.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 689.30 | 699.51 | 704.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:15:00 | 654.83 | 667.23 | 679.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 647.90 | 646.57 | 656.31 | SL hit (close>ema200) qty=0.50 sl=646.57 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 613.25 | 611.29 | 611.11 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 606.75 | 610.55 | 610.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 602.55 | 608.54 | 609.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 605.50 | 605.01 | 607.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 605.50 | 605.01 | 607.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 606.50 | 605.15 | 606.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 606.50 | 605.15 | 606.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 604.70 | 605.06 | 606.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:15:00 | 613.50 | 605.06 | 606.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 611.40 | 606.32 | 607.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 618.50 | 606.32 | 607.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 613.90 | 607.84 | 607.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 616.00 | 611.31 | 609.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 15:15:00 | 615.55 | 615.79 | 613.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 611.30 | 615.79 | 613.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 609.30 | 614.49 | 612.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 609.30 | 614.49 | 612.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 613.10 | 614.21 | 612.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 611.80 | 614.21 | 612.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 611.45 | 613.66 | 612.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 610.60 | 613.66 | 612.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 616.00 | 614.13 | 612.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 625.15 | 616.30 | 614.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 612.15 | 617.70 | 618.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 612.15 | 617.70 | 618.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 604.55 | 611.19 | 614.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 14:15:00 | 608.00 | 607.29 | 610.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 15:00:00 | 608.00 | 607.29 | 610.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 595.95 | 605.15 | 609.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 594.20 | 603.21 | 608.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:00:00 | 595.45 | 603.21 | 608.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 595.20 | 599.19 | 604.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 564.49 | 574.26 | 587.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 565.68 | 574.26 | 587.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 565.44 | 574.26 | 587.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-11 10:15:00 | 534.78 | 549.85 | 565.34 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-11 10:15:00 | 535.91 | 549.85 | 565.34 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-11 10:15:00 | 535.68 | 549.85 | 565.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 552.30 | 549.56 | 549.43 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 547.05 | 549.58 | 549.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 546.30 | 548.69 | 549.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 541.95 | 541.41 | 544.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:45:00 | 542.35 | 541.41 | 544.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 543.10 | 541.75 | 544.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 548.00 | 541.75 | 544.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 542.05 | 541.81 | 543.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:45:00 | 543.00 | 541.81 | 543.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 543.10 | 542.07 | 543.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 544.25 | 542.07 | 543.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 544.25 | 542.50 | 543.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 544.25 | 542.50 | 543.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 544.90 | 542.98 | 544.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 536.50 | 542.98 | 544.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 509.67 | 516.95 | 526.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 514.45 | 513.42 | 519.96 | SL hit (close>ema200) qty=0.50 sl=513.42 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 526.90 | 521.44 | 521.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 528.60 | 522.87 | 521.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 526.10 | 526.26 | 524.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 15:15:00 | 526.10 | 526.26 | 524.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 526.10 | 526.26 | 524.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 522.25 | 526.26 | 524.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 522.70 | 525.55 | 524.54 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 519.80 | 523.27 | 523.65 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 532.75 | 524.67 | 524.06 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 520.75 | 523.72 | 523.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 518.30 | 522.24 | 523.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 521.85 | 521.40 | 522.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 521.85 | 521.40 | 522.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 521.85 | 521.40 | 522.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:15:00 | 523.85 | 521.40 | 522.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 523.10 | 521.74 | 522.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 523.80 | 521.74 | 522.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 521.60 | 521.71 | 522.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 520.25 | 521.20 | 522.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 525.55 | 521.16 | 521.72 | SL hit (close>static) qty=1.00 sl=523.15 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 521.00 | 521.13 | 521.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 527.50 | 522.40 | 522.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 527.50 | 522.40 | 522.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 528.00 | 524.97 | 523.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 12:15:00 | 528.00 | 529.82 | 527.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:00:00 | 528.00 | 529.82 | 527.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 526.45 | 529.14 | 527.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 526.00 | 529.14 | 527.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 525.90 | 528.49 | 527.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 525.90 | 528.49 | 527.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 527.25 | 528.24 | 527.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 521.50 | 528.24 | 527.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 518.05 | 526.21 | 526.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 516.95 | 522.13 | 524.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 525.50 | 520.34 | 522.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 525.50 | 520.34 | 522.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 525.50 | 520.34 | 522.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 525.50 | 520.34 | 522.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 537.75 | 523.82 | 523.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 537.75 | 523.82 | 523.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 540.65 | 527.19 | 525.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 548.60 | 531.47 | 527.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 562.15 | 562.55 | 556.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 13:30:00 | 560.95 | 562.55 | 556.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 562.50 | 567.35 | 563.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 560.95 | 567.35 | 563.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 568.70 | 567.62 | 564.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 572.30 | 567.81 | 564.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:00:00 | 571.55 | 568.56 | 565.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:45:00 | 571.30 | 569.24 | 565.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:30:00 | 571.50 | 568.96 | 565.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 567.00 | 568.89 | 566.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 570.90 | 568.07 | 567.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 573.20 | 567.55 | 567.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 573.65 | 573.37 | 572.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 11:15:00 | 566.85 | 571.34 | 571.77 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 575.00 | 571.78 | 571.76 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 566.40 | 570.71 | 571.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 563.60 | 568.41 | 570.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 561.00 | 555.37 | 557.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 561.00 | 555.37 | 557.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 561.00 | 555.37 | 557.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 561.55 | 555.37 | 557.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 565.80 | 557.46 | 558.56 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 567.15 | 560.20 | 559.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 573.00 | 562.27 | 560.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 567.00 | 567.15 | 564.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 567.00 | 567.15 | 564.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 567.00 | 567.15 | 564.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 564.55 | 566.65 | 564.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 566.80 | 566.68 | 564.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 568.90 | 567.19 | 565.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 568.05 | 570.59 | 568.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:00:00 | 569.85 | 570.59 | 568.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 571.95 | 570.84 | 568.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-06 13:15:00 | 625.79 | 596.34 | 584.85 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-06 13:15:00 | 624.86 | 596.34 | 584.85 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-01-06 13:15:00 | 626.84 | 596.34 | 584.85 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 595.00 | 601.55 | 597.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 595.00 | 601.55 | 597.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 594.25 | 600.09 | 597.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 594.80 | 600.09 | 597.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 583.90 | 593.60 | 594.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 583.90 | 593.60 | 594.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 15:15:00 | 579.60 | 585.22 | 589.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 569.50 | 565.61 | 571.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 569.50 | 565.61 | 571.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 569.50 | 565.61 | 571.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 569.45 | 565.61 | 571.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 571.00 | 566.69 | 571.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 571.00 | 566.69 | 571.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 571.00 | 567.55 | 571.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 571.25 | 567.55 | 571.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 570.75 | 568.19 | 571.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 569.50 | 568.19 | 571.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 566.80 | 567.91 | 570.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 565.70 | 568.74 | 570.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 565.55 | 568.74 | 570.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 565.00 | 568.20 | 569.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 565.80 | 567.86 | 569.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 555.95 | 555.69 | 560.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 551.30 | 555.69 | 560.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 551.20 | 553.91 | 559.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 550.50 | 553.23 | 558.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 537.41 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 537.27 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 536.75 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 537.51 | 547.68 | 554.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 553.95 | 539.79 | 544.78 | SL hit (close>ema200) qty=0.50 sl=539.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 553.95 | 539.79 | 544.78 | SL hit (close>ema200) qty=0.50 sl=539.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 553.95 | 539.79 | 544.78 | SL hit (close>ema200) qty=0.50 sl=539.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 553.95 | 539.79 | 544.78 | SL hit (close>ema200) qty=0.50 sl=539.79 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 551.60 | 539.79 | 544.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 551.55 | 542.14 | 545.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 549.90 | 547.39 | 547.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 549.90 | 547.39 | 547.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 549.90 | 547.39 | 547.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 549.90 | 547.39 | 547.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 549.90 | 547.39 | 547.23 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 545.45 | 546.93 | 547.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 538.95 | 545.34 | 546.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 535.05 | 533.16 | 537.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 535.05 | 533.16 | 537.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 535.05 | 533.16 | 537.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 552.30 | 533.16 | 537.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 552.05 | 536.94 | 538.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 554.50 | 536.94 | 538.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 554.35 | 540.42 | 540.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 558.35 | 549.16 | 544.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 548.15 | 550.64 | 546.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 548.15 | 550.64 | 546.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 548.15 | 550.64 | 546.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 547.00 | 550.64 | 546.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 546.50 | 549.81 | 546.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 546.50 | 549.81 | 546.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 548.20 | 549.49 | 546.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 549.50 | 549.35 | 547.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:15:00 | 549.50 | 549.35 | 547.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 542.65 | 556.21 | 555.05 | SL hit (close<static) qty=1.00 sl=545.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 542.65 | 556.21 | 555.05 | SL hit (close<static) qty=1.00 sl=545.95 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 542.25 | 553.42 | 553.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 540.20 | 550.78 | 552.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 543.30 | 542.07 | 546.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 543.30 | 542.07 | 546.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 545.65 | 542.79 | 546.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 546.45 | 542.79 | 546.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 550.90 | 544.41 | 547.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 550.90 | 544.41 | 547.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 546.20 | 544.77 | 546.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 572.50 | 544.77 | 546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 565.75 | 548.96 | 548.67 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 558.95 | 564.17 | 564.76 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 575.55 | 563.99 | 563.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 582.85 | 572.06 | 567.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 576.50 | 576.56 | 572.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:30:00 | 576.40 | 576.56 | 572.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 573.10 | 575.59 | 572.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 573.10 | 575.59 | 572.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 574.00 | 575.27 | 572.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:15:00 | 571.65 | 575.27 | 572.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 571.85 | 574.59 | 572.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:15:00 | 571.00 | 574.59 | 572.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 571.00 | 573.87 | 572.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 574.75 | 573.80 | 572.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 567.20 | 572.48 | 572.03 | SL hit (close<static) qty=1.00 sl=569.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 567.90 | 571.56 | 571.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 557.40 | 566.00 | 568.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 553.90 | 553.84 | 558.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 553.90 | 553.84 | 558.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 550.10 | 553.16 | 557.49 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 568.50 | 559.66 | 558.56 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 554.30 | 558.71 | 558.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 553.80 | 557.33 | 558.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 563.55 | 556.88 | 557.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 563.55 | 556.88 | 557.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 563.55 | 556.88 | 557.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 567.50 | 556.88 | 557.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 564.05 | 558.31 | 558.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 575.50 | 568.68 | 565.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 593.85 | 594.29 | 587.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 593.85 | 594.29 | 587.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 576.90 | 590.79 | 588.60 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 576.70 | 585.24 | 586.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 568.55 | 578.32 | 582.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 566.90 | 564.83 | 571.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 563.50 | 564.56 | 570.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:45:00 | 562.75 | 564.54 | 570.33 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 566.00 | 564.83 | 569.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 568.00 | 564.83 | 569.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 572.05 | 566.62 | 569.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 572.05 | 566.62 | 569.89 | SL hit (close>ema400) qty=1.00 sl=569.89 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 572.05 | 566.62 | 569.89 | SL hit (close>ema400) qty=1.00 sl=569.89 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 572.05 | 566.62 | 569.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 576.00 | 568.50 | 570.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 574.45 | 569.67 | 570.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 569.85 | 569.71 | 570.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 566.00 | 570.06 | 570.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 537.70 | 563.70 | 567.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 555.95 | 549.58 | 556.34 | SL hit (close>ema200) qty=0.50 sl=549.58 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 569.00 | 560.27 | 559.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 580.65 | 564.35 | 561.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 566.85 | 569.93 | 565.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 566.85 | 569.93 | 565.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 566.85 | 569.93 | 565.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 566.85 | 569.93 | 565.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 572.55 | 569.91 | 566.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 577.60 | 571.93 | 567.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 576.55 | 575.90 | 571.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 556.30 | 569.07 | 569.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 556.30 | 569.07 | 569.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 556.30 | 569.07 | 569.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 553.55 | 565.96 | 567.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 551.40 | 545.35 | 551.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 551.40 | 545.35 | 551.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 551.40 | 545.35 | 551.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 551.40 | 545.35 | 551.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 549.85 | 546.25 | 551.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 546.35 | 546.25 | 551.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 561.50 | 551.00 | 551.66 | SL hit (close>static) qty=1.00 sl=553.60 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 563.85 | 553.57 | 552.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 564.80 | 557.28 | 554.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 554.25 | 559.68 | 557.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 554.25 | 559.68 | 557.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 554.25 | 559.68 | 557.01 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 544.20 | 553.70 | 554.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 539.50 | 550.49 | 552.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 551.10 | 547.40 | 550.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 13:15:00 | 551.10 | 547.40 | 550.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 551.10 | 547.40 | 550.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 14:00:00 | 551.10 | 547.40 | 550.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 544.80 | 546.88 | 549.70 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 559.50 | 551.00 | 550.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 563.75 | 553.55 | 551.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 568.10 | 568.90 | 562.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 563.55 | 568.90 | 562.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 558.40 | 566.80 | 562.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 558.40 | 566.80 | 562.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 557.25 | 564.89 | 562.02 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 553.80 | 560.29 | 560.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 551.00 | 558.43 | 559.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 553.55 | 543.21 | 548.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 553.55 | 543.21 | 548.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 553.55 | 543.21 | 548.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 554.00 | 543.21 | 548.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 551.30 | 544.83 | 548.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 550.30 | 544.83 | 548.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 554.10 | 551.31 | 551.14 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 545.50 | 550.46 | 550.91 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 554.85 | 551.56 | 551.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 559.80 | 553.20 | 552.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 552.50 | 554.72 | 553.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 552.50 | 554.72 | 553.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 552.50 | 554.72 | 553.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 552.50 | 554.72 | 553.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 557.85 | 555.34 | 553.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 563.85 | 556.84 | 554.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 562.10 | 557.71 | 555.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:45:00 | 564.00 | 559.56 | 556.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:45:00 | 564.80 | 561.37 | 557.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 562.00 | 562.79 | 560.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 573.35 | 562.79 | 560.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 620.24 | 610.97 | 602.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-15 09:15:00 | 618.31 | 610.97 | 602.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-15 09:15:00 | 620.40 | 610.97 | 602.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-15 09:15:00 | 621.28 | 610.97 | 602.91 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-17 09:15:00 | 630.69 | 636.15 | 623.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 672.10 | 689.68 | 691.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 662.00 | 684.14 | 689.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 668.00 | 665.93 | 675.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 668.55 | 665.93 | 675.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 665.60 | 666.40 | 671.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:45:00 | 665.45 | 666.40 | 671.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 673.60 | 667.84 | 671.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 673.60 | 667.84 | 671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 668.75 | 668.02 | 671.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 667.55 | 667.93 | 670.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 668.00 | 669.08 | 670.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 668.35 | 669.42 | 670.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 667.50 | 668.57 | 669.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 659.15 | 666.44 | 668.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 666.60 | 666.44 | 668.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 664.20 | 662.60 | 665.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:15:00 | 672.95 | 662.60 | 665.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | SL hit (close>static) qty=1.00 sl=674.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | SL hit (close>static) qty=1.00 sl=674.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | SL hit (close>static) qty=1.00 sl=674.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 675.60 | 665.20 | 666.03 | SL hit (close>static) qty=1.00 sl=674.30 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 675.60 | 665.20 | 666.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 676.15 | 667.39 | 666.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 688.85 | 678.65 | 674.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 719.50 | 719.98 | 709.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 719.50 | 719.98 | 709.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 12:00:00 | 456.05 | 2025-05-28 14:15:00 | 459.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-22 14:15:00 | 458.45 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-05-23 09:45:00 | 458.80 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-05-23 10:45:00 | 455.30 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-26 10:45:00 | 455.85 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-26 11:15:00 | 453.85 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-26 12:15:00 | 455.60 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-05-26 13:15:00 | 454.55 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-05-28 11:15:00 | 454.00 | 2025-05-28 15:15:00 | 459.50 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-03 10:15:00 | 491.10 | 2025-06-04 10:15:00 | 540.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 09:15:00 | 591.05 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2025-06-18 09:45:00 | 562.05 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-18 10:30:00 | 559.95 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-18 11:00:00 | 560.50 | 2025-06-18 12:15:00 | 556.60 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-19 11:30:00 | 554.10 | 2025-06-20 09:15:00 | 563.35 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-06-25 09:15:00 | 589.10 | 2025-07-01 14:15:00 | 592.50 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-07-10 10:45:00 | 590.00 | 2025-07-14 09:15:00 | 593.10 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-10 12:00:00 | 589.50 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-07-11 09:15:00 | 590.30 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-11 10:30:00 | 590.00 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-07-14 09:15:00 | 589.25 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-14 11:30:00 | 589.50 | 2025-07-14 14:15:00 | 605.90 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-24 10:15:00 | 637.90 | 2025-07-25 12:15:00 | 616.35 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-07-30 10:15:00 | 604.25 | 2025-07-30 14:15:00 | 611.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-08-14 09:15:00 | 618.70 | 2025-08-14 12:15:00 | 582.20 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest2 | 2025-08-26 14:30:00 | 667.10 | 2025-08-28 14:15:00 | 657.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-08-28 11:00:00 | 673.85 | 2025-08-28 14:15:00 | 657.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-08-29 10:15:00 | 667.80 | 2025-09-04 12:15:00 | 693.70 | STOP_HIT | 1.00 | 3.88% |
| SELL | retest2 | 2025-09-08 15:00:00 | 685.95 | 2025-09-09 11:15:00 | 702.80 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-09-09 09:30:00 | 683.30 | 2025-09-09 11:15:00 | 702.80 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-09-09 11:15:00 | 685.70 | 2025-09-09 11:15:00 | 702.80 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-09-15 09:15:00 | 712.70 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-15 13:15:00 | 707.70 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-15 14:30:00 | 709.00 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-16 09:15:00 | 712.25 | 2025-09-17 13:15:00 | 702.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-22 09:30:00 | 724.40 | 2025-09-22 14:15:00 | 709.35 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-26 11:15:00 | 689.30 | 2025-09-30 10:15:00 | 654.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 11:15:00 | 689.30 | 2025-10-01 15:15:00 | 647.90 | STOP_HIT | 0.50 | 6.01% |
| BUY | retest2 | 2025-10-29 13:30:00 | 625.15 | 2025-11-03 11:15:00 | 612.15 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-11-06 10:30:00 | 594.20 | 2025-11-10 09:15:00 | 564.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 11:00:00 | 595.45 | 2025-11-10 09:15:00 | 565.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 15:00:00 | 595.20 | 2025-11-10 09:15:00 | 565.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:30:00 | 594.20 | 2025-11-11 10:15:00 | 534.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 11:00:00 | 595.45 | 2025-11-11 10:15:00 | 535.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 15:00:00 | 595.20 | 2025-11-11 10:15:00 | 535.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 536.50 | 2025-11-24 14:15:00 | 509.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 536.50 | 2025-11-25 13:15:00 | 514.45 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-12-02 12:30:00 | 520.25 | 2025-12-03 09:15:00 | 525.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-03 10:45:00 | 521.00 | 2025-12-03 11:15:00 | 527.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-16 12:15:00 | 572.30 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-16 13:00:00 | 571.55 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-16 13:45:00 | 571.30 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-16 14:30:00 | 571.50 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-18 11:45:00 | 570.90 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-19 09:15:00 | 573.20 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-23 09:45:00 | 573.65 | 2025-12-23 11:15:00 | 566.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-02 10:45:00 | 568.90 | 2026-01-06 13:15:00 | 625.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 09:30:00 | 568.05 | 2026-01-06 13:15:00 | 624.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 10:00:00 | 569.85 | 2026-01-06 13:15:00 | 626.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-05 10:30:00 | 571.95 | 2026-01-08 15:15:00 | 583.90 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2026-01-16 11:45:00 | 565.70 | 2026-01-20 14:15:00 | 537.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 565.55 | 2026-01-20 14:15:00 | 537.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:45:00 | 565.00 | 2026-01-20 14:15:00 | 536.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 14:15:00 | 565.80 | 2026-01-20 14:15:00 | 537.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 565.70 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2026-01-16 12:15:00 | 565.55 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-01-16 12:45:00 | 565.00 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2026-01-16 14:15:00 | 565.80 | 2026-01-22 09:15:00 | 553.95 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2026-01-20 09:15:00 | 551.30 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-01-20 10:30:00 | 551.20 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-01-20 12:00:00 | 550.50 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-22 10:15:00 | 551.60 | 2026-01-22 14:15:00 | 549.90 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-01-29 13:30:00 | 549.50 | 2026-02-01 12:15:00 | 542.65 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-01-29 14:15:00 | 549.50 | 2026-02-01 12:15:00 | 542.65 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-11 09:30:00 | 574.75 | 2026-02-11 10:15:00 | 567.20 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2026-03-05 11:00:00 | 563.50 | 2026-03-05 14:15:00 | 572.05 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest1 | 2026-03-05 11:45:00 | 562.75 | 2026-03-05 14:15:00 | 572.05 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-03-06 15:15:00 | 566.00 | 2026-03-09 09:15:00 | 537.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 566.00 | 2026-03-10 09:15:00 | 555.95 | STOP_HIT | 0.50 | 1.78% |
| BUY | retest2 | 2026-03-12 10:45:00 | 577.60 | 2026-03-13 10:15:00 | 556.30 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-03-12 15:00:00 | 576.55 | 2026-03-13 10:15:00 | 556.30 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-03-17 11:15:00 | 546.35 | 2026-03-18 09:15:00 | 561.50 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-04-06 11:45:00 | 563.85 | 2026-04-15 09:15:00 | 620.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:15:00 | 562.10 | 2026-04-15 09:15:00 | 618.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 14:45:00 | 564.00 | 2026-04-15 09:15:00 | 620.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 09:45:00 | 564.80 | 2026-04-15 09:15:00 | 621.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 573.35 | 2026-04-17 09:15:00 | 630.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 14:00:00 | 667.55 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-29 11:00:00 | 668.00 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-29 13:15:00 | 668.35 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-04-29 14:45:00 | 667.50 | 2026-05-04 10:15:00 | 675.60 | STOP_HIT | 1.00 | -1.21% |
