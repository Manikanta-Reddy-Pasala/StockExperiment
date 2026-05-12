# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1100.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 107 |
| ALERT2 | 105 |
| ALERT2_SKIP | 56 |
| ALERT3 | 267 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 12 |
| ENTRY2 | 112 |
| PARTIAL | 16 |
| TARGET_HIT | 9 |
| STOP_HIT | 118 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 81
- **Target hits / Stop hits / Partials:** 9 / 115 / 16
- **Avg / median % per leg:** 0.56% / -0.81%
- **Sum % (uncompounded):** 78.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 27 | 44.3% | 7 | 51 | 3 | 0.87% | 53.1% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 0 | 6 | 3 | 2.08% | 18.8% |
| BUY @ 3rd Alert (retest2) | 52 | 21 | 40.4% | 7 | 45 | 0 | 0.66% | 34.3% |
| SELL (all) | 79 | 32 | 40.5% | 2 | 64 | 13 | 0.33% | 25.9% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 6 | 0 | -2.07% | -12.4% |
| SELL @ 3rd Alert (retest2) | 73 | 30 | 41.1% | 2 | 58 | 13 | 0.52% | 38.3% |
| retest1 (combined) | 15 | 8 | 53.3% | 0 | 12 | 3 | 0.42% | 6.3% |
| retest2 (combined) | 125 | 51 | 40.8% | 9 | 103 | 13 | 0.58% | 72.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 573.60 | 564.95 | 564.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 581.70 | 568.30 | 566.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 574.85 | 576.48 | 572.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 10:00:00 | 574.85 | 576.48 | 572.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 593.90 | 595.60 | 592.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 593.90 | 595.60 | 592.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 594.00 | 595.28 | 592.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 600.25 | 595.28 | 592.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 606.90 | 608.84 | 609.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 606.90 | 608.84 | 609.07 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 612.00 | 609.47 | 609.34 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 606.00 | 609.68 | 609.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 604.70 | 608.18 | 609.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 606.85 | 605.84 | 607.46 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 594.90 | 604.05 | 606.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 607.35 | 604.70 | 606.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 607.35 | 604.70 | 606.37 | SL hit (close>ema400) qty=1.00 sl=606.37 alert=retest1 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 632.15 | 609.74 | 608.01 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 577.30 | 611.85 | 612.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 544.00 | 598.28 | 606.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 585.35 | 554.25 | 565.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 585.35 | 554.25 | 565.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 585.35 | 554.25 | 565.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 585.35 | 554.25 | 565.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 589.50 | 561.30 | 568.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 589.50 | 561.30 | 568.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 598.95 | 576.66 | 574.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 614.45 | 584.22 | 577.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 601.15 | 602.08 | 594.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:00:00 | 601.15 | 602.08 | 594.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 599.90 | 601.27 | 596.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:30:00 | 601.45 | 601.38 | 597.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:00:00 | 601.80 | 601.38 | 597.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:30:00 | 601.90 | 601.49 | 597.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 661.60 | 646.76 | 639.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 658.50 | 660.70 | 660.88 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 670.00 | 662.56 | 661.71 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 09:15:00 | 658.15 | 661.75 | 661.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 11:15:00 | 652.00 | 659.13 | 660.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 668.75 | 655.34 | 657.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 668.75 | 655.34 | 657.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 668.75 | 655.34 | 657.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 668.75 | 655.34 | 657.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 669.10 | 658.09 | 658.58 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 664.00 | 659.28 | 659.08 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 647.95 | 657.47 | 658.72 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 660.05 | 656.34 | 656.31 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 12:15:00 | 650.40 | 655.17 | 655.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 648.50 | 652.63 | 654.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 10:15:00 | 654.35 | 652.97 | 654.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 10:15:00 | 654.35 | 652.97 | 654.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 654.35 | 652.97 | 654.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 654.35 | 652.97 | 654.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 653.60 | 653.10 | 654.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:30:00 | 654.00 | 653.10 | 654.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 651.80 | 652.84 | 654.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:30:00 | 655.20 | 652.84 | 654.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 655.30 | 653.33 | 654.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 655.30 | 653.33 | 654.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 657.35 | 654.14 | 654.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 657.35 | 654.14 | 654.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 15:15:00 | 660.60 | 655.43 | 654.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 10:15:00 | 671.85 | 659.52 | 656.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 13:15:00 | 669.25 | 670.16 | 665.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 14:00:00 | 669.25 | 670.16 | 665.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 663.35 | 668.80 | 665.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:45:00 | 664.95 | 668.80 | 665.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 663.50 | 667.74 | 665.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 669.95 | 667.74 | 665.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:00:00 | 666.40 | 666.43 | 665.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:45:00 | 664.95 | 667.21 | 667.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 14:15:00 | 665.00 | 666.77 | 667.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 14:15:00 | 665.00 | 666.77 | 667.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 15:15:00 | 663.00 | 666.01 | 666.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 10:15:00 | 668.10 | 666.43 | 666.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 10:15:00 | 668.10 | 666.43 | 666.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 668.10 | 666.43 | 666.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 667.50 | 666.43 | 666.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 670.05 | 667.15 | 667.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 675.15 | 670.01 | 668.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 669.90 | 669.99 | 668.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 10:15:00 | 669.90 | 669.99 | 668.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 669.90 | 669.99 | 668.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 669.90 | 669.99 | 668.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 663.30 | 668.65 | 668.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 663.30 | 668.65 | 668.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 666.40 | 668.20 | 668.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 662.25 | 668.20 | 668.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 662.00 | 666.96 | 667.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 12:15:00 | 659.90 | 663.96 | 665.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 661.00 | 645.97 | 651.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 661.00 | 645.97 | 651.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 661.00 | 645.97 | 651.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 664.90 | 645.97 | 651.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 659.40 | 648.66 | 652.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 662.50 | 648.66 | 652.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 647.50 | 648.96 | 651.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:30:00 | 651.20 | 648.96 | 651.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 647.00 | 645.61 | 649.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 649.20 | 645.61 | 649.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 634.05 | 632.51 | 639.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 633.45 | 632.51 | 639.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 633.50 | 632.71 | 638.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 632.65 | 632.71 | 638.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:15:00 | 632.55 | 633.13 | 638.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 632.10 | 632.93 | 637.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:45:00 | 632.70 | 632.88 | 637.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 635.00 | 633.24 | 636.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-25 11:15:00 | 642.60 | 635.83 | 637.15 | SL hit (close>static) qty=1.00 sl=640.90 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 13:15:00 | 644.50 | 638.65 | 638.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 662.65 | 643.66 | 640.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 651.00 | 651.50 | 646.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 662.35 | 651.50 | 646.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 650.85 | 654.02 | 650.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:30:00 | 650.90 | 654.02 | 650.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 652.00 | 653.62 | 650.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 647.95 | 653.62 | 650.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 654.75 | 653.84 | 651.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 654.75 | 653.84 | 651.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 649.90 | 653.38 | 651.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 649.90 | 653.38 | 651.47 | SL hit (close<ema400) qty=1.00 sl=651.47 alert=retest1 |

### Cycle 20 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 646.90 | 650.13 | 650.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 15:15:00 | 643.75 | 648.14 | 649.09 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 666.95 | 651.90 | 650.72 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 641.75 | 652.54 | 653.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 641.00 | 650.23 | 652.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 619.05 | 615.01 | 628.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:30:00 | 603.90 | 613.25 | 622.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 09:45:00 | 609.15 | 610.42 | 619.50 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 619.05 | 612.15 | 619.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 619.05 | 612.15 | 619.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 626.70 | 615.06 | 620.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 626.70 | 615.06 | 620.12 | SL hit (close>ema400) qty=1.00 sl=620.12 alert=retest1 |

### Cycle 23 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 635.60 | 623.60 | 623.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 643.05 | 629.33 | 625.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 15:15:00 | 639.00 | 640.50 | 634.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:15:00 | 647.70 | 640.50 | 634.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 13:15:00 | 645.00 | 645.45 | 639.04 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 650.25 | 646.41 | 640.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:30:00 | 645.95 | 646.41 | 640.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 644.00 | 648.80 | 643.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 644.00 | 648.80 | 643.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 641.95 | 647.43 | 643.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 641.95 | 647.43 | 643.49 | SL hit (close<ema400) qty=1.00 sl=643.49 alert=retest1 |

### Cycle 24 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 639.25 | 646.14 | 646.20 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 655.90 | 645.68 | 645.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 663.40 | 650.84 | 647.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 12:15:00 | 706.85 | 707.71 | 692.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 13:00:00 | 706.85 | 707.71 | 692.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 703.00 | 706.03 | 695.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 714.05 | 706.03 | 695.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:00:00 | 708.00 | 711.34 | 710.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 13:15:00 | 706.35 | 710.34 | 710.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 706.35 | 710.34 | 710.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 699.70 | 707.89 | 709.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 703.70 | 703.66 | 706.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 703.70 | 703.66 | 706.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 703.70 | 703.66 | 706.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 707.45 | 703.66 | 706.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 706.00 | 704.13 | 706.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:00:00 | 706.00 | 704.13 | 706.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 711.00 | 705.50 | 706.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:30:00 | 710.70 | 705.50 | 706.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 704.00 | 705.20 | 706.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:15:00 | 700.35 | 704.96 | 706.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:30:00 | 703.70 | 702.93 | 704.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 14:15:00 | 668.51 | 677.25 | 684.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 680.45 | 676.82 | 682.38 | SL hit (close>ema200) qty=0.50 sl=676.82 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 686.95 | 682.67 | 682.66 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 12:15:00 | 682.05 | 682.55 | 682.60 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 685.00 | 683.04 | 682.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 689.00 | 684.23 | 683.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 688.35 | 688.73 | 686.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 12:00:00 | 688.35 | 688.73 | 686.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 686.10 | 688.21 | 686.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:00:00 | 686.10 | 688.21 | 686.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 686.95 | 687.95 | 686.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 686.95 | 687.95 | 686.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 689.90 | 688.34 | 686.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 692.20 | 688.14 | 686.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 14:15:00 | 697.65 | 689.67 | 688.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 674.65 | 689.02 | 690.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 674.65 | 689.02 | 690.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 668.90 | 682.75 | 687.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 677.45 | 676.43 | 682.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 677.45 | 676.43 | 682.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 681.40 | 677.35 | 682.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 682.10 | 677.35 | 682.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 684.40 | 678.76 | 682.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:15:00 | 687.10 | 678.76 | 682.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 683.05 | 679.62 | 682.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:45:00 | 676.00 | 680.03 | 682.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:00:00 | 679.50 | 678.96 | 681.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:15:00 | 680.00 | 679.38 | 681.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:15:00 | 680.75 | 679.70 | 681.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 679.95 | 679.75 | 680.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:15:00 | 677.60 | 679.75 | 680.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:00:00 | 676.00 | 679.00 | 680.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 12:30:00 | 675.00 | 677.31 | 678.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 676.80 | 677.06 | 677.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 680.90 | 677.83 | 678.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 680.90 | 677.83 | 678.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 675.00 | 677.26 | 677.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 14:30:00 | 673.00 | 676.05 | 677.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 09:45:00 | 673.20 | 675.06 | 676.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 645.52 | 662.04 | 664.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 646.00 | 662.04 | 664.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 646.71 | 662.04 | 664.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 642.20 | 657.12 | 661.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 643.72 | 657.12 | 661.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 642.20 | 657.12 | 661.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 641.25 | 657.12 | 661.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 642.96 | 657.12 | 661.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 660.00 | 657.07 | 660.81 | SL hit (close>ema200) qty=0.50 sl=657.07 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 681.65 | 664.99 | 662.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 687.00 | 675.14 | 668.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 683.30 | 683.66 | 675.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 11:00:00 | 683.30 | 683.66 | 675.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 680.20 | 687.35 | 684.39 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 670.05 | 681.41 | 682.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 668.30 | 675.92 | 679.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 660.35 | 656.64 | 663.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 660.35 | 656.64 | 663.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 660.35 | 656.64 | 663.01 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 681.65 | 667.28 | 666.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 15:15:00 | 686.00 | 671.02 | 667.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 657.60 | 668.34 | 667.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 657.60 | 668.34 | 667.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 657.60 | 668.34 | 667.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 657.60 | 668.34 | 667.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 657.60 | 666.19 | 666.29 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 11:15:00 | 671.00 | 664.88 | 664.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 12:15:00 | 680.00 | 667.91 | 666.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 658.40 | 666.61 | 665.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 658.40 | 666.61 | 665.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 658.40 | 666.61 | 665.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 657.40 | 666.61 | 665.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 661.05 | 665.50 | 665.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 647.40 | 661.88 | 663.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 15:15:00 | 636.95 | 633.89 | 641.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 09:15:00 | 651.30 | 633.89 | 641.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 644.25 | 635.96 | 641.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:45:00 | 643.35 | 643.02 | 643.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:45:00 | 643.15 | 642.95 | 643.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 660.70 | 646.12 | 644.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 660.70 | 646.12 | 644.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 10:15:00 | 663.40 | 649.57 | 646.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 14:15:00 | 717.00 | 721.23 | 707.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 15:00:00 | 717.00 | 721.23 | 707.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 709.80 | 718.59 | 708.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 709.80 | 718.59 | 708.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 706.20 | 716.11 | 708.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 706.20 | 716.11 | 708.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 705.85 | 714.06 | 708.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 705.85 | 714.06 | 708.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 704.85 | 712.22 | 707.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:00:00 | 704.85 | 712.22 | 707.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 709.85 | 709.98 | 707.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 692.80 | 709.98 | 707.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 713.45 | 710.67 | 708.25 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 699.00 | 706.79 | 706.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 692.20 | 701.21 | 703.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 678.30 | 677.14 | 687.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 678.30 | 677.14 | 687.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 690.35 | 678.81 | 685.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 690.35 | 678.81 | 685.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 684.65 | 679.98 | 685.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 675.50 | 683.51 | 686.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:15:00 | 678.70 | 684.13 | 686.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 677.85 | 679.69 | 682.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 15:00:00 | 679.85 | 682.48 | 682.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 680.60 | 682.11 | 682.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 678.70 | 682.11 | 682.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 675.20 | 680.73 | 682.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:30:00 | 673.35 | 678.74 | 680.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 672.00 | 677.23 | 679.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 689.25 | 679.67 | 678.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 689.25 | 679.67 | 678.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 694.70 | 682.68 | 679.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 13:15:00 | 704.90 | 705.15 | 698.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 13:30:00 | 704.15 | 705.15 | 698.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 702.50 | 704.62 | 698.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 15:00:00 | 702.50 | 704.62 | 698.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 712.55 | 706.08 | 700.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 713.60 | 708.14 | 702.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 10:45:00 | 713.70 | 714.00 | 708.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 11:15:00 | 714.40 | 723.06 | 717.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 13:45:00 | 716.25 | 719.71 | 717.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 718.45 | 719.45 | 717.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 707.95 | 715.31 | 716.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 707.95 | 715.31 | 716.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 706.85 | 712.43 | 714.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 712.20 | 702.87 | 706.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 712.20 | 702.87 | 706.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 712.20 | 702.87 | 706.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 712.20 | 702.87 | 706.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 704.80 | 703.26 | 706.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 698.95 | 703.28 | 705.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 15:15:00 | 664.00 | 681.05 | 690.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 679.00 | 677.75 | 682.83 | SL hit (close>ema200) qty=0.50 sl=677.75 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 697.55 | 684.12 | 683.92 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 673.45 | 684.25 | 684.86 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 697.95 | 683.67 | 682.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 14:15:00 | 704.75 | 698.59 | 693.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 12:15:00 | 695.65 | 700.51 | 696.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 12:15:00 | 695.65 | 700.51 | 696.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 695.65 | 700.51 | 696.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:45:00 | 696.25 | 700.51 | 696.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 696.15 | 699.64 | 696.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 695.80 | 699.64 | 696.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 695.85 | 698.88 | 696.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 695.85 | 698.88 | 696.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 696.85 | 698.47 | 696.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 698.10 | 698.47 | 696.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 694.60 | 697.70 | 696.48 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 689.75 | 694.55 | 695.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 685.10 | 691.03 | 693.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 687.50 | 686.61 | 689.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 15:00:00 | 687.50 | 686.61 | 689.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 688.90 | 686.65 | 689.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 688.90 | 686.65 | 689.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 687.15 | 686.75 | 689.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:15:00 | 693.55 | 686.75 | 689.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 683.20 | 686.04 | 688.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 693.85 | 686.04 | 688.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 686.85 | 682.26 | 685.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 687.30 | 682.26 | 685.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 684.10 | 682.63 | 685.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 687.25 | 682.63 | 685.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 681.00 | 682.30 | 684.74 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 692.80 | 686.16 | 685.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 13:15:00 | 707.05 | 694.27 | 689.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 15:15:00 | 704.25 | 707.47 | 701.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 09:15:00 | 703.55 | 707.47 | 701.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 706.00 | 707.18 | 701.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:45:00 | 715.75 | 709.72 | 704.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 713.05 | 713.70 | 708.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-10 10:15:00 | 787.33 | 744.66 | 728.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 765.35 | 775.41 | 775.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 760.20 | 770.56 | 773.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 763.00 | 752.94 | 756.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 763.00 | 752.94 | 756.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 763.00 | 752.94 | 756.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 768.50 | 752.94 | 756.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 763.80 | 755.11 | 757.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 755.25 | 755.53 | 757.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:45:00 | 756.75 | 749.24 | 750.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 756.65 | 751.66 | 751.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 09:15:00 | 756.65 | 751.66 | 751.31 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 12:15:00 | 744.55 | 750.00 | 750.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 13:15:00 | 741.00 | 748.20 | 749.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 744.05 | 738.21 | 742.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 744.05 | 738.21 | 742.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 744.05 | 738.21 | 742.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 744.05 | 738.21 | 742.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 731.55 | 736.88 | 741.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 727.50 | 734.71 | 739.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 742.90 | 731.40 | 730.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 742.90 | 731.40 | 730.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 745.25 | 734.17 | 731.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 740.70 | 743.02 | 737.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 11:00:00 | 740.70 | 743.02 | 737.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 748.65 | 756.38 | 752.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 748.65 | 756.38 | 752.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 729.30 | 750.97 | 750.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 729.30 | 750.97 | 750.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 724.10 | 745.59 | 747.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 709.95 | 723.84 | 732.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 649.50 | 641.95 | 652.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:00:00 | 649.50 | 641.95 | 652.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 657.15 | 644.99 | 653.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:30:00 | 654.25 | 644.99 | 653.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 662.75 | 648.54 | 654.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:45:00 | 661.80 | 648.54 | 654.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 661.00 | 655.46 | 656.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:45:00 | 661.60 | 655.46 | 656.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 678.40 | 660.94 | 658.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 10:15:00 | 682.40 | 665.24 | 660.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 675.60 | 683.11 | 678.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 675.60 | 683.11 | 678.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 675.60 | 683.11 | 678.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 673.65 | 683.11 | 678.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 683.20 | 683.13 | 678.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:15:00 | 684.10 | 683.30 | 678.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:00:00 | 685.05 | 683.91 | 680.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 665.35 | 680.29 | 679.72 | SL hit (close<static) qty=1.00 sl=675.30 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 664.20 | 677.07 | 678.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 660.25 | 673.71 | 676.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 653.85 | 650.58 | 658.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:45:00 | 653.00 | 650.58 | 658.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 657.15 | 651.90 | 658.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 652.30 | 655.92 | 659.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 634.10 | 655.16 | 658.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 619.68 | 638.67 | 648.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 602.39 | 623.02 | 637.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 11:15:00 | 587.07 | 610.22 | 628.81 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 583.95 | 580.13 | 579.77 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 10:15:00 | 572.55 | 579.52 | 579.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 11:15:00 | 565.15 | 576.65 | 578.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 582.35 | 575.44 | 577.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 14:15:00 | 582.35 | 575.44 | 577.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 582.35 | 575.44 | 577.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 582.35 | 575.44 | 577.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 583.00 | 576.95 | 577.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 566.90 | 576.95 | 577.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 585.05 | 572.78 | 573.81 | SL hit (close>static) qty=1.00 sl=584.95 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 580.35 | 574.37 | 574.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 582.00 | 575.89 | 574.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 606.70 | 613.63 | 599.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:30:00 | 616.50 | 613.63 | 599.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 599.20 | 610.74 | 599.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 599.20 | 610.74 | 599.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 599.10 | 608.41 | 599.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 599.10 | 608.41 | 599.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 600.25 | 606.78 | 599.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 607.00 | 605.11 | 600.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 586.80 | 598.31 | 599.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 586.80 | 598.31 | 599.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 585.30 | 595.71 | 597.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 545.80 | 543.53 | 559.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 545.80 | 543.53 | 559.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 553.05 | 550.97 | 557.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 544.40 | 554.65 | 557.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 14:15:00 | 517.18 | 531.45 | 543.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 521.60 | 520.97 | 530.70 | SL hit (close>ema200) qty=0.50 sl=520.97 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 532.45 | 523.92 | 523.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 536.90 | 526.51 | 524.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 529.20 | 531.54 | 528.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 529.20 | 531.54 | 528.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 529.20 | 531.54 | 528.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 529.20 | 531.54 | 528.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 533.80 | 531.99 | 529.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 534.50 | 531.99 | 529.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 535.05 | 535.99 | 532.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 530.65 | 535.99 | 532.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 530.00 | 534.79 | 532.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 524.50 | 534.79 | 532.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 524.55 | 532.75 | 531.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 514.70 | 532.75 | 531.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 541.60 | 534.37 | 532.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 12:15:00 | 542.90 | 534.37 | 532.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 529.35 | 533.76 | 532.63 | SL hit (close<static) qty=1.00 sl=531.30 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 530.00 | 533.53 | 533.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 516.00 | 524.96 | 528.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 515.45 | 511.98 | 516.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 515.45 | 511.98 | 516.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 515.45 | 511.98 | 516.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 515.45 | 511.98 | 516.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 517.60 | 513.10 | 516.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 521.70 | 513.10 | 516.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 520.25 | 514.53 | 517.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:30:00 | 523.75 | 514.53 | 517.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 518.85 | 515.39 | 517.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 517.25 | 516.06 | 517.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 517.85 | 516.41 | 517.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:30:00 | 517.00 | 517.11 | 517.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 521.45 | 518.44 | 518.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 521.45 | 518.44 | 518.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 536.90 | 523.36 | 520.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 530.65 | 534.75 | 531.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 530.65 | 534.75 | 531.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 530.65 | 534.75 | 531.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 530.20 | 534.75 | 531.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 531.60 | 534.12 | 531.24 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 525.00 | 530.11 | 530.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 10:15:00 | 521.15 | 528.32 | 529.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 520.00 | 519.65 | 523.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 11:00:00 | 520.00 | 519.65 | 523.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 529.95 | 521.71 | 524.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 529.95 | 521.71 | 524.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 540.10 | 525.39 | 525.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 540.10 | 525.39 | 525.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 536.25 | 527.56 | 526.78 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 519.40 | 526.96 | 527.08 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 532.80 | 527.17 | 526.70 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 525.10 | 526.46 | 526.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 15:15:00 | 523.95 | 525.95 | 526.28 | Break + close below crossover candle low |

### Cycle 65 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 537.20 | 528.20 | 527.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 538.85 | 531.73 | 529.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 558.40 | 558.66 | 553.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 558.40 | 558.66 | 553.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 553.65 | 557.65 | 553.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 576.40 | 557.65 | 553.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 12:15:00 | 578.50 | 585.76 | 586.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 578.50 | 585.76 | 586.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 575.30 | 581.13 | 583.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 573.00 | 572.01 | 576.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:45:00 | 574.55 | 572.01 | 576.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 575.30 | 572.27 | 575.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 575.30 | 572.27 | 575.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 580.00 | 573.82 | 576.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 580.00 | 573.82 | 576.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 593.45 | 577.74 | 577.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 595.00 | 581.19 | 579.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 582.35 | 582.77 | 580.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 13:00:00 | 582.35 | 582.77 | 580.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 574.45 | 581.65 | 580.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 574.45 | 581.65 | 580.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 567.45 | 578.81 | 579.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 565.75 | 574.87 | 577.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 556.60 | 541.55 | 550.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 11:15:00 | 556.60 | 541.55 | 550.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 556.60 | 541.55 | 550.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 556.60 | 541.55 | 550.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 565.85 | 546.41 | 551.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 565.85 | 546.41 | 551.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 560.00 | 553.54 | 554.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 543.00 | 553.54 | 554.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:45:00 | 555.20 | 545.84 | 546.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 13:15:00 | 551.40 | 546.95 | 546.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 551.40 | 546.95 | 546.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 567.00 | 552.09 | 549.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 10:15:00 | 654.45 | 654.75 | 637.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:45:00 | 653.30 | 654.75 | 637.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 663.90 | 667.48 | 659.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 663.90 | 667.48 | 659.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 637.75 | 661.54 | 657.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 637.75 | 661.54 | 657.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 634.50 | 656.13 | 655.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 632.70 | 656.13 | 655.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 642.95 | 653.49 | 654.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 10:15:00 | 633.00 | 642.12 | 647.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 639.35 | 638.05 | 642.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 639.35 | 638.05 | 642.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 639.35 | 638.05 | 642.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 630.65 | 636.43 | 641.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:00:00 | 631.05 | 635.36 | 640.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 646.90 | 633.38 | 637.28 | SL hit (close>static) qty=1.00 sl=644.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 12:15:00 | 650.35 | 639.73 | 639.45 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 633.00 | 639.58 | 639.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 631.05 | 636.92 | 638.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 638.70 | 634.42 | 636.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 638.70 | 634.42 | 636.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 638.70 | 634.42 | 636.12 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 645.15 | 638.16 | 637.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 651.00 | 643.04 | 640.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 642.05 | 646.26 | 643.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 642.05 | 646.26 | 643.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 642.05 | 646.26 | 643.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 642.95 | 646.26 | 643.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 638.80 | 644.77 | 642.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 638.80 | 644.77 | 642.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 641.50 | 644.11 | 642.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 14:30:00 | 645.75 | 642.33 | 641.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 15:15:00 | 636.00 | 641.07 | 641.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 636.00 | 641.07 | 641.36 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 653.75 | 643.98 | 642.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 11:15:00 | 656.70 | 646.52 | 643.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 648.45 | 650.20 | 647.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 648.45 | 650.20 | 647.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 648.45 | 650.20 | 647.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:45:00 | 647.30 | 650.20 | 647.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 645.20 | 649.20 | 646.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 645.20 | 649.20 | 646.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 649.90 | 649.34 | 647.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 653.15 | 650.10 | 647.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 14:00:00 | 655.00 | 651.08 | 648.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 639.70 | 648.80 | 647.61 | SL hit (close<static) qty=1.00 sl=644.20 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 633.60 | 644.31 | 645.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 617.50 | 638.95 | 643.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 668.95 | 637.90 | 639.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 668.95 | 637.90 | 639.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 668.95 | 637.90 | 639.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 671.10 | 637.90 | 639.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 669.55 | 644.23 | 642.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 688.00 | 652.98 | 646.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 695.30 | 696.08 | 681.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:45:00 | 696.00 | 696.08 | 681.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 744.00 | 740.50 | 728.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 729.15 | 740.50 | 728.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 743.65 | 743.48 | 737.78 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 726.50 | 735.51 | 736.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 721.80 | 732.77 | 734.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 724.85 | 724.48 | 729.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 724.85 | 724.48 | 729.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 730.20 | 725.62 | 729.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 730.95 | 725.62 | 729.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 723.85 | 725.27 | 728.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 720.10 | 725.27 | 728.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 721.00 | 726.67 | 727.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 12:15:00 | 736.00 | 729.49 | 728.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 736.00 | 729.49 | 728.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 741.00 | 732.67 | 730.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 733.00 | 737.94 | 734.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 733.00 | 737.94 | 734.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 733.00 | 737.94 | 734.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 733.00 | 737.94 | 734.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 735.60 | 737.47 | 734.65 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 732.95 | 733.86 | 733.90 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 742.10 | 735.33 | 734.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 754.45 | 742.53 | 739.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 738.45 | 745.26 | 741.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 738.45 | 745.26 | 741.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 738.45 | 745.26 | 741.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 738.45 | 745.26 | 741.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 735.00 | 743.20 | 741.18 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 730.00 | 739.17 | 739.61 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 746.20 | 739.57 | 739.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 765.15 | 748.44 | 743.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 788.95 | 798.39 | 784.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 793.50 | 798.39 | 784.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 779.95 | 794.70 | 784.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 779.95 | 794.70 | 784.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 780.90 | 791.94 | 784.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 781.55 | 791.94 | 784.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 762.85 | 780.24 | 780.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 756.60 | 772.63 | 776.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 735.15 | 732.05 | 743.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 735.15 | 732.05 | 743.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 735.65 | 732.32 | 739.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 735.65 | 732.32 | 739.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 740.60 | 733.98 | 739.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 740.60 | 733.98 | 739.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 745.45 | 736.27 | 740.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 745.45 | 736.27 | 740.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 745.00 | 738.02 | 740.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 743.05 | 738.02 | 740.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 750.85 | 742.74 | 742.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 13:15:00 | 763.05 | 750.94 | 746.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 746.25 | 753.65 | 749.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 746.25 | 753.65 | 749.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 746.25 | 753.65 | 749.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 746.25 | 753.65 | 749.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 748.70 | 752.66 | 749.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:15:00 | 742.80 | 752.66 | 749.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 745.60 | 751.25 | 749.24 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 742.20 | 746.88 | 747.48 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 753.80 | 748.26 | 748.06 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 740.05 | 746.79 | 747.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 14:15:00 | 734.85 | 743.30 | 745.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 752.10 | 743.40 | 745.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 752.10 | 743.40 | 745.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 752.10 | 743.40 | 745.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 752.10 | 743.40 | 745.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 757.05 | 746.13 | 746.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 757.05 | 746.13 | 746.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 756.80 | 748.26 | 747.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 12:15:00 | 761.30 | 750.87 | 748.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 781.05 | 787.60 | 781.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 781.05 | 787.60 | 781.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 781.05 | 787.60 | 781.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 781.05 | 787.60 | 781.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 779.50 | 785.98 | 781.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 782.85 | 785.98 | 781.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 789.80 | 787.15 | 782.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 795.65 | 800.30 | 800.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 795.65 | 800.30 | 800.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 791.00 | 798.44 | 799.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 796.95 | 794.35 | 796.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 10:15:00 | 796.95 | 794.35 | 796.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 796.95 | 794.35 | 796.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 796.95 | 794.35 | 796.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 783.90 | 792.26 | 795.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 780.05 | 792.26 | 795.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 781.25 | 782.77 | 788.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 801.95 | 790.06 | 788.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 801.95 | 790.06 | 788.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 804.50 | 792.95 | 790.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 799.40 | 808.92 | 803.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 799.40 | 808.92 | 803.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 799.40 | 808.92 | 803.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 799.40 | 808.92 | 803.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 807.00 | 808.53 | 803.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:15:00 | 807.55 | 808.53 | 803.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 807.15 | 806.51 | 804.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:45:00 | 812.95 | 807.99 | 806.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 848.50 | 852.96 | 853.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 848.50 | 852.96 | 853.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 838.25 | 847.64 | 850.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 822.35 | 818.05 | 827.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 822.35 | 818.05 | 827.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 810.00 | 802.82 | 809.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 815.95 | 802.82 | 809.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 818.15 | 805.88 | 810.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 824.65 | 805.88 | 810.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 816.65 | 808.04 | 810.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 820.50 | 808.04 | 810.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 820.15 | 812.40 | 812.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 823.60 | 816.01 | 814.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 10:15:00 | 815.35 | 817.22 | 815.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 10:15:00 | 815.35 | 817.22 | 815.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 815.35 | 817.22 | 815.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 819.00 | 818.18 | 816.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 811.05 | 817.37 | 816.08 | SL hit (close<static) qty=1.00 sl=814.05 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 811.70 | 815.22 | 815.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 802.00 | 811.10 | 813.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 810.00 | 808.59 | 811.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 810.00 | 808.59 | 811.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 810.00 | 808.59 | 811.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 806.90 | 808.59 | 811.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 808.45 | 807.96 | 810.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 808.45 | 807.96 | 810.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 811.95 | 808.59 | 810.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 811.95 | 808.59 | 810.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 813.00 | 809.47 | 810.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 820.30 | 809.47 | 810.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 821.00 | 811.78 | 811.41 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 810.40 | 813.96 | 814.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 807.10 | 812.23 | 813.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 810.60 | 810.20 | 811.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 810.60 | 810.20 | 811.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 810.60 | 810.20 | 811.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 810.60 | 810.20 | 811.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 804.80 | 809.12 | 811.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 803.05 | 809.12 | 811.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 802.50 | 808.71 | 810.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:45:00 | 803.00 | 805.62 | 808.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 803.90 | 805.85 | 808.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 807.90 | 806.26 | 808.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 807.50 | 806.26 | 808.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 806.00 | 806.21 | 807.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 813.30 | 808.74 | 808.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 813.30 | 808.74 | 808.55 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 803.00 | 807.70 | 808.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 799.45 | 806.05 | 807.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 809.75 | 805.97 | 807.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 809.75 | 805.97 | 807.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 809.75 | 805.97 | 807.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:00:00 | 802.60 | 805.29 | 806.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 14:15:00 | 800.50 | 800.30 | 802.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 803.95 | 801.25 | 802.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 827.00 | 806.83 | 804.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 827.00 | 806.83 | 804.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 829.10 | 811.28 | 806.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 15:15:00 | 863.00 | 863.57 | 849.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 856.95 | 863.57 | 849.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 863.00 | 867.20 | 859.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 875.55 | 867.20 | 859.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 858.20 | 865.38 | 863.92 | SL hit (close<static) qty=1.00 sl=858.40 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 852.25 | 862.75 | 862.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 841.65 | 856.82 | 860.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 796.85 | 794.70 | 807.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 797.10 | 794.70 | 807.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 814.60 | 797.38 | 799.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 813.90 | 797.38 | 799.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 810.15 | 799.93 | 800.41 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 814.55 | 802.86 | 801.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 815.95 | 807.50 | 804.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 810.00 | 810.89 | 807.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:00:00 | 810.00 | 810.89 | 807.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 810.05 | 810.90 | 807.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 811.10 | 810.90 | 807.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 811.95 | 811.48 | 808.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 818.30 | 811.48 | 808.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 801.10 | 809.00 | 808.29 | SL hit (close<static) qty=1.00 sl=804.05 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 798.40 | 806.88 | 807.39 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 807.15 | 804.97 | 804.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 823.50 | 808.67 | 806.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 815.00 | 815.03 | 810.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 815.00 | 815.03 | 810.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 816.90 | 818.96 | 815.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 816.90 | 818.96 | 815.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 824.70 | 828.86 | 824.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 824.70 | 828.86 | 824.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 823.20 | 827.73 | 824.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 823.20 | 827.73 | 824.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 827.05 | 827.59 | 824.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:45:00 | 829.10 | 827.88 | 825.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 830.80 | 828.47 | 825.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 834.75 | 839.20 | 839.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 834.75 | 839.20 | 839.58 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 849.90 | 840.98 | 840.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 852.95 | 846.06 | 843.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 838.05 | 846.94 | 845.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 838.05 | 846.94 | 845.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 838.05 | 846.94 | 845.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 838.05 | 846.94 | 845.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 840.00 | 845.55 | 845.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 847.15 | 845.55 | 845.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 873.55 | 878.92 | 878.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 873.55 | 878.92 | 878.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 869.80 | 874.70 | 876.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 871.25 | 860.42 | 863.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 871.25 | 860.42 | 863.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 871.25 | 860.42 | 863.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 871.25 | 860.42 | 863.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 870.65 | 862.47 | 864.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 870.65 | 862.47 | 864.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 870.05 | 866.38 | 865.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 15:15:00 | 873.00 | 868.42 | 867.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 883.45 | 888.79 | 882.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 883.45 | 888.79 | 882.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 883.45 | 888.79 | 882.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 882.30 | 888.79 | 882.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 878.75 | 886.08 | 882.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 878.75 | 886.08 | 882.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 876.00 | 884.07 | 881.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 882.00 | 884.07 | 881.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 882.55 | 883.76 | 881.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 885.95 | 885.07 | 882.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-23 09:15:00 | 974.55 | 950.15 | 932.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 921.05 | 934.17 | 935.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 914.45 | 926.18 | 930.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 870.40 | 863.74 | 876.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 15:00:00 | 870.40 | 863.74 | 876.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 872.00 | 865.40 | 876.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 880.65 | 865.40 | 876.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 885.65 | 869.45 | 877.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 890.50 | 869.45 | 877.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 873.60 | 870.28 | 877.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 868.95 | 869.54 | 876.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 872.75 | 870.42 | 874.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 884.35 | 874.96 | 874.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 884.35 | 874.96 | 874.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 886.65 | 877.30 | 875.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 873.00 | 876.44 | 875.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 12:15:00 | 873.00 | 876.44 | 875.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 873.00 | 876.44 | 875.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 873.00 | 876.44 | 875.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 874.40 | 876.03 | 875.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 14:30:00 | 877.50 | 877.27 | 875.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 882.20 | 877.27 | 875.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 868.15 | 877.12 | 875.97 | SL hit (close<static) qty=1.00 sl=872.05 alert=retest2 |

### Cycle 110 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 868.95 | 874.05 | 874.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 866.15 | 871.82 | 873.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 863.50 | 862.67 | 867.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 862.65 | 862.67 | 867.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 863.65 | 857.96 | 862.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 863.65 | 857.96 | 862.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 870.15 | 860.40 | 863.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 870.15 | 860.40 | 863.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 870.60 | 862.44 | 864.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 869.90 | 862.44 | 864.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 892.85 | 869.90 | 867.37 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 868.00 | 875.64 | 876.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 866.00 | 873.71 | 875.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 864.35 | 860.39 | 864.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 12:15:00 | 864.35 | 860.39 | 864.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 864.35 | 860.39 | 864.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 865.15 | 860.39 | 864.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 864.40 | 861.19 | 864.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 864.40 | 861.19 | 864.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 860.55 | 861.06 | 864.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:45:00 | 864.00 | 861.06 | 864.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 865.00 | 862.00 | 863.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 863.40 | 862.00 | 863.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 870.40 | 863.68 | 864.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 870.40 | 863.68 | 864.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 865.80 | 864.10 | 864.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:15:00 | 873.80 | 864.10 | 864.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 877.70 | 866.82 | 865.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 882.85 | 871.83 | 868.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 879.60 | 883.72 | 878.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 879.60 | 883.72 | 878.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 879.60 | 883.72 | 878.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 879.60 | 883.72 | 878.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 885.90 | 884.16 | 879.50 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 862.60 | 876.42 | 877.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 858.25 | 872.79 | 875.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 878.80 | 870.38 | 873.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 878.80 | 870.38 | 873.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 878.80 | 870.38 | 873.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 878.80 | 870.38 | 873.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 870.10 | 870.33 | 873.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 867.00 | 870.33 | 873.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 14:15:00 | 876.55 | 864.59 | 864.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 876.55 | 864.59 | 864.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 879.60 | 868.55 | 866.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 862.55 | 872.01 | 869.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 862.55 | 872.01 | 869.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 862.55 | 872.01 | 869.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 862.55 | 872.01 | 869.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 859.75 | 869.56 | 868.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 859.50 | 869.56 | 868.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 861.55 | 867.96 | 868.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 853.65 | 862.78 | 865.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 832.05 | 828.44 | 838.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 832.05 | 828.44 | 838.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 832.05 | 828.44 | 838.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 810.60 | 830.65 | 834.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 822.25 | 821.37 | 825.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 822.05 | 820.45 | 820.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 822.05 | 820.45 | 820.39 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 819.60 | 820.25 | 820.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 813.95 | 818.80 | 819.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 823.90 | 818.73 | 819.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 823.90 | 818.73 | 819.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 823.90 | 818.73 | 819.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 823.90 | 818.73 | 819.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 824.05 | 819.79 | 819.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 824.00 | 819.79 | 819.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 831.65 | 822.17 | 820.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 849.20 | 830.25 | 825.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 863.40 | 863.40 | 853.82 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 12:30:00 | 868.20 | 865.13 | 857.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 14:45:00 | 868.00 | 865.79 | 858.78 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 15:15:00 | 868.00 | 865.05 | 861.78 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 868.00 | 865.64 | 862.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 918.15 | 865.64 | 862.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 911.61 | 874.42 | 866.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 911.40 | 874.42 | 866.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 911.40 | 874.42 | 866.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 888.30 | 895.48 | 882.54 | SL hit (close<ema200) qty=0.50 sl=895.48 alert=retest1 |

### Cycle 120 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 880.00 | 890.98 | 892.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 876.45 | 884.47 | 888.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 877.40 | 875.55 | 880.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 877.40 | 875.55 | 880.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 875.75 | 868.11 | 870.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 875.75 | 868.11 | 870.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 877.50 | 869.98 | 871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 876.25 | 869.98 | 871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 880.00 | 873.69 | 873.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 882.20 | 876.76 | 874.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 875.60 | 876.53 | 874.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 875.60 | 876.53 | 874.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 875.60 | 876.53 | 874.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 875.60 | 876.53 | 874.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 877.90 | 876.80 | 875.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:00:00 | 880.20 | 877.04 | 875.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:30:00 | 880.70 | 877.64 | 876.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:30:00 | 881.60 | 878.85 | 876.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 882.75 | 889.27 | 889.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 882.75 | 889.27 | 889.33 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 890.35 | 889.48 | 889.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 892.95 | 890.27 | 889.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 12:15:00 | 890.65 | 896.64 | 894.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 12:15:00 | 890.65 | 896.64 | 894.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 890.65 | 896.64 | 894.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 890.65 | 896.64 | 894.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 892.50 | 895.82 | 893.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 892.50 | 895.82 | 893.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 894.20 | 895.49 | 893.94 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 877.95 | 891.93 | 892.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 861.45 | 880.16 | 885.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 864.05 | 863.94 | 872.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 864.05 | 863.94 | 872.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 864.05 | 863.94 | 872.50 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 887.15 | 876.16 | 875.31 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 863.00 | 874.00 | 874.96 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 887.00 | 877.18 | 876.13 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 862.70 | 878.41 | 879.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 858.95 | 874.52 | 877.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 848.15 | 842.32 | 852.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 848.15 | 842.32 | 852.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 848.15 | 842.32 | 852.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 851.50 | 842.32 | 852.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 839.85 | 841.36 | 846.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 833.85 | 840.09 | 845.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 834.90 | 838.75 | 844.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 838.65 | 827.28 | 825.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 838.65 | 827.28 | 825.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 847.85 | 834.91 | 829.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 875.00 | 876.54 | 861.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:15:00 | 867.85 | 876.54 | 861.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 861.55 | 873.54 | 861.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 859.35 | 873.54 | 861.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 875.00 | 873.83 | 862.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 875.00 | 873.83 | 862.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 851.00 | 869.45 | 862.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:45:00 | 853.95 | 869.45 | 862.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 861.00 | 867.76 | 862.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 15:00:00 | 865.20 | 867.25 | 862.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 863.20 | 865.30 | 862.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 865.15 | 865.44 | 862.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 846.80 | 861.71 | 861.03 | SL hit (close<static) qty=1.00 sl=850.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 849.25 | 859.22 | 859.96 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 876.00 | 861.36 | 860.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 926.00 | 874.29 | 866.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 13:15:00 | 921.20 | 923.35 | 906.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:45:00 | 920.00 | 923.35 | 906.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 911.50 | 921.82 | 916.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 911.50 | 921.82 | 916.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 911.25 | 919.70 | 915.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 907.55 | 919.70 | 915.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 916.25 | 916.83 | 915.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 912.00 | 916.83 | 915.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 919.70 | 917.40 | 915.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 929.90 | 917.92 | 916.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 924.90 | 920.56 | 918.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 12:15:00 | 913.95 | 921.03 | 920.00 | SL hit (close<static) qty=1.00 sl=915.10 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 909.80 | 918.79 | 919.07 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 930.85 | 919.85 | 919.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 937.05 | 923.29 | 920.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 936.30 | 954.22 | 945.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 936.30 | 954.22 | 945.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 936.30 | 954.22 | 945.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 938.75 | 954.22 | 945.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 939.85 | 951.35 | 945.01 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 926.00 | 940.26 | 941.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 915.75 | 935.36 | 938.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 934.85 | 929.77 | 933.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 934.85 | 929.77 | 933.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 934.85 | 929.77 | 933.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 940.70 | 929.77 | 933.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 929.85 | 929.79 | 933.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 934.45 | 929.79 | 933.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 924.95 | 927.25 | 930.96 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 942.00 | 932.50 | 932.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 946.90 | 935.38 | 933.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 960.00 | 964.57 | 954.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 960.00 | 964.57 | 954.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 957.20 | 963.10 | 954.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 956.00 | 963.10 | 954.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 948.20 | 960.12 | 953.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 947.85 | 960.12 | 953.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 942.85 | 956.67 | 952.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 942.85 | 956.67 | 952.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 951.75 | 952.29 | 951.53 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 949.10 | 950.97 | 951.02 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 956.10 | 951.78 | 951.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 962.15 | 954.08 | 952.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 972.60 | 975.62 | 970.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 972.60 | 975.62 | 970.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 972.60 | 975.62 | 970.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 971.75 | 975.62 | 970.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 964.95 | 974.79 | 971.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 964.95 | 974.79 | 971.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 962.70 | 972.37 | 970.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 965.75 | 972.37 | 970.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 964.65 | 971.42 | 970.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 964.65 | 971.42 | 970.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 10:15:00 | 954.95 | 968.13 | 969.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 944.60 | 963.42 | 967.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 866.00 | 859.50 | 875.71 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:00:00 | 852.15 | 858.02 | 872.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 12:00:00 | 851.25 | 856.66 | 870.32 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 09:15:00 | 818.20 | 858.14 | 866.71 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 850.50 | 839.83 | 849.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 850.50 | 839.83 | 849.14 | SL hit (close>ema400) qty=1.00 sl=849.14 alert=retest1 |

### Cycle 139 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 879.50 | 858.50 | 855.88 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 838.60 | 857.87 | 859.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 829.10 | 845.80 | 851.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 832.10 | 829.83 | 838.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:45:00 | 835.40 | 829.83 | 838.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 832.20 | 826.81 | 833.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 832.90 | 826.81 | 833.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 844.35 | 830.66 | 834.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 844.35 | 830.66 | 834.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 848.25 | 834.18 | 835.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 848.25 | 834.18 | 835.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 850.50 | 837.44 | 836.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 852.40 | 842.16 | 839.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 855.20 | 870.39 | 859.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 855.20 | 870.39 | 859.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 855.20 | 870.39 | 859.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 854.50 | 870.39 | 859.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 859.50 | 868.22 | 859.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 857.10 | 868.22 | 859.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 860.55 | 866.68 | 859.96 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 850.00 | 857.78 | 857.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 845.75 | 855.37 | 856.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 826.95 | 824.52 | 833.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 826.95 | 824.52 | 833.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 841.00 | 828.98 | 833.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 841.55 | 828.98 | 833.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 850.70 | 833.33 | 835.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 850.70 | 833.33 | 835.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 846.00 | 838.54 | 837.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 867.45 | 844.32 | 840.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 852.00 | 859.68 | 852.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 852.00 | 859.68 | 852.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 852.00 | 859.68 | 852.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 852.00 | 859.68 | 852.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 846.40 | 857.03 | 851.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 846.40 | 857.03 | 851.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 843.00 | 854.22 | 850.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 843.00 | 854.22 | 850.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 845.45 | 851.85 | 850.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 845.45 | 851.85 | 850.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 833.05 | 848.09 | 848.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 830.30 | 844.53 | 847.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 833.45 | 820.30 | 829.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 833.45 | 820.30 | 829.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 833.45 | 820.30 | 829.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 825.60 | 824.31 | 830.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 850.60 | 829.56 | 831.98 | SL hit (close>static) qty=1.00 sl=845.75 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 869.80 | 840.25 | 836.58 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 831.05 | 835.59 | 835.71 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 895.00 | 843.32 | 836.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 907.00 | 856.05 | 843.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 889.60 | 890.85 | 870.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 889.60 | 890.85 | 870.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 917.65 | 928.49 | 918.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:00:00 | 917.65 | 928.49 | 918.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 916.00 | 925.99 | 917.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 916.00 | 925.99 | 917.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 917.00 | 924.19 | 917.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 935.55 | 924.19 | 917.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 11:15:00 | 1029.11 | 1012.45 | 998.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 997.00 | 1028.90 | 1032.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 983.05 | 1014.70 | 1024.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 995.40 | 994.16 | 1006.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 995.40 | 994.16 | 1006.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1057.00 | 1006.14 | 1008.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 1059.10 | 1006.14 | 1008.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 1045.40 | 1013.99 | 1011.57 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 992.75 | 1016.47 | 1019.03 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1032.10 | 1019.42 | 1017.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 1051.30 | 1026.42 | 1021.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 1096.60 | 1099.96 | 1085.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 1096.60 | 1099.96 | 1085.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 09:15:00 | 600.25 | 2024-05-29 09:15:00 | 606.90 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest1 | 2024-05-31 09:45:00 | 594.90 | 2024-05-31 11:15:00 | 607.35 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-05-31 15:00:00 | 598.60 | 2024-06-03 09:15:00 | 632.15 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2024-06-11 09:30:00 | 601.45 | 2024-06-18 09:15:00 | 661.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 10:00:00 | 601.80 | 2024-06-18 09:15:00 | 661.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 10:30:00 | 601.90 | 2024-06-18 09:15:00 | 662.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-10 09:15:00 | 669.95 | 2024-07-12 14:15:00 | 665.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-07-10 14:00:00 | 666.40 | 2024-07-12 14:15:00 | 665.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-07-12 13:45:00 | 664.95 | 2024-07-12 14:15:00 | 665.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-07-24 11:15:00 | 632.65 | 2024-07-25 11:15:00 | 642.60 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-07-24 13:15:00 | 632.55 | 2024-07-25 11:15:00 | 642.60 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-07-24 14:00:00 | 632.10 | 2024-07-25 11:15:00 | 642.60 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-07-24 14:45:00 | 632.70 | 2024-07-25 11:15:00 | 642.60 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest1 | 2024-07-29 09:15:00 | 662.35 | 2024-07-30 11:15:00 | 649.90 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2024-08-06 14:30:00 | 603.90 | 2024-08-07 11:15:00 | 626.70 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest1 | 2024-08-07 09:45:00 | 609.15 | 2024-08-07 11:15:00 | 626.70 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest1 | 2024-08-09 09:15:00 | 647.70 | 2024-08-12 11:15:00 | 641.95 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest1 | 2024-08-09 13:15:00 | 645.00 | 2024-08-12 11:15:00 | 641.95 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-08-21 09:15:00 | 714.05 | 2024-08-23 13:15:00 | 706.35 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-08-23 13:00:00 | 708.00 | 2024-08-23 13:15:00 | 706.35 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-08-27 14:15:00 | 700.35 | 2024-08-30 14:15:00 | 668.51 | PARTIAL | 0.50 | 4.55% |
| SELL | retest2 | 2024-08-27 14:15:00 | 700.35 | 2024-09-02 10:15:00 | 680.45 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2024-08-28 11:30:00 | 703.70 | 2024-09-03 11:15:00 | 686.95 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2024-09-05 09:15:00 | 692.20 | 2024-09-09 09:15:00 | 674.65 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-09-05 14:15:00 | 697.65 | 2024-09-09 09:15:00 | 674.65 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-09-10 14:45:00 | 676.00 | 2024-09-19 11:15:00 | 645.52 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2024-09-11 10:00:00 | 679.50 | 2024-09-19 11:15:00 | 646.00 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-09-11 11:15:00 | 680.00 | 2024-09-19 11:15:00 | 646.71 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-09-11 12:15:00 | 680.75 | 2024-09-19 14:15:00 | 642.20 | PARTIAL | 0.50 | 5.66% |
| SELL | retest2 | 2024-09-11 13:15:00 | 677.60 | 2024-09-19 14:15:00 | 643.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 14:00:00 | 676.00 | 2024-09-19 14:15:00 | 642.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 12:30:00 | 675.00 | 2024-09-19 14:15:00 | 641.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 11:15:00 | 676.80 | 2024-09-19 14:15:00 | 642.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-10 14:45:00 | 676.00 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-09-11 10:00:00 | 679.50 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2024-09-11 11:15:00 | 680.00 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2024-09-11 12:15:00 | 680.75 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2024-09-11 13:15:00 | 677.60 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2024-09-11 14:00:00 | 676.00 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-09-12 12:30:00 | 675.00 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-09-13 11:15:00 | 676.80 | 2024-09-20 09:15:00 | 660.00 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2024-09-13 14:30:00 | 673.00 | 2024-09-23 09:15:00 | 681.65 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-09-16 09:45:00 | 673.20 | 2024-09-23 09:15:00 | 681.65 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-10-10 10:45:00 | 643.35 | 2024-10-11 09:15:00 | 660.70 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-10-10 12:45:00 | 643.15 | 2024-10-11 09:15:00 | 660.70 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-10-24 09:15:00 | 675.50 | 2024-10-30 11:15:00 | 689.25 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-10-24 11:15:00 | 678.70 | 2024-10-30 11:15:00 | 689.25 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-10-25 10:15:00 | 677.85 | 2024-10-30 11:15:00 | 689.25 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-10-25 15:00:00 | 679.85 | 2024-10-30 11:15:00 | 689.25 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-10-28 11:30:00 | 673.35 | 2024-10-30 11:15:00 | 689.25 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-10-29 10:30:00 | 672.00 | 2024-10-30 11:15:00 | 689.25 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-11-05 13:15:00 | 713.60 | 2024-11-08 11:15:00 | 707.95 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-11-06 10:45:00 | 713.70 | 2024-11-08 11:15:00 | 707.95 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-07 11:15:00 | 714.40 | 2024-11-08 11:15:00 | 707.95 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-11-07 13:45:00 | 716.25 | 2024-11-08 11:15:00 | 707.95 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-11-12 12:15:00 | 698.95 | 2024-11-13 15:15:00 | 664.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 698.95 | 2024-11-18 11:15:00 | 679.00 | STOP_HIT | 0.50 | 2.85% |
| BUY | retest2 | 2024-12-06 12:45:00 | 715.75 | 2024-12-10 10:15:00 | 787.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-09 09:30:00 | 713.05 | 2024-12-10 10:15:00 | 784.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 12:45:00 | 755.25 | 2024-12-26 09:15:00 | 756.65 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-12-24 11:45:00 | 756.75 | 2024-12-26 09:15:00 | 756.65 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-12-30 11:30:00 | 727.50 | 2025-01-01 11:15:00 | 742.90 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-01-20 12:15:00 | 684.10 | 2025-01-21 10:15:00 | 665.35 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-01-20 14:00:00 | 685.05 | 2025-01-21 10:15:00 | 665.35 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-01-23 14:15:00 | 652.30 | 2025-01-24 12:15:00 | 619.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 634.10 | 2025-01-27 09:15:00 | 602.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 652.30 | 2025-01-27 11:15:00 | 587.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 634.10 | 2025-01-28 09:15:00 | 570.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-03 09:15:00 | 566.90 | 2025-02-04 09:15:00 | 585.05 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-02-04 10:15:00 | 578.10 | 2025-02-04 14:15:00 | 580.35 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-02-06 15:15:00 | 607.00 | 2025-02-10 09:15:00 | 586.80 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-02-14 09:15:00 | 544.40 | 2025-02-14 14:15:00 | 517.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 544.40 | 2025-02-17 14:15:00 | 521.60 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest2 | 2025-02-24 12:15:00 | 542.90 | 2025-02-24 14:15:00 | 529.35 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-02-25 13:00:00 | 543.60 | 2025-02-27 09:15:00 | 523.20 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2025-02-25 13:45:00 | 542.95 | 2025-02-27 09:15:00 | 523.20 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-03-04 11:30:00 | 517.25 | 2025-03-05 09:15:00 | 521.45 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-03-04 13:00:00 | 517.85 | 2025-03-05 09:15:00 | 521.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-03-04 13:30:00 | 517.00 | 2025-03-05 09:15:00 | 521.45 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-03-21 09:15:00 | 576.40 | 2025-03-28 12:15:00 | 578.50 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-04-09 09:15:00 | 543.00 | 2025-04-11 13:15:00 | 551.40 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-04-11 12:45:00 | 555.20 | 2025-04-11 13:15:00 | 551.40 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-04-29 12:15:00 | 630.65 | 2025-04-30 09:15:00 | 646.90 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-04-29 13:00:00 | 631.05 | 2025-04-30 09:15:00 | 646.90 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-05-06 14:30:00 | 645.75 | 2025-05-06 15:15:00 | 636.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-08 13:00:00 | 653.15 | 2025-05-08 14:15:00 | 639.70 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-05-08 14:00:00 | 655.00 | 2025-05-08 14:15:00 | 639.70 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-23 13:15:00 | 720.10 | 2025-05-27 12:15:00 | 736.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-05-27 09:15:00 | 721.00 | 2025-05-27 12:15:00 | 736.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-26 12:15:00 | 782.85 | 2025-07-04 10:15:00 | 795.65 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-06-26 12:45:00 | 789.80 | 2025-07-04 10:15:00 | 795.65 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-07 12:15:00 | 780.05 | 2025-07-09 12:15:00 | 801.95 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-08 10:45:00 | 781.25 | 2025-07-09 12:15:00 | 801.95 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-07-11 12:15:00 | 807.55 | 2025-07-23 11:15:00 | 848.50 | STOP_HIT | 1.00 | 5.07% |
| BUY | retest2 | 2025-07-14 10:00:00 | 807.15 | 2025-07-23 11:15:00 | 848.50 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2025-07-14 14:45:00 | 812.95 | 2025-07-23 11:15:00 | 848.50 | STOP_HIT | 1.00 | 4.37% |
| BUY | retest2 | 2025-07-31 12:30:00 | 819.00 | 2025-07-31 14:15:00 | 811.05 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-08 13:15:00 | 803.05 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-08 14:15:00 | 802.50 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-11 09:45:00 | 803.00 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-11 11:30:00 | 803.90 | 2025-08-12 11:15:00 | 813.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-08-13 11:00:00 | 802.60 | 2025-08-18 09:15:00 | 827.00 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-14 14:15:00 | 800.50 | 2025-08-18 09:15:00 | 827.00 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-08-14 15:15:00 | 803.95 | 2025-08-18 09:15:00 | 827.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-08-22 09:15:00 | 875.55 | 2025-08-25 11:15:00 | 858.20 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-09-05 09:15:00 | 818.30 | 2025-09-05 11:15:00 | 801.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-09-15 13:45:00 | 829.10 | 2025-09-19 14:15:00 | 834.75 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2025-09-15 15:00:00 | 830.80 | 2025-09-19 14:15:00 | 834.75 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-09-24 09:15:00 | 847.15 | 2025-10-06 15:15:00 | 873.55 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2025-10-15 14:45:00 | 885.95 | 2025-10-23 09:15:00 | 974.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-31 11:45:00 | 868.95 | 2025-11-04 10:15:00 | 884.35 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-03 09:15:00 | 872.75 | 2025-11-04 10:15:00 | 884.35 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-04 14:30:00 | 877.50 | 2025-11-06 09:15:00 | 868.15 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-11-04 15:00:00 | 882.20 | 2025-11-06 09:15:00 | 868.15 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-11-25 09:15:00 | 867.00 | 2025-11-27 14:15:00 | 876.55 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-05 10:15:00 | 810.60 | 2025-12-10 10:15:00 | 822.05 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-08 11:45:00 | 822.25 | 2025-12-10 10:15:00 | 822.05 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest1 | 2025-12-16 12:30:00 | 868.20 | 2025-12-18 09:15:00 | 911.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-16 14:45:00 | 868.00 | 2025-12-18 09:15:00 | 911.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-17 15:15:00 | 868.00 | 2025-12-18 09:15:00 | 911.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-16 12:30:00 | 868.20 | 2025-12-18 15:15:00 | 888.30 | STOP_HIT | 0.50 | 2.32% |
| BUY | retest1 | 2025-12-16 14:45:00 | 868.00 | 2025-12-18 15:15:00 | 888.30 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest1 | 2025-12-17 15:15:00 | 868.00 | 2025-12-18 15:15:00 | 888.30 | STOP_HIT | 0.50 | 2.34% |
| BUY | retest2 | 2025-12-18 09:15:00 | 918.15 | 2025-12-24 14:15:00 | 880.00 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2026-01-02 11:00:00 | 880.20 | 2026-01-07 10:15:00 | 882.75 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-01-02 11:30:00 | 880.70 | 2026-01-07 10:15:00 | 882.75 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2026-01-02 12:30:00 | 881.60 | 2026-01-07 10:15:00 | 882.75 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2026-01-23 10:45:00 | 833.85 | 2026-01-29 09:15:00 | 838.65 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-23 11:30:00 | 834.90 | 2026-01-29 09:15:00 | 838.65 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-02-01 15:00:00 | 865.20 | 2026-02-02 10:15:00 | 846.80 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-02-02 09:15:00 | 863.20 | 2026-02-02 10:15:00 | 846.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-02 09:45:00 | 865.15 | 2026-02-02 10:15:00 | 846.80 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-02-02 10:30:00 | 863.25 | 2026-02-02 11:15:00 | 849.25 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-02-09 09:15:00 | 929.90 | 2026-02-10 12:15:00 | 913.95 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-10 09:15:00 | 924.90 | 2026-02-10 12:15:00 | 913.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2026-03-06 11:00:00 | 852.15 | 2026-03-10 09:15:00 | 850.50 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest1 | 2026-03-06 12:00:00 | 851.25 | 2026-03-10 09:15:00 | 850.50 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest1 | 2026-03-09 09:15:00 | 818.20 | 2026-03-10 09:15:00 | 850.50 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2026-04-01 11:30:00 | 825.60 | 2026-04-01 12:15:00 | 850.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-04-15 09:15:00 | 935.55 | 2026-04-21 11:15:00 | 1029.11 | TARGET_HIT | 1.00 | 10.00% |
