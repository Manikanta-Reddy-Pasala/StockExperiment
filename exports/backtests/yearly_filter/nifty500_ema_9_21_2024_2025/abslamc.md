# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1075.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 48 |
| ALERT3 | 283 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 119 |
| PARTIAL | 17 |
| TARGET_HIT | 14 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 142 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 81
- **Target hits / Stop hits / Partials:** 14 / 111 / 17
- **Avg / median % per leg:** 1.06% / -0.26%
- **Sum % (uncompounded):** 150.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 21 | 37.5% | 7 | 48 | 1 | 0.98% | 54.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 2.07% | 8.3% |
| BUY @ 3rd Alert (retest2) | 52 | 19 | 36.5% | 7 | 45 | 0 | 0.89% | 46.5% |
| SELL (all) | 86 | 40 | 46.5% | 7 | 63 | 16 | 1.11% | 95.8% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 3rd Alert (retest2) | 80 | 34 | 42.5% | 4 | 63 | 13 | 0.63% | 50.8% |
| retest1 (combined) | 10 | 8 | 80.0% | 3 | 3 | 4 | 5.33% | 53.3% |
| retest2 (combined) | 132 | 53 | 40.2% | 11 | 108 | 13 | 0.74% | 97.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 536.95 | 535.41 | 535.34 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 530.35 | 534.43 | 534.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 11:15:00 | 528.00 | 532.33 | 533.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 528.50 | 528.23 | 530.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-18 09:30:00 | 527.15 | 528.23 | 530.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 531.95 | 528.97 | 530.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 12:00:00 | 531.95 | 528.97 | 530.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 531.95 | 529.57 | 531.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 522.40 | 529.57 | 531.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 12:15:00 | 535.05 | 524.30 | 524.86 | SL hit (close>static) qty=1.00 sl=533.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 534.65 | 526.37 | 525.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 535.35 | 531.82 | 529.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 527.25 | 531.75 | 530.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 527.25 | 531.75 | 530.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 527.25 | 531.75 | 530.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 527.25 | 531.75 | 530.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 529.35 | 531.27 | 530.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 537.65 | 530.31 | 530.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 536.60 | 530.62 | 530.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:45:00 | 536.45 | 535.60 | 533.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 528.00 | 531.75 | 532.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 528.00 | 531.75 | 532.26 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 535.30 | 532.32 | 532.22 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 529.00 | 532.67 | 532.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 525.00 | 531.14 | 532.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 529.50 | 527.90 | 530.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 529.50 | 527.90 | 530.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 522.70 | 526.86 | 529.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:30:00 | 523.45 | 526.86 | 529.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 546.95 | 529.04 | 529.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:30:00 | 550.00 | 529.04 | 529.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 566.15 | 536.46 | 533.10 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 12:15:00 | 517.35 | 542.00 | 542.91 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 563.50 | 543.75 | 542.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 577.15 | 559.56 | 551.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 575.30 | 578.17 | 567.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 10:00:00 | 575.30 | 578.17 | 567.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 575.00 | 577.32 | 571.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 595.40 | 577.32 | 571.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-14 09:15:00 | 654.94 | 651.43 | 635.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 10:15:00 | 657.55 | 661.74 | 662.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 655.40 | 658.40 | 659.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 659.70 | 658.66 | 659.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 659.70 | 658.66 | 659.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 659.70 | 658.66 | 659.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:30:00 | 660.00 | 658.66 | 659.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 654.20 | 657.77 | 659.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:00:00 | 652.45 | 656.41 | 658.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 651.00 | 656.54 | 657.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 11:30:00 | 652.05 | 654.53 | 656.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 652.00 | 654.53 | 656.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 15:15:00 | 619.83 | 643.65 | 648.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 15:15:00 | 618.45 | 643.65 | 648.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 15:15:00 | 619.45 | 643.65 | 648.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 15:15:00 | 619.40 | 643.65 | 648.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 640.00 | 639.92 | 644.38 | SL hit (close>ema200) qty=0.50 sl=639.92 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 15:15:00 | 652.20 | 642.13 | 641.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 10:15:00 | 658.75 | 647.18 | 643.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 647.05 | 649.03 | 645.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 647.05 | 649.03 | 645.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 647.05 | 649.03 | 645.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 647.05 | 649.03 | 645.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 650.00 | 649.22 | 646.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 647.10 | 649.22 | 646.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 655.00 | 655.96 | 652.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 649.00 | 655.96 | 652.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 653.95 | 655.56 | 652.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:15:00 | 650.70 | 655.56 | 652.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 650.30 | 654.51 | 652.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:00:00 | 650.30 | 654.51 | 652.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 650.20 | 653.65 | 652.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:45:00 | 650.25 | 653.65 | 652.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 653.45 | 655.15 | 653.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 651.20 | 655.15 | 653.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 652.25 | 654.57 | 653.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 652.25 | 654.57 | 653.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 651.85 | 654.03 | 653.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:30:00 | 651.00 | 654.03 | 653.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 653.75 | 653.97 | 653.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:45:00 | 656.55 | 654.15 | 653.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 12:15:00 | 648.00 | 659.07 | 658.67 | SL hit (close<static) qty=1.00 sl=651.60 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 655.50 | 658.36 | 658.39 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 663.00 | 658.28 | 658.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 669.55 | 661.79 | 659.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 703.30 | 707.24 | 698.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 703.30 | 707.24 | 698.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 700.75 | 707.73 | 701.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 700.75 | 707.73 | 701.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 709.05 | 708.00 | 702.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 698.45 | 708.00 | 702.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 693.60 | 705.12 | 701.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 693.60 | 705.12 | 701.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 676.00 | 699.29 | 699.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 676.00 | 699.29 | 699.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 680.00 | 695.43 | 697.32 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 696.40 | 686.29 | 686.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 700.60 | 692.05 | 689.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 12:15:00 | 693.00 | 696.08 | 692.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 693.00 | 696.08 | 692.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 693.00 | 696.08 | 692.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 693.00 | 696.08 | 692.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 693.35 | 695.54 | 692.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 693.35 | 695.54 | 692.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 695.40 | 695.51 | 693.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:45:00 | 693.55 | 695.51 | 693.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 699.00 | 696.21 | 693.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 701.90 | 696.21 | 693.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 702.10 | 698.48 | 694.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 12:15:00 | 705.95 | 706.87 | 706.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 705.95 | 706.87 | 706.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 702.80 | 705.95 | 706.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 15:15:00 | 665.00 | 664.44 | 676.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:15:00 | 666.10 | 664.44 | 676.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 663.80 | 664.32 | 675.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 660.20 | 664.32 | 675.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 684.95 | 660.75 | 659.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 684.95 | 660.75 | 659.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 10:15:00 | 693.50 | 667.30 | 662.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 695.00 | 697.79 | 691.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 15:00:00 | 695.00 | 697.79 | 691.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 691.25 | 696.48 | 691.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 692.10 | 696.48 | 691.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 694.25 | 696.03 | 691.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 686.45 | 696.03 | 691.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 694.55 | 695.30 | 692.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:30:00 | 693.05 | 695.30 | 692.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 706.10 | 707.69 | 703.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 707.25 | 707.69 | 703.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 749.00 | 760.87 | 751.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:15:00 | 749.25 | 760.87 | 751.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 749.25 | 758.54 | 751.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 749.25 | 758.54 | 751.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 747.00 | 754.91 | 750.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 748.85 | 754.91 | 750.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 748.00 | 753.53 | 750.37 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 743.00 | 748.05 | 748.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 730.45 | 741.61 | 745.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 15:15:00 | 737.00 | 734.55 | 739.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:15:00 | 734.00 | 734.55 | 739.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 735.60 | 734.76 | 739.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 727.00 | 736.27 | 737.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:00:00 | 726.70 | 730.31 | 734.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 12:15:00 | 739.95 | 733.94 | 733.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 739.95 | 733.94 | 733.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 748.45 | 737.79 | 735.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 741.60 | 742.31 | 739.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 12:15:00 | 741.60 | 742.31 | 739.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 741.60 | 742.31 | 739.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 15:00:00 | 747.30 | 743.70 | 740.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 13:30:00 | 747.80 | 745.09 | 742.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 15:15:00 | 748.50 | 744.98 | 742.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 738.45 | 741.82 | 742.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 738.45 | 741.82 | 742.05 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 745.25 | 741.76 | 741.53 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 731.60 | 739.63 | 740.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 729.85 | 736.04 | 738.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 733.80 | 726.09 | 729.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 733.80 | 726.09 | 729.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 733.80 | 726.09 | 729.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 738.50 | 726.09 | 729.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 738.40 | 728.55 | 730.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 738.40 | 728.55 | 730.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 738.95 | 732.29 | 732.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 741.50 | 734.13 | 732.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 15:15:00 | 735.00 | 738.94 | 737.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 15:15:00 | 735.00 | 738.94 | 737.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 735.00 | 738.94 | 737.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 734.85 | 738.12 | 736.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 740.80 | 738.66 | 737.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 735.75 | 738.66 | 737.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 749.70 | 748.70 | 745.99 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 740.30 | 749.91 | 749.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 732.30 | 746.39 | 748.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 743.95 | 740.77 | 744.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 743.95 | 740.77 | 744.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 743.95 | 740.77 | 744.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 743.95 | 740.77 | 744.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 747.65 | 742.15 | 744.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 747.65 | 742.15 | 744.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 749.10 | 743.54 | 744.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 747.95 | 743.54 | 744.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 744.25 | 743.59 | 744.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 742.30 | 743.59 | 744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 772.05 | 749.29 | 747.22 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 15:15:00 | 745.00 | 750.60 | 751.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 739.75 | 746.76 | 749.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 732.55 | 730.12 | 735.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 732.55 | 730.12 | 735.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 732.55 | 730.12 | 735.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 732.55 | 730.12 | 735.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 730.95 | 730.29 | 734.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 722.30 | 728.89 | 733.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:45:00 | 722.40 | 727.32 | 732.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 13:15:00 | 741.85 | 729.39 | 729.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 13:15:00 | 741.85 | 729.39 | 729.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 750.55 | 733.62 | 731.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 738.05 | 738.30 | 734.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 738.05 | 738.30 | 734.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 734.90 | 737.33 | 734.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 734.90 | 737.33 | 734.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 731.80 | 736.23 | 734.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 731.10 | 736.23 | 734.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 730.25 | 735.03 | 733.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 729.35 | 735.03 | 733.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 725.90 | 732.40 | 732.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 716.95 | 727.90 | 730.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 15:15:00 | 689.40 | 686.94 | 696.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 09:15:00 | 704.35 | 686.94 | 696.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 709.50 | 691.45 | 698.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 709.50 | 691.45 | 698.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 703.15 | 693.79 | 698.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 701.90 | 695.92 | 699.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 14:15:00 | 706.30 | 700.87 | 700.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 14:15:00 | 706.30 | 700.87 | 700.78 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 697.20 | 700.59 | 700.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 696.30 | 699.36 | 700.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 11:15:00 | 692.70 | 691.61 | 694.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 11:15:00 | 692.70 | 691.61 | 694.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 692.70 | 691.61 | 694.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:45:00 | 693.50 | 691.61 | 694.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 692.85 | 691.86 | 694.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 693.05 | 691.86 | 694.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 693.40 | 692.17 | 694.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 693.40 | 692.17 | 694.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 696.25 | 692.98 | 694.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:45:00 | 696.95 | 692.98 | 694.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 698.00 | 693.99 | 694.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 704.10 | 693.99 | 694.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 699.85 | 695.16 | 695.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:45:00 | 703.65 | 695.16 | 695.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 698.65 | 695.86 | 695.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 744.00 | 707.16 | 701.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 14:15:00 | 764.25 | 770.26 | 752.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 15:00:00 | 764.25 | 770.26 | 752.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 763.80 | 767.82 | 754.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 769.35 | 762.11 | 756.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 769.50 | 763.10 | 758.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 10:15:00 | 748.40 | 755.77 | 756.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 748.40 | 755.77 | 756.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 735.10 | 751.64 | 754.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 12:15:00 | 742.40 | 737.05 | 743.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 12:15:00 | 742.40 | 737.05 | 743.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 742.40 | 737.05 | 743.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 742.40 | 737.05 | 743.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 740.10 | 737.66 | 742.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:15:00 | 735.55 | 737.66 | 742.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:30:00 | 737.30 | 737.58 | 741.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 738.30 | 738.65 | 741.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:15:00 | 737.10 | 738.90 | 741.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 728.60 | 735.68 | 738.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 752.20 | 738.35 | 738.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 09:15:00 | 752.20 | 738.35 | 738.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 773.80 | 755.95 | 748.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 10:15:00 | 772.15 | 774.10 | 763.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 11:00:00 | 772.15 | 774.10 | 763.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 771.20 | 772.02 | 766.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:30:00 | 767.50 | 772.02 | 766.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 780.45 | 773.56 | 767.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 11:45:00 | 788.45 | 777.25 | 770.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 792.70 | 793.36 | 786.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 801.00 | 814.81 | 815.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 801.00 | 814.81 | 815.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 793.80 | 804.36 | 809.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 823.10 | 802.50 | 805.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 823.10 | 802.50 | 805.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 823.10 | 802.50 | 805.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 823.10 | 802.50 | 805.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 810.90 | 804.18 | 805.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 806.90 | 804.90 | 805.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 806.60 | 804.55 | 805.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 12:15:00 | 807.70 | 796.92 | 796.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 12:15:00 | 807.70 | 796.92 | 796.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 09:15:00 | 813.90 | 802.53 | 799.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 12:15:00 | 803.45 | 803.76 | 800.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 13:00:00 | 803.45 | 803.76 | 800.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 808.35 | 805.80 | 802.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 826.60 | 805.80 | 802.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 818.60 | 821.00 | 814.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 10:45:00 | 815.10 | 819.58 | 815.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 09:15:00 | 900.46 | 875.03 | 870.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 867.00 | 873.40 | 874.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 10:15:00 | 861.65 | 871.56 | 873.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 10:15:00 | 858.50 | 856.34 | 862.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:45:00 | 858.90 | 856.34 | 862.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 848.50 | 853.26 | 858.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 841.70 | 849.61 | 852.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:00:00 | 843.05 | 848.30 | 852.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:45:00 | 843.00 | 847.07 | 850.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 15:15:00 | 840.00 | 846.66 | 849.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 842.65 | 844.79 | 848.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:45:00 | 846.80 | 844.79 | 848.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 829.35 | 819.59 | 825.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 829.35 | 819.59 | 825.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 835.95 | 822.86 | 826.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 835.95 | 822.86 | 826.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 827.65 | 826.27 | 827.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:45:00 | 829.10 | 826.27 | 827.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 827.30 | 826.47 | 827.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 828.20 | 826.47 | 827.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 826.75 | 826.53 | 827.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:15:00 | 828.30 | 826.53 | 827.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 828.90 | 827.00 | 827.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 832.70 | 828.54 | 828.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 832.70 | 828.54 | 828.12 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 11:15:00 | 817.65 | 827.40 | 828.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 811.15 | 820.25 | 822.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 11:15:00 | 799.40 | 797.56 | 804.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 12:00:00 | 799.40 | 797.56 | 804.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 805.50 | 799.67 | 803.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 805.50 | 799.67 | 803.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 796.10 | 798.96 | 803.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 806.00 | 798.96 | 803.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 801.15 | 799.40 | 802.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 809.85 | 799.40 | 802.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 797.05 | 798.93 | 802.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:15:00 | 794.15 | 798.93 | 802.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 12:15:00 | 804.30 | 800.26 | 802.40 | SL hit (close>static) qty=1.00 sl=803.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 814.50 | 804.06 | 803.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 835.10 | 818.41 | 811.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 15:15:00 | 827.45 | 830.32 | 822.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 09:15:00 | 827.05 | 830.32 | 822.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 826.95 | 829.65 | 822.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 840.00 | 829.72 | 824.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 851.00 | 832.16 | 826.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 846.95 | 838.93 | 834.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 15:15:00 | 829.10 | 833.06 | 833.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 15:15:00 | 829.10 | 833.06 | 833.25 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 836.95 | 833.84 | 833.58 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 827.15 | 832.28 | 832.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 818.70 | 826.16 | 829.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 820.30 | 808.89 | 816.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 820.30 | 808.89 | 816.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 820.30 | 808.89 | 816.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 821.20 | 808.89 | 816.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 816.70 | 810.45 | 816.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 810.50 | 817.24 | 818.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 14:15:00 | 769.97 | 794.32 | 801.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 729.45 | 752.27 | 770.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 761.95 | 754.75 | 754.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 15:15:00 | 769.75 | 757.75 | 755.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 764.25 | 766.29 | 762.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 764.25 | 766.29 | 762.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 764.25 | 766.29 | 762.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 758.90 | 766.29 | 762.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 760.00 | 765.03 | 762.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 760.00 | 765.03 | 762.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 760.10 | 764.04 | 762.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 760.25 | 764.04 | 762.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 759.90 | 763.51 | 762.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 759.90 | 763.51 | 762.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 758.75 | 762.56 | 761.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:00:00 | 758.75 | 762.56 | 761.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 747.55 | 759.15 | 760.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 739.20 | 750.55 | 754.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 725.35 | 724.43 | 733.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 11:30:00 | 715.85 | 722.40 | 731.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 14:15:00 | 716.15 | 720.72 | 728.75 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 15:15:00 | 717.00 | 720.07 | 727.72 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 709.45 | 717.46 | 725.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:45:00 | 706.30 | 714.05 | 721.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 705.60 | 712.36 | 720.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 15:15:00 | 705.00 | 711.48 | 719.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 680.06 | 700.19 | 712.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 680.34 | 700.19 | 712.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 681.15 | 700.19 | 712.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 670.98 | 700.19 | 712.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 670.32 | 700.19 | 712.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 669.75 | 700.19 | 712.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 13:15:00 | 644.26 | 668.62 | 692.06 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 45 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 661.10 | 648.48 | 647.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 676.00 | 657.22 | 652.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 676.40 | 678.68 | 669.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 12:00:00 | 676.40 | 678.68 | 669.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 652.25 | 672.84 | 670.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 645.60 | 672.84 | 670.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 658.55 | 669.98 | 669.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 648.75 | 669.98 | 669.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 652.60 | 666.50 | 667.56 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 676.80 | 666.46 | 666.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 684.55 | 671.94 | 668.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 702.60 | 702.80 | 693.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:45:00 | 703.30 | 702.80 | 693.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 692.75 | 699.71 | 694.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 707.70 | 700.99 | 696.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 704.55 | 701.44 | 696.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 705.60 | 701.44 | 696.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 673.95 | 694.11 | 694.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 673.95 | 694.11 | 694.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 670.90 | 686.55 | 690.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 668.85 | 662.08 | 670.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 668.85 | 662.08 | 670.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 668.85 | 662.08 | 670.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 672.70 | 662.08 | 670.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 670.30 | 663.72 | 670.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 670.30 | 663.72 | 670.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 668.50 | 664.68 | 670.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 668.50 | 664.68 | 670.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 662.15 | 663.90 | 667.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:30:00 | 668.85 | 663.90 | 667.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 662.80 | 663.68 | 667.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:30:00 | 665.70 | 663.68 | 667.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 657.60 | 661.86 | 665.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 15:00:00 | 657.60 | 661.86 | 665.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 650.65 | 651.66 | 657.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:30:00 | 658.20 | 651.66 | 657.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 650.50 | 647.94 | 652.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 653.20 | 647.94 | 652.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 648.10 | 647.90 | 651.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 640.80 | 647.90 | 651.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:30:00 | 644.90 | 643.71 | 646.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 652.95 | 648.88 | 648.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 652.95 | 648.88 | 648.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 15:15:00 | 659.95 | 651.10 | 649.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 654.05 | 654.97 | 652.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:45:00 | 653.40 | 654.97 | 652.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 650.35 | 654.04 | 651.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:00:00 | 650.35 | 654.04 | 651.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 648.55 | 652.94 | 651.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 648.55 | 652.94 | 651.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 649.50 | 652.26 | 651.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 663.45 | 652.26 | 651.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 648.90 | 650.58 | 650.77 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 664.70 | 653.40 | 652.04 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 646.00 | 654.46 | 654.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 640.40 | 651.65 | 653.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 608.15 | 607.73 | 612.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 608.15 | 607.73 | 612.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 608.15 | 607.73 | 612.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 612.30 | 607.73 | 612.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 616.20 | 609.43 | 613.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 616.20 | 609.43 | 613.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 622.65 | 612.07 | 613.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 622.65 | 612.07 | 613.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 625.30 | 614.72 | 614.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:45:00 | 625.75 | 614.72 | 614.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 619.05 | 615.58 | 615.29 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 14:15:00 | 612.70 | 615.01 | 615.06 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 618.95 | 615.47 | 615.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 627.85 | 617.95 | 616.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 617.40 | 617.84 | 616.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 617.40 | 617.84 | 616.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 617.40 | 617.84 | 616.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 617.40 | 617.84 | 616.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 617.50 | 617.77 | 616.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:15:00 | 616.35 | 617.77 | 616.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 620.00 | 618.22 | 616.88 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 609.30 | 616.94 | 617.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 606.30 | 612.56 | 614.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 11:15:00 | 602.65 | 597.38 | 601.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 602.65 | 597.38 | 601.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 602.65 | 597.38 | 601.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 602.65 | 597.38 | 601.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 607.20 | 599.34 | 601.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 606.75 | 599.34 | 601.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 607.40 | 603.70 | 603.52 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 597.00 | 602.34 | 602.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 593.80 | 600.63 | 602.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 597.60 | 595.94 | 598.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 597.60 | 595.94 | 598.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 597.60 | 595.94 | 598.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 602.55 | 595.94 | 598.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 593.70 | 595.49 | 598.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 593.00 | 594.87 | 597.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:30:00 | 592.55 | 593.50 | 595.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 593.00 | 594.00 | 595.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 607.80 | 597.34 | 596.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 607.80 | 597.34 | 596.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 612.95 | 600.46 | 598.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 638.05 | 638.57 | 628.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 11:45:00 | 637.55 | 638.57 | 628.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 646.40 | 648.72 | 645.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 645.25 | 648.72 | 645.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 644.85 | 647.94 | 645.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 639.15 | 647.94 | 645.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 641.15 | 646.58 | 644.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 641.15 | 646.58 | 644.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 638.60 | 644.99 | 644.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 638.60 | 644.99 | 644.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 637.75 | 643.54 | 643.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 634.65 | 641.76 | 642.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 637.90 | 637.12 | 639.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 637.90 | 637.12 | 639.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 637.90 | 637.12 | 639.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 638.55 | 637.12 | 639.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 643.40 | 638.16 | 639.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:15:00 | 646.20 | 638.16 | 639.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 642.85 | 639.10 | 639.91 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 649.15 | 642.05 | 641.17 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 635.45 | 640.99 | 641.25 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 640.80 | 640.38 | 640.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 648.10 | 641.93 | 641.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 652.80 | 655.59 | 650.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 652.80 | 655.59 | 650.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 652.80 | 655.59 | 650.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 652.00 | 655.59 | 650.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 655.45 | 655.28 | 650.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 655.45 | 655.28 | 650.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 655.15 | 655.25 | 651.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 655.15 | 655.25 | 651.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 654.05 | 655.01 | 651.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:30:00 | 652.80 | 655.01 | 651.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 654.00 | 654.81 | 651.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:45:00 | 652.50 | 654.81 | 651.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 622.25 | 648.33 | 649.33 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 639.85 | 628.99 | 628.84 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 13:15:00 | 623.55 | 627.90 | 628.36 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 640.55 | 630.65 | 629.38 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 10:15:00 | 626.10 | 629.97 | 630.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 13:15:00 | 622.85 | 627.27 | 628.79 | Break + close below crossover candle low |

### Cycle 69 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 640.60 | 629.38 | 629.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 11:15:00 | 659.50 | 637.50 | 633.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 14:15:00 | 643.25 | 643.96 | 637.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 14:45:00 | 646.70 | 643.96 | 637.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 657.10 | 661.64 | 658.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 657.10 | 661.64 | 658.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 655.85 | 660.48 | 658.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 656.30 | 660.48 | 658.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 647.85 | 656.28 | 657.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 13:15:00 | 632.70 | 640.30 | 645.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 14:15:00 | 642.20 | 640.68 | 645.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 14:15:00 | 642.20 | 640.68 | 645.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 642.20 | 640.68 | 645.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:45:00 | 644.50 | 640.68 | 645.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 644.15 | 641.38 | 645.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 660.70 | 641.38 | 645.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 660.10 | 645.12 | 646.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 655.85 | 645.12 | 646.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 682.00 | 652.50 | 649.88 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 652.10 | 659.34 | 659.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 647.35 | 656.94 | 658.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 12:15:00 | 657.15 | 656.98 | 658.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 12:15:00 | 657.15 | 656.98 | 658.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 657.15 | 656.98 | 658.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 657.15 | 656.98 | 658.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 660.00 | 657.59 | 658.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 649.10 | 657.25 | 658.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 10:45:00 | 652.80 | 655.21 | 657.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 14:15:00 | 660.75 | 657.95 | 657.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 660.75 | 657.95 | 657.85 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 649.00 | 656.49 | 657.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 643.00 | 653.80 | 655.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 642.60 | 640.95 | 647.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 642.60 | 640.95 | 647.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 645.55 | 642.28 | 646.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:30:00 | 643.75 | 642.28 | 646.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 647.85 | 643.40 | 646.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 647.85 | 643.40 | 646.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 648.00 | 644.32 | 646.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 648.60 | 644.32 | 646.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 648.05 | 645.06 | 646.77 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 658.15 | 648.14 | 647.90 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 640.05 | 647.16 | 647.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 633.35 | 644.40 | 646.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 643.45 | 640.26 | 643.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 643.45 | 640.26 | 643.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 643.45 | 640.26 | 643.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:30:00 | 648.80 | 640.26 | 643.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 646.00 | 641.41 | 643.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 671.70 | 641.41 | 643.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 671.70 | 647.47 | 645.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 680.00 | 657.96 | 651.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 671.60 | 674.31 | 665.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:30:00 | 671.90 | 674.31 | 665.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 690.00 | 682.71 | 679.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 693.00 | 684.41 | 680.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:45:00 | 691.70 | 686.64 | 681.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 725.00 | 727.28 | 727.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 725.00 | 727.28 | 727.53 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 736.60 | 729.15 | 728.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 740.00 | 731.32 | 729.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 15:15:00 | 746.90 | 748.85 | 743.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:15:00 | 755.00 | 748.85 | 743.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 753.70 | 755.52 | 749.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 753.70 | 755.52 | 749.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 750.00 | 754.41 | 749.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 758.00 | 754.41 | 749.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 11:30:00 | 757.30 | 758.53 | 755.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:45:00 | 758.60 | 759.06 | 756.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 14:15:00 | 752.65 | 757.69 | 756.01 | SL hit (close<ema400) qty=1.00 sl=756.01 alert=retest1 |

### Cycle 80 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 782.00 | 791.96 | 792.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 774.10 | 785.57 | 789.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 775.35 | 770.91 | 777.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 775.35 | 770.91 | 777.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 775.35 | 770.91 | 777.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 774.30 | 770.91 | 777.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 772.35 | 770.99 | 774.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 762.10 | 768.42 | 771.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 757.00 | 766.61 | 769.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 756.05 | 754.63 | 754.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 756.05 | 754.63 | 754.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 769.90 | 757.68 | 755.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 762.50 | 762.99 | 759.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 765.00 | 762.99 | 759.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 773.05 | 775.95 | 773.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:45:00 | 772.80 | 775.95 | 773.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 771.85 | 775.13 | 772.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 772.50 | 775.13 | 772.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 764.90 | 773.08 | 772.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 764.90 | 773.08 | 772.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 785.60 | 791.61 | 786.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 786.30 | 791.61 | 786.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 783.25 | 789.94 | 786.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 783.25 | 789.94 | 786.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 785.00 | 788.95 | 786.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:15:00 | 781.75 | 788.95 | 786.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 795.00 | 789.65 | 786.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:30:00 | 800.85 | 795.31 | 791.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 880.94 | 869.38 | 861.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 878.25 | 879.90 | 879.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 872.85 | 878.19 | 879.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 875.70 | 862.03 | 867.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 875.70 | 862.03 | 867.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 875.70 | 862.03 | 867.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 875.70 | 862.03 | 867.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 875.50 | 864.72 | 867.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 875.50 | 864.72 | 867.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 14:15:00 | 872.30 | 869.97 | 869.76 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 865.00 | 868.97 | 869.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 09:15:00 | 861.95 | 867.57 | 868.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 874.70 | 867.70 | 868.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 11:15:00 | 874.70 | 867.70 | 868.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 874.70 | 867.70 | 868.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 874.70 | 867.70 | 868.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 12:15:00 | 876.50 | 869.46 | 869.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 881.70 | 875.84 | 873.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 877.45 | 877.60 | 874.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 868.85 | 877.60 | 874.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 872.30 | 876.54 | 874.45 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 871.05 | 873.10 | 873.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 864.00 | 871.19 | 872.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 851.55 | 846.19 | 855.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 851.55 | 846.19 | 855.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 851.55 | 846.19 | 855.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 850.55 | 846.19 | 855.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 854.70 | 847.89 | 855.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:30:00 | 856.15 | 847.89 | 855.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 858.80 | 850.08 | 855.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 858.80 | 850.08 | 855.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 859.85 | 852.03 | 855.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 860.50 | 852.03 | 855.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 865.70 | 859.47 | 858.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 872.65 | 862.08 | 860.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 14:15:00 | 861.40 | 862.92 | 861.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 14:15:00 | 861.40 | 862.92 | 861.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 861.40 | 862.92 | 861.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:15:00 | 857.80 | 862.92 | 861.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 857.80 | 861.89 | 860.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 863.95 | 861.89 | 860.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 848.45 | 858.06 | 859.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 848.45 | 858.06 | 859.14 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 867.30 | 854.34 | 853.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 881.95 | 859.86 | 855.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 860.25 | 865.20 | 860.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 860.25 | 865.20 | 860.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 860.25 | 865.20 | 860.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 856.95 | 865.20 | 860.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 856.95 | 863.55 | 860.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 855.95 | 863.55 | 860.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 850.55 | 858.62 | 858.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 849.10 | 858.62 | 858.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 844.50 | 855.80 | 857.02 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 867.60 | 856.82 | 856.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 870.45 | 864.91 | 861.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 884.10 | 885.17 | 878.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 884.10 | 885.17 | 878.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 885.85 | 888.33 | 882.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 884.40 | 888.33 | 882.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 882.70 | 887.21 | 882.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 882.70 | 887.21 | 882.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 884.85 | 886.73 | 883.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 884.95 | 886.73 | 883.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 885.30 | 886.45 | 883.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 883.45 | 886.45 | 883.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 897.45 | 888.65 | 884.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 892.15 | 888.65 | 884.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 884.30 | 889.21 | 886.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 884.30 | 889.21 | 886.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 881.95 | 887.76 | 885.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 881.95 | 887.76 | 885.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 884.45 | 886.41 | 885.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 883.85 | 886.41 | 885.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 883.55 | 885.83 | 885.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 883.40 | 885.83 | 885.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 882.65 | 885.20 | 885.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 878.75 | 883.91 | 884.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 842.65 | 838.72 | 845.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 842.65 | 838.72 | 845.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 842.65 | 838.72 | 845.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 843.80 | 838.72 | 845.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 839.00 | 836.63 | 841.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 831.60 | 836.63 | 841.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 838.00 | 836.90 | 840.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 829.00 | 835.44 | 838.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:00:00 | 827.60 | 833.87 | 837.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 828.70 | 832.68 | 835.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:30:00 | 828.25 | 831.67 | 834.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 836.00 | 832.80 | 834.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 832.70 | 832.80 | 834.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 835.85 | 833.41 | 834.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:00:00 | 829.55 | 832.64 | 834.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:15:00 | 830.00 | 832.23 | 832.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:00:00 | 830.50 | 831.54 | 832.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:45:00 | 827.70 | 831.00 | 831.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 830.00 | 830.80 | 831.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 830.05 | 830.80 | 831.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 848.00 | 834.24 | 833.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 848.00 | 834.24 | 833.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 870.15 | 853.30 | 847.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 858.80 | 861.80 | 854.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:30:00 | 860.05 | 861.80 | 854.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 856.80 | 861.59 | 856.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 856.80 | 861.59 | 856.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 858.40 | 860.63 | 856.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 854.80 | 860.63 | 856.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 857.45 | 860.00 | 856.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:30:00 | 856.25 | 860.00 | 856.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 855.10 | 859.02 | 856.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 855.10 | 859.02 | 856.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 854.45 | 858.10 | 856.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:15:00 | 854.70 | 858.10 | 856.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 854.70 | 857.42 | 856.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 853.40 | 857.42 | 856.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 852.50 | 856.44 | 855.90 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 847.45 | 854.64 | 855.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 846.00 | 852.92 | 854.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 852.90 | 852.71 | 854.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 852.90 | 852.71 | 854.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 852.90 | 852.71 | 854.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 852.65 | 852.71 | 854.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 847.50 | 851.39 | 853.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 857.65 | 851.39 | 853.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 844.15 | 845.43 | 849.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 844.00 | 845.43 | 849.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 839.90 | 841.75 | 845.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 843.45 | 841.75 | 845.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 824.00 | 823.03 | 826.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 816.65 | 823.03 | 826.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 775.82 | 788.91 | 798.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 790.05 | 787.24 | 796.07 | SL hit (close>ema200) qty=0.50 sl=787.24 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 793.55 | 789.72 | 789.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 801.00 | 792.09 | 790.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 818.25 | 819.35 | 814.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 823.30 | 819.35 | 814.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 820.20 | 819.52 | 815.36 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 809.70 | 814.94 | 815.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 13:15:00 | 805.95 | 813.14 | 814.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 817.70 | 811.28 | 813.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 817.70 | 811.28 | 813.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 817.70 | 811.28 | 813.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 821.50 | 811.28 | 813.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 814.75 | 811.98 | 813.34 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 820.25 | 815.30 | 814.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 833.15 | 821.43 | 818.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 13:15:00 | 846.25 | 851.92 | 842.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:00:00 | 846.25 | 851.92 | 842.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 864.90 | 862.97 | 856.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 855.20 | 862.97 | 856.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 861.80 | 863.11 | 858.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 869.70 | 864.72 | 860.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 852.40 | 860.44 | 859.96 | SL hit (close<static) qty=1.00 sl=857.10 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 854.45 | 859.24 | 859.46 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 864.65 | 860.32 | 859.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 870.70 | 863.20 | 861.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 10:15:00 | 864.00 | 864.67 | 862.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 10:30:00 | 864.55 | 864.67 | 862.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 864.45 | 864.63 | 862.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 861.80 | 864.63 | 862.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 858.85 | 863.47 | 862.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:45:00 | 859.45 | 863.47 | 862.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 856.30 | 862.04 | 861.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 856.30 | 862.04 | 861.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 850.95 | 859.82 | 860.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 846.00 | 857.06 | 859.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 767.45 | 767.19 | 774.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 767.45 | 767.19 | 774.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 766.70 | 767.47 | 772.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 775.20 | 767.47 | 772.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 771.65 | 768.30 | 772.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 771.25 | 768.30 | 772.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 772.60 | 769.16 | 772.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 772.60 | 769.16 | 772.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 771.50 | 769.63 | 772.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:30:00 | 773.50 | 769.63 | 772.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 773.30 | 770.36 | 772.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:30:00 | 773.40 | 770.36 | 772.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 776.15 | 771.52 | 772.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 776.65 | 771.52 | 772.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 787.00 | 774.62 | 774.07 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 770.80 | 773.72 | 773.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 756.95 | 768.97 | 771.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 741.10 | 739.28 | 745.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:45:00 | 740.05 | 739.28 | 745.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 747.75 | 740.41 | 744.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 744.30 | 740.41 | 744.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 742.00 | 740.73 | 743.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 748.95 | 740.73 | 743.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 748.70 | 742.32 | 744.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 748.70 | 742.32 | 744.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 746.45 | 743.15 | 744.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 747.25 | 743.15 | 744.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 13:15:00 | 755.95 | 745.71 | 745.63 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 738.75 | 745.35 | 745.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 10:15:00 | 735.20 | 743.32 | 744.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 742.25 | 738.06 | 740.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 742.25 | 738.06 | 740.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 742.25 | 738.06 | 740.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 742.25 | 738.06 | 740.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 742.20 | 738.89 | 740.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 742.10 | 738.89 | 740.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 739.20 | 739.43 | 740.81 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 749.00 | 742.62 | 742.00 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 738.40 | 741.60 | 741.68 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 742.00 | 740.81 | 740.80 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 738.65 | 740.38 | 740.60 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 742.60 | 740.76 | 740.73 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 740.40 | 740.69 | 740.70 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 743.50 | 741.25 | 740.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 14:15:00 | 744.20 | 742.06 | 741.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 736.05 | 741.54 | 741.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 736.05 | 741.54 | 741.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 736.05 | 741.54 | 741.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:15:00 | 732.05 | 741.54 | 741.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 732.55 | 739.74 | 740.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 719.75 | 732.90 | 736.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 717.15 | 715.39 | 722.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 717.15 | 715.39 | 722.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 725.05 | 717.32 | 722.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:45:00 | 725.00 | 717.32 | 722.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 725.05 | 718.87 | 722.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 720.15 | 718.87 | 722.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 730.00 | 724.43 | 723.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 730.00 | 724.43 | 723.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 737.60 | 728.11 | 725.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 728.55 | 728.73 | 726.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:30:00 | 730.35 | 728.73 | 726.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 729.45 | 728.87 | 726.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 728.25 | 728.87 | 726.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 727.35 | 728.57 | 726.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 727.35 | 728.57 | 726.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 728.80 | 728.61 | 726.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 730.00 | 728.61 | 726.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 730.15 | 728.94 | 727.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 730.45 | 729.24 | 727.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 730.50 | 731.17 | 729.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 733.40 | 731.61 | 729.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 734.50 | 731.61 | 729.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:15:00 | 735.45 | 731.94 | 730.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:00:00 | 735.10 | 733.24 | 731.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 728.20 | 732.51 | 731.22 | SL hit (close<static) qty=1.00 sl=728.85 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 723.35 | 729.22 | 729.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 719.15 | 726.65 | 728.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 12:15:00 | 725.65 | 723.56 | 725.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 12:15:00 | 725.65 | 723.56 | 725.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 725.65 | 723.56 | 725.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 725.65 | 723.56 | 725.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 726.80 | 724.21 | 725.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 726.65 | 724.21 | 725.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 727.85 | 724.94 | 725.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 726.60 | 724.94 | 725.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 723.05 | 723.12 | 724.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 723.05 | 723.12 | 724.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 725.25 | 723.55 | 724.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 725.25 | 723.55 | 724.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 724.10 | 723.66 | 724.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 727.15 | 723.66 | 724.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 724.80 | 723.88 | 724.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 724.80 | 723.88 | 724.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 726.40 | 724.39 | 724.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 724.70 | 724.45 | 724.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:15:00 | 724.65 | 724.56 | 724.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 726.15 | 724.88 | 724.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 11:15:00 | 726.15 | 724.88 | 724.82 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 717.50 | 723.93 | 724.46 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 734.30 | 726.49 | 725.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 736.60 | 731.88 | 729.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 748.95 | 750.50 | 745.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 748.95 | 750.50 | 745.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 752.70 | 750.59 | 746.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 756.15 | 751.96 | 747.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 755.10 | 767.77 | 768.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 755.10 | 767.77 | 768.15 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 14:15:00 | 774.70 | 765.26 | 764.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 15:15:00 | 779.80 | 768.17 | 766.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 771.15 | 773.11 | 770.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 771.15 | 773.11 | 770.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 771.15 | 773.11 | 770.67 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 765.35 | 769.99 | 770.19 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 793.00 | 774.46 | 772.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 803.35 | 785.24 | 778.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 806.55 | 810.48 | 796.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:45:00 | 804.80 | 810.48 | 796.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 853.00 | 850.87 | 843.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 856.00 | 851.48 | 845.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:00:00 | 855.00 | 852.80 | 847.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 857.25 | 853.04 | 848.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 841.00 | 850.64 | 847.58 | SL hit (close<static) qty=1.00 sl=841.65 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 824.70 | 841.39 | 843.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 820.65 | 832.88 | 838.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 798.35 | 793.12 | 807.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 798.35 | 793.12 | 807.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 800.00 | 794.49 | 806.81 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 825.00 | 810.83 | 810.01 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 806.65 | 809.34 | 809.48 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 811.00 | 809.67 | 809.61 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 807.10 | 809.45 | 809.55 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 830.05 | 813.57 | 811.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 848.90 | 824.07 | 816.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 828.50 | 830.39 | 823.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:45:00 | 829.25 | 829.74 | 823.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 823.50 | 828.49 | 823.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 823.50 | 828.49 | 823.69 | SL hit (close<ema400) qty=1.00 sl=823.69 alert=retest1 |

### Cycle 128 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 804.70 | 819.59 | 820.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 797.80 | 812.27 | 817.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 795.25 | 792.57 | 801.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 795.25 | 792.57 | 801.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 778.00 | 782.82 | 790.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 778.00 | 782.82 | 790.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 794.40 | 782.70 | 788.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 784.60 | 783.73 | 788.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:45:00 | 787.55 | 783.78 | 786.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 786.05 | 783.78 | 786.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:30:00 | 788.50 | 785.79 | 786.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 778.80 | 784.40 | 785.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:45:00 | 775.50 | 781.51 | 784.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:30:00 | 775.65 | 780.40 | 783.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:00:00 | 775.95 | 780.40 | 783.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:45:00 | 776.00 | 779.62 | 782.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 779.15 | 779.10 | 781.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 777.70 | 779.10 | 781.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 775.00 | 778.28 | 781.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 771.00 | 778.28 | 781.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 745.37 | 766.38 | 772.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 748.17 | 766.38 | 772.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 746.75 | 766.38 | 772.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 749.07 | 766.38 | 772.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 763.65 | 763.12 | 768.67 | SL hit (close>ema200) qty=0.50 sl=763.12 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 775.80 | 763.50 | 762.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 777.00 | 766.20 | 763.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 817.60 | 817.98 | 803.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 812.35 | 817.98 | 803.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 806.70 | 812.47 | 807.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 800.20 | 812.47 | 807.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 800.80 | 810.13 | 807.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:15:00 | 801.00 | 810.13 | 807.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 805.90 | 809.29 | 806.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 809.40 | 809.29 | 806.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:30:00 | 807.00 | 809.68 | 807.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-17 09:15:00 | 887.70 | 876.45 | 870.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 15:15:00 | 910.10 | 915.88 | 916.53 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 921.40 | 916.98 | 916.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 930.40 | 919.67 | 918.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 910.25 | 917.78 | 917.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 910.25 | 917.78 | 917.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 910.25 | 917.78 | 917.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 910.25 | 917.78 | 917.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 909.30 | 916.09 | 916.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 904.05 | 912.27 | 914.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 13:15:00 | 867.15 | 866.69 | 879.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:30:00 | 865.60 | 866.69 | 879.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 886.90 | 871.37 | 878.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 886.90 | 871.37 | 878.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 900.35 | 877.16 | 880.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 900.35 | 877.16 | 880.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 897.15 | 884.09 | 883.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 900.75 | 890.27 | 886.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 877.60 | 900.69 | 894.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 877.60 | 900.69 | 894.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 877.60 | 900.69 | 894.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 879.50 | 900.69 | 894.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 871.80 | 894.92 | 892.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 879.50 | 894.92 | 892.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 870.85 | 890.10 | 890.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 866.45 | 883.15 | 886.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 894.00 | 882.54 | 885.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 894.00 | 882.54 | 885.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 894.00 | 882.54 | 885.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 894.00 | 882.54 | 885.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 897.15 | 885.47 | 886.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 897.20 | 885.47 | 886.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 930.05 | 894.38 | 890.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 1046.55 | 924.82 | 904.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 988.40 | 992.61 | 964.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 988.40 | 992.61 | 964.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 988.40 | 992.61 | 964.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:45:00 | 997.00 | 992.61 | 964.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 974.00 | 988.19 | 975.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 974.00 | 988.19 | 975.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 962.80 | 983.11 | 974.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 962.80 | 983.11 | 974.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 952.95 | 977.08 | 972.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 952.95 | 977.08 | 972.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 946.70 | 966.04 | 968.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 935.50 | 959.93 | 965.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 980.05 | 958.96 | 963.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 980.05 | 958.96 | 963.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 980.05 | 958.96 | 963.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 984.20 | 958.96 | 963.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 969.80 | 961.13 | 963.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 11:30:00 | 961.45 | 960.82 | 963.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:00:00 | 959.60 | 960.82 | 963.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:15:00 | 960.00 | 956.17 | 959.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 987.15 | 962.37 | 961.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 987.15 | 962.37 | 961.85 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 959.40 | 961.58 | 961.72 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 978.00 | 964.61 | 963.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 981.65 | 970.56 | 966.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 968.45 | 970.71 | 967.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 15:00:00 | 968.45 | 970.71 | 967.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 970.00 | 970.57 | 967.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 956.95 | 970.57 | 967.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 964.25 | 969.31 | 967.19 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 961.15 | 965.75 | 965.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 946.25 | 961.85 | 964.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 929.85 | 906.62 | 914.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 929.85 | 906.62 | 914.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 929.85 | 906.62 | 914.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 929.85 | 906.62 | 914.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 937.65 | 912.82 | 916.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 937.65 | 912.82 | 916.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 943.50 | 922.67 | 920.51 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 894.90 | 920.47 | 921.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 891.35 | 908.53 | 914.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 900.00 | 888.22 | 898.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 900.00 | 888.22 | 898.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 900.00 | 888.22 | 898.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 895.75 | 888.22 | 898.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 897.30 | 896.55 | 899.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 892.00 | 898.29 | 899.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 898.00 | 898.29 | 899.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 898.00 | 898.23 | 899.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 882.10 | 898.23 | 899.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 914.70 | 895.16 | 896.11 | SL hit (close>static) qty=1.00 sl=906.25 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 912.00 | 898.53 | 897.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 938.95 | 906.61 | 901.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 934.05 | 936.45 | 922.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 949.10 | 933.38 | 926.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 970.70 | 969.68 | 962.44 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:15:00 | 996.56 | 975.11 | 965.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 987.65 | 975.11 | 965.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 989.80 | 990.03 | 977.88 | SL hit (close<ema200) qty=0.50 sl=990.03 alert=retest1 |

### Cycle 144 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 1038.60 | 1054.35 | 1054.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 15:15:00 | 1034.00 | 1050.28 | 1052.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 1032.30 | 1031.10 | 1039.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:45:00 | 1034.05 | 1031.10 | 1039.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 1041.10 | 1032.67 | 1038.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 1038.65 | 1032.67 | 1038.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 1042.10 | 1034.55 | 1039.06 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 1060.35 | 1043.57 | 1042.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 1073.50 | 1057.50 | 1050.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 11:15:00 | 1054.00 | 1065.04 | 1056.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 11:15:00 | 1054.00 | 1065.04 | 1056.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1054.00 | 1065.04 | 1056.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 1052.80 | 1065.04 | 1056.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1059.00 | 1063.83 | 1057.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 14:15:00 | 1067.85 | 1062.74 | 1057.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 12:45:00 | 1062.05 | 1064.81 | 1061.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:30:00 | 1064.95 | 1063.09 | 1060.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 1063.05 | 1064.34 | 1061.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1052.20 | 1061.92 | 1061.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 1052.20 | 1061.92 | 1061.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 1046.95 | 1058.92 | 1059.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 1046.95 | 1058.92 | 1059.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1034.90 | 1052.48 | 1056.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1024.30 | 1023.50 | 1034.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1024.30 | 1023.50 | 1034.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1024.30 | 1023.50 | 1034.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 1034.70 | 1023.50 | 1034.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1040.50 | 1026.90 | 1034.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 1040.50 | 1026.90 | 1034.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1035.30 | 1028.58 | 1034.89 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 1077.40 | 1041.63 | 1038.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 1082.40 | 1070.35 | 1061.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1075.00 | 1083.20 | 1073.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1075.00 | 1083.20 | 1073.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1075.00 | 1083.20 | 1073.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 1073.30 | 1083.20 | 1073.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1080.50 | 1082.66 | 1074.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 1081.60 | 1082.66 | 1074.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 1064.90 | 1077.02 | 1072.95 | SL hit (close<static) qty=1.00 sl=1068.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 09:45:00 | 535.55 | 2024-05-16 14:15:00 | 536.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-05-16 10:45:00 | 536.00 | 2024-05-16 14:15:00 | 536.95 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-05-21 09:15:00 | 522.40 | 2024-05-22 12:15:00 | 535.05 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-05-27 09:15:00 | 537.65 | 2024-05-28 13:15:00 | 528.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-27 10:15:00 | 536.60 | 2024-05-28 13:15:00 | 528.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-05-28 09:45:00 | 536.45 | 2024-05-28 13:15:00 | 528.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-06-10 09:15:00 | 595.40 | 2024-06-14 09:15:00 | 654.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-24 14:00:00 | 652.45 | 2024-06-27 15:15:00 | 619.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 09:15:00 | 651.00 | 2024-06-27 15:15:00 | 618.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 11:30:00 | 652.05 | 2024-06-27 15:15:00 | 619.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-26 12:15:00 | 652.00 | 2024-06-27 15:15:00 | 619.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 14:00:00 | 652.45 | 2024-06-28 13:15:00 | 640.00 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2024-06-26 09:15:00 | 651.00 | 2024-06-28 13:15:00 | 640.00 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2024-06-26 11:30:00 | 652.05 | 2024-06-28 13:15:00 | 640.00 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2024-06-26 12:15:00 | 652.00 | 2024-06-28 13:15:00 | 640.00 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2024-07-01 11:15:00 | 639.85 | 2024-07-02 15:15:00 | 652.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-07-08 13:45:00 | 656.55 | 2024-07-10 12:15:00 | 648.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-07-26 09:15:00 | 701.90 | 2024-07-31 12:15:00 | 705.95 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-07-26 09:45:00 | 702.10 | 2024-07-31 12:15:00 | 705.95 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2024-08-06 10:15:00 | 660.20 | 2024-08-09 09:15:00 | 684.95 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-08-29 10:15:00 | 727.00 | 2024-08-30 12:15:00 | 739.95 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-08-29 13:00:00 | 726.70 | 2024-08-30 12:15:00 | 739.95 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-09-02 15:00:00 | 747.30 | 2024-09-04 13:15:00 | 738.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-03 13:30:00 | 747.80 | 2024-09-04 13:15:00 | 738.45 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-09-03 15:15:00 | 748.50 | 2024-09-04 13:15:00 | 738.45 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-09-30 09:30:00 | 722.30 | 2024-10-01 13:15:00 | 741.85 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-09-30 10:45:00 | 722.40 | 2024-10-01 13:15:00 | 741.85 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-10-09 11:45:00 | 701.90 | 2024-10-09 14:15:00 | 706.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-10-21 09:15:00 | 769.35 | 2024-10-22 10:15:00 | 748.40 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-10-21 10:30:00 | 769.50 | 2024-10-22 10:15:00 | 748.40 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-10-23 14:15:00 | 735.55 | 2024-10-28 09:15:00 | 752.20 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-10-24 09:30:00 | 737.30 | 2024-10-28 09:15:00 | 752.20 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-10-24 11:30:00 | 738.30 | 2024-10-28 09:15:00 | 752.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-10-24 13:15:00 | 737.10 | 2024-10-28 09:15:00 | 752.20 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-10-31 11:45:00 | 788.45 | 2024-11-08 11:15:00 | 801.00 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2024-11-04 13:45:00 | 792.70 | 2024-11-08 11:15:00 | 801.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2024-11-12 11:45:00 | 806.90 | 2024-11-14 12:15:00 | 807.70 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-11-12 12:30:00 | 806.60 | 2024-11-14 12:15:00 | 807.70 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-11-19 09:15:00 | 826.60 | 2024-12-02 09:15:00 | 900.46 | TARGET_HIT | 1.00 | 8.94% |
| BUY | retest2 | 2024-11-21 09:15:00 | 818.60 | 2024-12-02 09:15:00 | 896.61 | TARGET_HIT | 1.00 | 9.53% |
| BUY | retest2 | 2024-11-21 10:45:00 | 815.10 | 2024-12-02 10:15:00 | 909.26 | TARGET_HIT | 1.00 | 11.55% |
| SELL | retest2 | 2024-12-10 09:15:00 | 841.70 | 2024-12-17 09:15:00 | 832.70 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2024-12-10 10:00:00 | 843.05 | 2024-12-17 09:15:00 | 832.70 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2024-12-10 13:45:00 | 843.00 | 2024-12-17 09:15:00 | 832.70 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2024-12-10 15:15:00 | 840.00 | 2024-12-17 09:15:00 | 832.70 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2024-12-26 11:15:00 | 794.15 | 2024-12-26 12:15:00 | 804.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-12-31 14:15:00 | 840.00 | 2025-01-02 15:15:00 | 829.10 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-01 09:15:00 | 851.00 | 2025-01-02 15:15:00 | 829.10 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-01-02 09:15:00 | 846.95 | 2025-01-02 15:15:00 | 829.10 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-01-08 09:15:00 | 810.50 | 2025-01-09 14:15:00 | 769.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 810.50 | 2025-01-13 13:15:00 | 729.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-01-23 11:30:00 | 715.85 | 2025-01-27 09:15:00 | 680.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-23 14:15:00 | 716.15 | 2025-01-27 09:15:00 | 680.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-23 15:15:00 | 717.00 | 2025-01-27 09:15:00 | 681.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 706.30 | 2025-01-27 09:15:00 | 670.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:00:00 | 705.60 | 2025-01-27 09:15:00 | 670.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 15:15:00 | 705.00 | 2025-01-27 09:15:00 | 669.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-23 11:30:00 | 715.85 | 2025-01-27 13:15:00 | 644.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-01-23 14:15:00 | 716.15 | 2025-01-27 13:15:00 | 644.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-01-23 15:15:00 | 717.00 | 2025-01-27 13:15:00 | 645.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 12:45:00 | 706.30 | 2025-01-27 13:15:00 | 635.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 14:00:00 | 705.60 | 2025-01-27 13:15:00 | 635.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 15:15:00 | 705.00 | 2025-01-27 13:15:00 | 634.50 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 11:30:00 | 707.70 | 2025-02-10 09:15:00 | 673.95 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2025-02-07 12:30:00 | 704.55 | 2025-02-10 09:15:00 | 673.95 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2025-02-07 13:15:00 | 705.60 | 2025-02-10 09:15:00 | 673.95 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-02-18 09:15:00 | 640.80 | 2025-02-19 14:15:00 | 652.95 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-02-19 10:30:00 | 644.90 | 2025-02-19 14:15:00 | 652.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-03-17 11:30:00 | 593.00 | 2025-03-19 09:15:00 | 607.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-03-18 10:30:00 | 592.55 | 2025-03-19 09:15:00 | 607.80 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-03-18 12:30:00 | 593.00 | 2025-03-19 09:15:00 | 607.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-05-05 09:15:00 | 649.10 | 2025-05-05 14:15:00 | 660.75 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-05-05 10:45:00 | 652.80 | 2025-05-05 14:15:00 | 660.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-16 10:45:00 | 693.00 | 2025-05-29 15:15:00 | 725.00 | STOP_HIT | 1.00 | 4.62% |
| BUY | retest2 | 2025-05-16 11:45:00 | 691.70 | 2025-05-29 15:15:00 | 725.00 | STOP_HIT | 1.00 | 4.81% |
| BUY | retest1 | 2025-06-03 09:15:00 | 755.00 | 2025-06-05 14:15:00 | 752.65 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-06-04 09:15:00 | 758.00 | 2025-06-11 15:15:00 | 782.00 | STOP_HIT | 1.00 | 3.17% |
| BUY | retest2 | 2025-06-05 11:30:00 | 757.30 | 2025-06-11 15:15:00 | 782.00 | STOP_HIT | 1.00 | 3.26% |
| BUY | retest2 | 2025-06-05 12:45:00 | 758.60 | 2025-06-11 15:15:00 | 782.00 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2025-06-06 09:15:00 | 761.50 | 2025-06-11 15:15:00 | 782.00 | STOP_HIT | 1.00 | 2.69% |
| SELL | retest2 | 2025-06-18 11:15:00 | 762.10 | 2025-06-23 15:15:00 | 756.05 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-06-18 12:30:00 | 757.00 | 2025-06-23 15:15:00 | 756.05 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-07-03 10:30:00 | 800.85 | 2025-07-17 09:15:00 | 880.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 09:15:00 | 863.95 | 2025-08-06 10:15:00 | 848.45 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-02 14:15:00 | 829.00 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-09-02 15:00:00 | 827.60 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-09-03 12:45:00 | 828.70 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-09-03 13:30:00 | 828.25 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-09-04 11:00:00 | 829.55 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-09-05 12:15:00 | 830.00 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-09-05 14:00:00 | 830.50 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-09-05 14:45:00 | 827.70 | 2025-09-08 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-09-22 09:15:00 | 816.65 | 2025-09-26 14:15:00 | 775.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 816.65 | 2025-09-29 09:15:00 | 790.05 | STOP_HIT | 0.50 | 3.26% |
| BUY | retest2 | 2025-10-17 11:45:00 | 869.70 | 2025-10-20 10:15:00 | 852.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-11-26 09:15:00 | 720.15 | 2025-11-26 14:15:00 | 730.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-11-27 15:15:00 | 730.00 | 2025-12-02 09:15:00 | 728.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-28 12:00:00 | 730.15 | 2025-12-02 09:15:00 | 728.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-11-28 13:00:00 | 730.45 | 2025-12-02 09:15:00 | 728.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-12-01 10:45:00 | 730.50 | 2025-12-02 12:15:00 | 723.35 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-01 12:15:00 | 734.50 | 2025-12-02 12:15:00 | 723.35 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-12-01 13:15:00 | 735.45 | 2025-12-02 12:15:00 | 723.35 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-01 15:00:00 | 735.10 | 2025-12-02 12:15:00 | 723.35 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-12-08 10:00:00 | 724.70 | 2025-12-08 11:15:00 | 726.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-08 11:15:00 | 724.65 | 2025-12-08 11:15:00 | 726.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-12-16 10:30:00 | 756.15 | 2025-12-19 12:15:00 | 755.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-01-07 11:30:00 | 856.00 | 2026-01-08 09:15:00 | 841.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-07 14:00:00 | 855.00 | 2026-01-08 09:15:00 | 841.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-08 09:15:00 | 857.25 | 2026-01-08 09:15:00 | 841.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest1 | 2026-01-19 10:45:00 | 829.25 | 2026-01-19 11:15:00 | 823.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-23 10:30:00 | 784.60 | 2026-01-30 09:15:00 | 745.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 09:45:00 | 787.55 | 2026-01-30 09:15:00 | 748.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 10:15:00 | 786.05 | 2026-01-30 09:15:00 | 746.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 13:30:00 | 788.50 | 2026-01-30 09:15:00 | 749.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:30:00 | 784.60 | 2026-01-30 13:15:00 | 763.65 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2026-01-27 09:45:00 | 787.55 | 2026-01-30 13:15:00 | 763.65 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2026-01-27 10:15:00 | 786.05 | 2026-01-30 13:15:00 | 763.65 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-01-27 13:30:00 | 788.50 | 2026-01-30 13:15:00 | 763.65 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2026-01-28 10:45:00 | 775.50 | 2026-02-02 14:15:00 | 775.80 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-01-28 11:30:00 | 775.65 | 2026-02-02 14:15:00 | 775.80 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-01-28 12:00:00 | 775.95 | 2026-02-02 14:15:00 | 775.80 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2026-01-28 12:45:00 | 776.00 | 2026-02-02 14:15:00 | 775.80 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2026-01-29 10:15:00 | 771.00 | 2026-02-02 14:15:00 | 775.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-06 11:15:00 | 809.40 | 2026-02-17 09:15:00 | 887.70 | TARGET_HIT | 1.00 | 9.67% |
| BUY | retest2 | 2026-02-06 13:30:00 | 807.00 | 2026-02-17 10:15:00 | 890.34 | TARGET_HIT | 1.00 | 10.33% |
| SELL | retest2 | 2026-03-16 11:30:00 | 961.45 | 2026-03-17 10:15:00 | 987.15 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-16 12:00:00 | 959.60 | 2026-03-17 10:15:00 | 987.15 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-03-17 10:15:00 | 960.00 | 2026-03-17 10:15:00 | 987.15 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2026-04-01 10:15:00 | 895.75 | 2026-04-02 14:15:00 | 914.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-04-01 13:30:00 | 897.30 | 2026-04-02 15:15:00 | 912.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-04-01 14:45:00 | 892.00 | 2026-04-02 15:15:00 | 912.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-04-01 15:15:00 | 898.00 | 2026-04-02 15:15:00 | 912.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-02 09:15:00 | 882.10 | 2026-04-02 15:15:00 | 912.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest1 | 2026-04-08 09:15:00 | 949.10 | 2026-04-10 11:15:00 | 996.56 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 949.10 | 2026-04-13 09:15:00 | 989.80 | STOP_HIT | 0.50 | 4.29% |
| BUY | retest2 | 2026-04-10 11:30:00 | 987.65 | 2026-04-21 14:15:00 | 1038.60 | STOP_HIT | 1.00 | 5.16% |
| BUY | retest2 | 2026-04-13 09:45:00 | 990.00 | 2026-04-21 14:15:00 | 1038.60 | STOP_HIT | 1.00 | 4.91% |
| BUY | retest2 | 2026-04-27 14:15:00 | 1067.85 | 2026-04-29 11:15:00 | 1046.95 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-28 12:45:00 | 1062.05 | 2026-04-29 11:15:00 | 1046.95 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-28 14:30:00 | 1064.95 | 2026-04-29 11:15:00 | 1046.95 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-04-29 10:00:00 | 1063.05 | 2026-04-29 11:15:00 | 1046.95 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-05-08 11:15:00 | 1081.60 | 2026-05-08 12:15:00 | 1064.90 | STOP_HIT | 1.00 | -1.54% |
