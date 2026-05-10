# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 715.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 47 |
| ALERT2 | 47 |
| ALERT2_SKIP | 26 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 60 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 48
- **Target hits / Stop hits / Partials:** 3 / 62 / 6
- **Avg / median % per leg:** 0.31% / -1.03%
- **Sum % (uncompounded):** 21.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 9 | 32.1% | 3 | 25 | 0 | 0.65% | 18.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.22% | -6.7% |
| BUY @ 3rd Alert (retest2) | 25 | 9 | 36.0% | 3 | 22 | 0 | 1.00% | 24.9% |
| SELL (all) | 43 | 14 | 32.6% | 0 | 37 | 6 | 0.09% | 3.7% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.39% | -7.2% |
| SELL @ 3rd Alert (retest2) | 40 | 14 | 35.0% | 0 | 34 | 6 | 0.27% | 10.9% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.31% | -13.8% |
| retest2 (combined) | 65 | 23 | 35.4% | 3 | 56 | 6 | 0.55% | 35.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 646.85 | 641.08 | 640.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 649.85 | 643.82 | 642.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 652.15 | 653.71 | 649.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 652.15 | 653.71 | 649.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 652.15 | 653.71 | 649.70 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 640.00 | 647.48 | 647.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 15:15:00 | 639.80 | 645.94 | 647.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 14:15:00 | 645.45 | 635.69 | 638.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 14:15:00 | 645.45 | 635.69 | 638.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 645.45 | 635.69 | 638.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 645.45 | 635.69 | 638.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 643.00 | 637.15 | 638.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 647.20 | 637.15 | 638.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 636.00 | 638.42 | 638.92 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 660.30 | 643.17 | 640.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 669.70 | 658.28 | 650.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 672.35 | 672.99 | 667.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 672.35 | 672.99 | 667.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 665.50 | 670.65 | 668.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 664.50 | 670.65 | 668.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 662.00 | 668.92 | 667.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 661.25 | 668.92 | 667.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 657.90 | 666.72 | 666.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 11:15:00 | 654.90 | 664.35 | 665.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 660.00 | 657.42 | 661.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 660.00 | 657.42 | 661.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 660.00 | 657.42 | 661.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 660.00 | 657.42 | 661.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 653.00 | 655.29 | 658.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:30:00 | 648.20 | 654.26 | 657.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 667.50 | 655.76 | 656.89 | SL hit (close>static) qty=1.00 sl=659.80 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 666.70 | 657.95 | 657.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 672.95 | 660.95 | 659.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 11:15:00 | 674.20 | 674.76 | 668.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:45:00 | 673.80 | 674.76 | 668.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 667.80 | 673.63 | 669.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 667.80 | 673.63 | 669.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 665.50 | 672.01 | 669.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 669.10 | 672.01 | 669.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:00:00 | 669.15 | 671.89 | 671.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 669.00 | 671.31 | 671.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 669.00 | 671.31 | 671.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 669.00 | 671.31 | 671.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 664.00 | 669.64 | 670.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 661.00 | 660.65 | 664.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 661.00 | 660.65 | 664.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 661.00 | 660.65 | 664.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 643.10 | 656.70 | 659.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:15:00 | 651.15 | 654.67 | 658.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:15:00 | 650.75 | 654.17 | 657.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 650.30 | 652.19 | 655.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 648.30 | 651.41 | 654.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 664.65 | 656.26 | 656.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 672.00 | 659.41 | 657.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 658.50 | 664.39 | 661.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 658.50 | 664.39 | 661.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 658.50 | 664.39 | 661.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 658.50 | 664.39 | 661.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 660.80 | 663.67 | 661.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 09:45:00 | 669.70 | 665.15 | 662.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 13:15:00 | 662.20 | 669.81 | 670.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 662.20 | 669.81 | 670.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 660.00 | 667.85 | 669.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 15:15:00 | 666.30 | 662.57 | 665.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 15:15:00 | 666.30 | 662.57 | 665.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 666.30 | 662.57 | 665.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 659.00 | 661.40 | 664.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 09:15:00 | 626.05 | 637.72 | 646.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 632.50 | 629.28 | 636.71 | SL hit (close>ema200) qty=0.50 sl=629.28 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 644.35 | 636.06 | 635.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 651.60 | 640.34 | 637.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 665.70 | 667.06 | 658.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 665.70 | 667.06 | 658.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 667.30 | 670.84 | 666.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 667.30 | 670.84 | 666.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 662.15 | 669.10 | 666.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 662.15 | 669.10 | 666.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 668.55 | 668.99 | 666.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:00:00 | 671.95 | 669.58 | 667.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:30:00 | 670.75 | 674.24 | 671.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:00:00 | 671.15 | 672.87 | 671.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 667.10 | 670.56 | 670.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 667.10 | 670.56 | 670.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 667.10 | 670.56 | 670.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 667.10 | 670.56 | 670.93 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 674.90 | 671.37 | 671.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 683.40 | 673.72 | 672.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 675.85 | 680.31 | 677.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 675.85 | 680.31 | 677.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 675.85 | 680.31 | 677.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 678.05 | 680.31 | 677.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 674.60 | 679.17 | 676.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:30:00 | 676.95 | 678.15 | 676.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:15:00 | 676.90 | 678.15 | 676.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:00:00 | 682.20 | 678.96 | 677.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 13:15:00 | 744.65 | 730.30 | 721.36 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-16 13:15:00 | 744.59 | 730.30 | 721.36 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-07-18 09:15:00 | 750.42 | 742.21 | 734.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 754.20 | 765.89 | 767.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 12:15:00 | 751.05 | 760.81 | 764.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 756.85 | 755.57 | 760.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 756.85 | 755.57 | 760.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 756.85 | 755.57 | 760.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 747.90 | 752.04 | 755.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 12:15:00 | 760.75 | 756.31 | 755.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 12:15:00 | 760.75 | 756.31 | 755.91 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 750.50 | 755.19 | 755.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 748.40 | 753.83 | 755.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 756.60 | 745.69 | 748.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 756.60 | 745.69 | 748.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 756.60 | 745.69 | 748.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:15:00 | 757.20 | 745.69 | 748.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 744.90 | 745.53 | 747.98 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 13:15:00 | 763.60 | 751.45 | 750.24 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 734.90 | 747.02 | 748.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 711.50 | 736.09 | 741.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 721.80 | 716.43 | 725.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:45:00 | 721.50 | 716.43 | 725.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 726.30 | 719.50 | 724.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 726.30 | 719.50 | 724.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 726.00 | 720.80 | 724.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 721.35 | 720.80 | 724.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 727.00 | 722.04 | 724.92 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 738.45 | 728.76 | 727.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 742.45 | 731.50 | 728.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 732.70 | 733.39 | 730.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:15:00 | 731.50 | 733.39 | 730.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 729.40 | 732.60 | 730.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 729.40 | 732.60 | 730.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 731.75 | 732.43 | 730.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 736.00 | 732.25 | 731.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 737.60 | 732.88 | 731.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 745.25 | 756.96 | 757.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 745.25 | 756.96 | 757.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 745.25 | 756.96 | 757.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 742.55 | 754.08 | 756.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 744.85 | 741.01 | 744.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 744.85 | 741.01 | 744.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 744.85 | 741.01 | 744.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 744.85 | 741.01 | 744.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 756.05 | 744.02 | 745.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 757.50 | 744.02 | 745.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 756.65 | 746.55 | 746.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 756.60 | 746.55 | 746.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 15:15:00 | 756.15 | 748.47 | 747.64 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 739.85 | 747.02 | 747.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 737.65 | 742.93 | 745.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 736.25 | 731.75 | 736.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 736.25 | 731.75 | 736.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 736.25 | 731.75 | 736.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 736.25 | 731.75 | 736.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 732.00 | 731.80 | 736.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:30:00 | 729.00 | 731.44 | 735.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 730.25 | 731.20 | 735.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:30:00 | 729.55 | 731.52 | 734.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 727.80 | 731.52 | 734.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 730.65 | 730.10 | 733.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 741.90 | 734.81 | 734.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 742.60 | 736.36 | 735.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 737.20 | 738.48 | 736.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 13:15:00 | 737.20 | 738.48 | 736.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 737.20 | 738.48 | 736.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 734.60 | 738.48 | 736.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 739.40 | 738.66 | 737.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 739.40 | 738.66 | 737.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 741.05 | 743.11 | 740.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 741.05 | 743.11 | 740.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 737.00 | 741.89 | 740.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:45:00 | 738.40 | 741.89 | 740.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 736.15 | 740.74 | 739.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 741.20 | 741.91 | 740.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:00:00 | 740.50 | 741.63 | 740.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 12:30:00 | 740.85 | 740.80 | 740.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 740.20 | 740.80 | 740.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 741.55 | 740.95 | 740.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 741.10 | 740.95 | 740.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 743.00 | 741.36 | 740.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 740.05 | 741.36 | 740.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 740.05 | 741.10 | 740.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 737.80 | 741.10 | 740.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 742.20 | 741.32 | 740.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 733.20 | 738.89 | 739.62 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 741.15 | 739.53 | 739.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 747.60 | 741.94 | 740.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 760.55 | 762.13 | 757.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 760.55 | 762.13 | 757.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 758.10 | 761.32 | 757.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:15:00 | 758.05 | 761.32 | 757.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 756.95 | 760.45 | 757.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 756.95 | 760.45 | 757.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 760.00 | 760.36 | 757.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 748.35 | 760.36 | 757.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 753.95 | 759.08 | 757.54 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 754.10 | 756.19 | 756.44 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 758.15 | 756.25 | 756.11 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 754.30 | 755.86 | 755.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 11:15:00 | 746.25 | 753.68 | 754.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 741.95 | 740.59 | 744.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 741.95 | 740.59 | 744.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 741.95 | 740.59 | 744.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 739.00 | 740.59 | 744.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 743.15 | 741.10 | 744.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 744.55 | 741.10 | 744.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 745.15 | 742.17 | 744.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 744.20 | 742.17 | 744.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 746.60 | 743.05 | 744.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 746.10 | 743.05 | 744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 745.00 | 743.44 | 744.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 745.90 | 743.44 | 744.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 745.00 | 743.75 | 744.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 741.65 | 743.75 | 744.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 742.80 | 743.56 | 744.60 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 747.55 | 745.61 | 745.43 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 739.15 | 744.32 | 744.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 15:15:00 | 737.10 | 741.59 | 743.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 736.25 | 735.92 | 738.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:15:00 | 729.45 | 735.92 | 738.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 736.10 | 730.56 | 733.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 736.10 | 730.56 | 733.76 | SL hit (close>ema400) qty=1.00 sl=733.76 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 736.10 | 730.56 | 733.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 733.85 | 731.22 | 733.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:15:00 | 737.75 | 731.22 | 733.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 732.45 | 731.47 | 733.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 728.10 | 731.68 | 733.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 730.20 | 728.50 | 730.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 691.69 | 700.92 | 708.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 09:15:00 | 693.69 | 700.92 | 708.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 691.35 | 687.70 | 694.60 | SL hit (close>ema200) qty=0.50 sl=687.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 691.35 | 687.70 | 694.60 | SL hit (close>ema200) qty=0.50 sl=687.70 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 716.45 | 697.46 | 697.18 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 705.25 | 707.26 | 707.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 704.30 | 706.61 | 707.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 697.05 | 696.51 | 700.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 697.05 | 696.51 | 700.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 700.15 | 697.48 | 700.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:15:00 | 700.50 | 697.48 | 700.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 700.15 | 698.01 | 700.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:15:00 | 701.45 | 698.01 | 700.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 701.45 | 698.70 | 700.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 697.10 | 698.70 | 700.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 698.50 | 699.23 | 700.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 704.65 | 700.31 | 700.83 | SL hit (close>static) qty=1.00 sl=703.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 704.65 | 700.31 | 700.83 | SL hit (close>static) qty=1.00 sl=703.55 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 711.80 | 702.61 | 701.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 715.30 | 706.96 | 704.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 708.30 | 710.79 | 707.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 708.30 | 710.79 | 707.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 708.30 | 710.79 | 707.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 706.55 | 710.79 | 707.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 712.40 | 711.11 | 707.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 714.90 | 711.24 | 708.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 714.50 | 711.05 | 708.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 717.50 | 724.84 | 725.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 717.50 | 724.84 | 725.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 717.50 | 724.84 | 725.18 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 727.00 | 725.33 | 725.31 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 719.70 | 724.21 | 724.80 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 726.70 | 722.13 | 721.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 730.15 | 725.29 | 723.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 725.65 | 727.62 | 725.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 725.65 | 727.62 | 725.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 725.65 | 727.62 | 725.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 725.65 | 727.62 | 725.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 730.00 | 728.10 | 725.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 733.60 | 727.61 | 726.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 725.20 | 727.13 | 726.17 | SL hit (close<static) qty=1.00 sl=725.25 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 720.45 | 725.35 | 725.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 720.00 | 723.80 | 724.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 726.10 | 723.78 | 724.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 726.10 | 723.78 | 724.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 726.10 | 723.78 | 724.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 727.00 | 723.78 | 724.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 723.30 | 723.68 | 724.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 724.45 | 723.68 | 724.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 720.05 | 722.96 | 724.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:30:00 | 723.40 | 722.96 | 724.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 724.25 | 723.22 | 724.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 724.25 | 723.22 | 724.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 729.25 | 724.42 | 724.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:45:00 | 730.35 | 724.42 | 724.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 726.95 | 724.93 | 724.74 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 721.65 | 724.26 | 724.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 718.20 | 723.05 | 724.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 718.30 | 715.15 | 718.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 718.30 | 715.15 | 718.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 720.50 | 716.22 | 719.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 720.50 | 716.22 | 719.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 720.10 | 717.00 | 719.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 720.90 | 717.00 | 719.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 720.90 | 717.78 | 719.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 714.60 | 717.78 | 719.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 10:15:00 | 729.65 | 721.09 | 720.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 11:15:00 | 731.65 | 723.20 | 721.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 15:15:00 | 725.10 | 726.00 | 723.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 717.65 | 726.00 | 723.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 708.15 | 722.43 | 722.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 708.15 | 722.43 | 722.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 704.50 | 718.85 | 720.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 698.80 | 714.84 | 718.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 683.00 | 678.51 | 690.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 683.00 | 678.51 | 690.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 689.00 | 683.58 | 688.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 684.00 | 683.58 | 688.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 671.85 | 666.69 | 666.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 671.85 | 666.69 | 666.54 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 664.70 | 666.94 | 666.94 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 670.80 | 667.71 | 667.29 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 664.85 | 667.05 | 667.18 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 669.10 | 667.46 | 667.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 681.20 | 670.10 | 668.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 665.15 | 670.69 | 669.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 665.15 | 670.69 | 669.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 665.15 | 670.69 | 669.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 665.15 | 670.69 | 669.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 670.00 | 670.56 | 669.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 665.10 | 670.56 | 669.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 678.25 | 672.09 | 670.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:45:00 | 679.10 | 673.56 | 671.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 679.60 | 676.32 | 673.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 695.45 | 699.93 | 700.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 695.45 | 699.93 | 700.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 695.45 | 699.93 | 700.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 686.60 | 696.95 | 698.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 681.55 | 679.26 | 685.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 681.55 | 679.26 | 685.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 681.55 | 679.26 | 685.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 668.00 | 677.06 | 682.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 667.20 | 665.74 | 665.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 667.20 | 665.74 | 665.69 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 662.40 | 665.04 | 665.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 660.80 | 664.19 | 664.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 640.95 | 638.87 | 643.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 640.95 | 638.87 | 643.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 640.95 | 638.87 | 643.80 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 649.30 | 645.53 | 645.26 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 13:15:00 | 642.00 | 644.82 | 644.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 15:15:00 | 640.50 | 643.70 | 644.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 15:15:00 | 642.00 | 641.73 | 642.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 638.75 | 641.73 | 642.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 634.10 | 640.21 | 642.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 633.25 | 640.21 | 642.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 632.70 | 636.79 | 639.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 601.59 | 612.25 | 620.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:15:00 | 601.07 | 612.25 | 620.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 609.60 | 607.28 | 614.21 | SL hit (close>ema200) qty=0.50 sl=607.28 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 609.60 | 607.28 | 614.21 | SL hit (close>ema200) qty=0.50 sl=607.28 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 615.95 | 613.06 | 612.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 627.80 | 616.79 | 614.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 642.55 | 648.36 | 639.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 642.55 | 648.36 | 639.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 635.00 | 645.69 | 638.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 634.95 | 645.69 | 638.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 638.85 | 644.32 | 638.85 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 627.85 | 635.17 | 635.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 621.80 | 626.84 | 630.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 620.55 | 616.58 | 621.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:45:00 | 620.80 | 616.58 | 621.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 621.70 | 617.61 | 621.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 622.50 | 617.61 | 621.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 620.25 | 618.13 | 621.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 608.00 | 618.97 | 621.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 615.60 | 614.13 | 616.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:00:00 | 615.00 | 614.13 | 616.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 615.40 | 614.43 | 616.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 620.45 | 615.63 | 616.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 620.45 | 615.63 | 616.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 620.00 | 616.51 | 617.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 621.00 | 616.51 | 617.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 622.20 | 617.92 | 617.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 625.25 | 620.07 | 618.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 623.10 | 624.43 | 621.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 623.10 | 624.43 | 621.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 626.40 | 624.83 | 622.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 622.10 | 624.86 | 622.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 622.25 | 624.34 | 622.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 622.25 | 624.34 | 622.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 618.65 | 623.20 | 622.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 617.85 | 623.20 | 622.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 616.25 | 621.81 | 621.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 616.25 | 621.81 | 621.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 613.75 | 620.20 | 620.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 612.95 | 617.90 | 619.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 12:15:00 | 612.20 | 611.92 | 614.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-20 13:00:00 | 612.20 | 611.92 | 614.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 617.05 | 612.73 | 614.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 617.05 | 612.73 | 614.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 615.05 | 613.20 | 614.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 603.10 | 613.20 | 614.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 13:15:00 | 611.95 | 602.92 | 602.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 611.95 | 602.92 | 602.01 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 602.00 | 604.58 | 604.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 15:15:00 | 601.05 | 603.80 | 604.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 602.75 | 601.92 | 603.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 13:15:00 | 602.75 | 601.92 | 603.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 602.75 | 601.92 | 603.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:45:00 | 602.75 | 601.92 | 603.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 606.00 | 602.73 | 603.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 606.00 | 602.73 | 603.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 601.00 | 602.39 | 603.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 598.80 | 602.39 | 603.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 598.50 | 601.09 | 602.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 598.30 | 600.69 | 601.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 596.20 | 599.52 | 601.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 595.95 | 597.49 | 599.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 595.95 | 597.49 | 599.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 596.70 | 593.29 | 596.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 596.70 | 593.29 | 596.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 590.00 | 592.63 | 595.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 613.55 | 592.63 | 595.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | SL hit (close>static) qty=1.00 sl=608.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | SL hit (close>static) qty=1.00 sl=608.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | SL hit (close>static) qty=1.00 sl=608.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 608.75 | 595.85 | 597.04 | SL hit (close>static) qty=1.00 sl=608.45 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 610.50 | 598.78 | 598.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 621.15 | 607.03 | 602.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 605.10 | 609.12 | 605.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 605.10 | 609.12 | 605.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 605.10 | 609.12 | 605.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 605.10 | 609.12 | 605.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 607.05 | 608.71 | 605.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 606.20 | 608.71 | 605.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 605.50 | 608.07 | 605.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 605.50 | 608.07 | 605.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 608.20 | 608.09 | 605.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 609.70 | 608.42 | 606.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 603.35 | 606.69 | 605.98 | SL hit (close<static) qty=1.00 sl=604.85 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 598.45 | 604.84 | 605.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 596.90 | 602.21 | 604.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 604.80 | 599.63 | 601.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 604.80 | 599.63 | 601.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 604.80 | 599.63 | 601.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 604.80 | 599.63 | 601.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 630.35 | 605.78 | 604.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 646.50 | 613.92 | 608.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 648.30 | 651.92 | 637.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:30:00 | 648.85 | 651.92 | 637.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 678.00 | 691.45 | 684.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:00:00 | 698.10 | 688.55 | 685.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 697.90 | 702.65 | 701.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 14:15:00 | 695.00 | 699.60 | 700.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 14:15:00 | 695.00 | 699.60 | 700.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 695.00 | 699.60 | 700.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 688.00 | 697.28 | 699.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 710.70 | 699.96 | 700.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 710.70 | 699.96 | 700.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 710.70 | 699.96 | 700.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 710.70 | 699.96 | 700.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 720.95 | 704.16 | 702.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 725.45 | 712.98 | 707.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 721.60 | 723.44 | 716.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 721.60 | 723.44 | 716.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 744.10 | 739.57 | 731.90 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 721.25 | 732.67 | 733.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 716.85 | 727.53 | 730.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 707.00 | 703.72 | 709.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 699.25 | 703.72 | 709.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:45:00 | 700.60 | 703.01 | 708.59 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 721.85 | 702.44 | 705.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 721.85 | 702.44 | 705.53 | SL hit (close>ema400) qty=1.00 sl=705.53 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 721.85 | 702.44 | 705.53 | SL hit (close>ema400) qty=1.00 sl=705.53 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 721.85 | 702.44 | 705.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 721.00 | 706.15 | 706.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 680.40 | 706.15 | 706.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 646.38 | 659.99 | 672.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 645.25 | 643.01 | 654.70 | SL hit (close>ema200) qty=0.50 sl=643.01 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 673.60 | 656.42 | 654.40 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 650.70 | 658.98 | 659.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 643.05 | 654.49 | 657.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 649.50 | 647.59 | 652.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 649.50 | 647.59 | 652.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 626.00 | 630.86 | 638.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 645.50 | 630.86 | 638.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 630.45 | 630.78 | 638.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 628.40 | 630.67 | 637.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 649.85 | 638.05 | 637.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 649.85 | 638.05 | 637.31 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 634.45 | 637.22 | 637.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 632.75 | 635.74 | 636.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 632.25 | 624.84 | 628.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 632.25 | 624.84 | 628.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 632.25 | 624.84 | 628.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 629.70 | 624.84 | 628.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 627.45 | 625.36 | 628.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:15:00 | 624.15 | 626.54 | 628.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 624.65 | 627.11 | 628.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 619.25 | 626.83 | 628.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 617.30 | 625.15 | 626.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 617.20 | 623.56 | 625.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 631.80 | 624.87 | 624.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 635.45 | 626.99 | 625.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 661.00 | 661.30 | 654.37 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 666.65 | 661.30 | 654.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:45:00 | 664.65 | 661.88 | 655.84 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 12:00:00 | 665.00 | 662.50 | 656.68 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 650.65 | 661.66 | 658.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 650.65 | 661.66 | 658.78 | SL hit (close<ema400) qty=1.00 sl=658.78 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 650.65 | 661.66 | 658.78 | SL hit (close<ema400) qty=1.00 sl=658.78 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 650.65 | 661.66 | 658.78 | SL hit (close<ema400) qty=1.00 sl=658.78 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 655.65 | 661.40 | 658.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 665.90 | 658.43 | 658.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 651.45 | 674.75 | 676.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 651.45 | 674.75 | 676.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 651.45 | 674.75 | 676.78 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 674.65 | 661.89 | 661.39 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 659.90 | 664.26 | 664.44 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 667.00 | 664.84 | 664.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 669.20 | 665.71 | 665.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 665.00 | 666.46 | 665.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 665.00 | 666.46 | 665.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 665.00 | 666.46 | 665.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 665.00 | 666.46 | 665.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 665.05 | 666.18 | 665.54 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 659.55 | 664.83 | 665.03 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 666.00 | 665.16 | 665.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 673.25 | 666.78 | 665.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 675.80 | 676.30 | 672.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 675.80 | 676.30 | 672.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 675.80 | 676.30 | 672.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 673.75 | 676.30 | 672.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 679.45 | 676.93 | 673.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 682.45 | 678.21 | 675.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-29 13:30:00 | 648.20 | 2025-05-30 09:15:00 | 667.50 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-06-03 09:15:00 | 669.10 | 2025-06-04 14:15:00 | 669.00 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-06-04 14:00:00 | 669.15 | 2025-06-04 14:15:00 | 669.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-06-10 09:15:00 | 643.10 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-06-10 11:15:00 | 651.15 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-06-10 12:15:00 | 650.75 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-06-11 09:15:00 | 650.30 | 2025-06-11 13:15:00 | 664.65 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-13 09:45:00 | 669.70 | 2025-06-17 13:15:00 | 662.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-19 09:30:00 | 659.00 | 2025-06-23 09:15:00 | 626.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:30:00 | 659.00 | 2025-06-24 09:15:00 | 632.50 | STOP_HIT | 0.50 | 4.02% |
| BUY | retest2 | 2025-07-01 13:00:00 | 671.95 | 2025-07-03 12:15:00 | 667.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-02 13:30:00 | 670.75 | 2025-07-03 12:15:00 | 667.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-03 10:00:00 | 671.15 | 2025-07-03 12:15:00 | 667.10 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-07 12:30:00 | 676.95 | 2025-07-16 13:15:00 | 744.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-07 13:15:00 | 676.90 | 2025-07-16 13:15:00 | 744.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-07 14:00:00 | 682.20 | 2025-07-18 09:15:00 | 750.42 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 11:00:00 | 747.90 | 2025-08-01 12:15:00 | 760.75 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-08-14 10:00:00 | 736.00 | 2025-08-21 11:15:00 | 745.25 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-08-14 12:00:00 | 737.60 | 2025-08-21 11:15:00 | 745.25 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-09-01 11:30:00 | 729.00 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-01 13:00:00 | 730.25 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-09-01 13:30:00 | 729.55 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-09-01 14:15:00 | 727.80 | 2025-09-02 15:15:00 | 741.90 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-09-05 09:45:00 | 741.20 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-05 11:00:00 | 740.50 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-05 12:30:00 | 740.85 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-05 13:00:00 | 740.20 | 2025-09-08 11:15:00 | 733.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2025-09-24 09:15:00 | 729.45 | 2025-09-25 09:15:00 | 736.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-25 15:15:00 | 728.10 | 2025-10-01 09:15:00 | 691.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 10:00:00 | 730.20 | 2025-10-01 09:15:00 | 693.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 15:15:00 | 728.10 | 2025-10-03 13:15:00 | 691.35 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-09-29 10:00:00 | 730.20 | 2025-10-03 13:15:00 | 691.35 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2025-10-13 09:15:00 | 697.10 | 2025-10-13 10:15:00 | 704.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-13 10:15:00 | 698.50 | 2025-10-13 10:15:00 | 704.65 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-14 14:30:00 | 714.90 | 2025-10-20 14:15:00 | 717.50 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-10-15 09:15:00 | 714.50 | 2025-10-20 14:15:00 | 717.50 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-10-31 09:30:00 | 733.60 | 2025-10-31 10:15:00 | 725.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-13 09:15:00 | 684.00 | 2025-11-20 14:15:00 | 671.85 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-11-25 13:45:00 | 679.10 | 2025-12-08 10:15:00 | 695.45 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2025-11-26 14:45:00 | 679.60 | 2025-12-08 10:15:00 | 695.45 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-12-10 14:15:00 | 668.00 | 2025-12-16 09:15:00 | 667.20 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-12-24 10:15:00 | 633.25 | 2025-12-30 11:15:00 | 601.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 13:30:00 | 632.70 | 2025-12-30 11:15:00 | 601.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 10:15:00 | 633.25 | 2025-12-31 09:15:00 | 609.60 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-12-24 13:30:00 | 632.70 | 2025-12-31 09:15:00 | 609.60 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2026-01-12 09:15:00 | 608.00 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-01-13 09:30:00 | 615.60 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-13 10:00:00 | 615.00 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-13 10:45:00 | 615.40 | 2026-01-13 14:15:00 | 622.20 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-21 09:15:00 | 603.10 | 2026-01-27 13:15:00 | 611.95 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-01 09:15:00 | 598.80 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-01 11:45:00 | 598.50 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-01 13:15:00 | 598.30 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-02-01 14:45:00 | 596.20 | 2026-02-03 09:15:00 | 608.75 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-04 15:00:00 | 609.70 | 2026-02-05 11:15:00 | 603.35 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-17 10:00:00 | 698.10 | 2026-02-20 14:15:00 | 695.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-02-20 09:30:00 | 697.90 | 2026-02-20 14:15:00 | 695.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-03-06 09:15:00 | 699.25 | 2026-03-06 14:15:00 | 721.85 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest1 | 2026-03-06 09:45:00 | 700.60 | 2026-03-06 14:15:00 | 721.85 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2026-03-09 09:15:00 | 680.40 | 2026-03-13 10:15:00 | 646.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 680.40 | 2026-03-16 11:15:00 | 645.25 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-24 10:30:00 | 628.40 | 2026-03-25 14:15:00 | 649.85 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-04-01 12:15:00 | 624.15 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-04-01 13:45:00 | 624.65 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-04-02 09:15:00 | 619.25 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-04-06 09:15:00 | 617.30 | 2026-04-07 10:15:00 | 631.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest1 | 2026-04-10 09:15:00 | 666.65 | 2026-04-13 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2026-04-10 10:45:00 | 664.65 | 2026-04-13 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest1 | 2026-04-10 12:00:00 | 665.00 | 2026-04-13 09:15:00 | 650.65 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-04-13 10:30:00 | 655.65 | 2026-04-23 09:15:00 | 651.45 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-15 09:15:00 | 665.90 | 2026-04-23 09:15:00 | 651.45 | STOP_HIT | 1.00 | -2.17% |
