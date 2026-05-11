# Tata Technologies Ltd. (TATATECH)

## Backtest Summary

- **Window:** 2023-11-30 09:15:00 → 2026-05-08 15:15:00 (4210 bars)
- **Last close:** 632.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 27
- **Target hits / Stop hits / Partials:** 0 / 34 / 8
- **Avg / median % per leg:** 0.59% / -0.68%
- **Sum % (uncompounded):** 24.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.91% | -21.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.91% | -21.0% |
| SELL (all) | 31 | 15 | 48.4% | 0 | 23 | 8 | 1.48% | 45.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 15 | 48.4% | 0 | 23 | 8 | 1.48% | 45.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 15 | 35.7% | 0 | 34 | 8 | 0.59% | 24.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1077.00 | 1025.27 | 1025.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1083.15 | 1032.91 | 1029.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 1057.25 | 1060.70 | 1046.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 12:00:00 | 1057.25 | 1060.70 | 1046.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1064.45 | 1077.78 | 1060.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1058.50 | 1077.78 | 1060.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1071.60 | 1077.71 | 1060.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 1075.40 | 1077.71 | 1060.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 1056.65 | 1077.28 | 1061.02 | SL hit (close<static) qty=1.00 sl=1059.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 998.10 | 1053.74 | 1053.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 994.50 | 1053.16 | 1053.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 11:15:00 | 702.00 | 696.89 | 753.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 702.00 | 696.89 | 753.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 709.05 | 670.63 | 710.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:00:00 | 709.05 | 670.63 | 710.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 705.35 | 670.97 | 710.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:45:00 | 701.80 | 677.87 | 710.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:30:00 | 704.75 | 680.52 | 710.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 672.80 | 680.77 | 710.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:15:00 | 666.71 | 680.64 | 710.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:15:00 | 669.51 | 680.64 | 710.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 639.16 | 674.06 | 701.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 681.70 | 671.68 | 697.76 | SL hit (close>ema200) qty=0.50 sl=671.68 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 775.50 | 713.30 | 713.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 778.70 | 722.84 | 718.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 732.25 | 747.89 | 734.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 737.70 | 747.79 | 734.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 740.00 | 747.71 | 734.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 742.00 | 747.51 | 735.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 13:15:00 | 731.25 | 746.86 | 735.21 | SL hit (close<static) qty=1.00 sl=731.90 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 705.10 | 727.03 | 727.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 703.25 | 723.67 | 725.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 733.40 | 718.81 | 722.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 723.90 | 718.86 | 722.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 719.50 | 720.92 | 723.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 683.52 | 710.92 | 716.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 692.30 | 690.78 | 703.20 | SL hit (close>ema200) qty=0.50 sl=690.78 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-04 11:30:00 | 1075.40 | 2024-10-04 14:15:00 | 1056.65 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-10-10 09:45:00 | 1084.85 | 2024-10-11 10:15:00 | 1056.50 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-10-16 14:45:00 | 1076.00 | 2024-10-18 09:15:00 | 1051.10 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-10-17 10:30:00 | 1076.60 | 2024-10-18 09:15:00 | 1051.10 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-10-18 10:30:00 | 1058.95 | 2024-10-21 14:15:00 | 1052.90 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-10-18 11:30:00 | 1058.25 | 2024-10-22 11:15:00 | 1038.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-10-18 12:00:00 | 1058.15 | 2024-10-22 11:15:00 | 1038.15 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-18 13:00:00 | 1058.10 | 2024-10-22 11:15:00 | 1038.15 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-21 10:00:00 | 1071.60 | 2024-10-22 11:15:00 | 1038.15 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-04-25 09:45:00 | 701.80 | 2025-04-29 09:15:00 | 666.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 14:30:00 | 704.75 | 2025-04-29 09:15:00 | 669.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 09:15:00 | 672.80 | 2025-05-07 09:15:00 | 639.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:45:00 | 701.80 | 2025-05-12 09:15:00 | 681.70 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-04-28 14:30:00 | 704.75 | 2025-05-12 09:15:00 | 681.70 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-04-29 09:15:00 | 672.80 | 2025-05-12 09:15:00 | 681.70 | STOP_HIT | 0.50 | -1.32% |
| SELL | retest2 | 2025-05-13 09:45:00 | 704.65 | 2025-05-15 09:15:00 | 721.65 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-06-16 12:00:00 | 740.00 | 2025-06-18 13:15:00 | 731.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-06-18 09:15:00 | 742.00 | 2025-06-18 13:15:00 | 731.25 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-18 10:00:00 | 719.50 | 2025-08-06 09:15:00 | 683.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 719.50 | 2025-08-20 11:15:00 | 692.30 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-09-18 09:30:00 | 720.60 | 2025-09-24 14:15:00 | 684.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 720.30 | 2025-09-24 14:15:00 | 684.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 14:30:00 | 721.20 | 2025-09-24 14:15:00 | 685.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:30:00 | 720.60 | 2025-10-01 14:15:00 | 691.30 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-09-18 10:00:00 | 720.30 | 2025-10-01 14:15:00 | 691.30 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-09-19 14:30:00 | 721.20 | 2025-10-01 14:15:00 | 691.30 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2025-09-24 09:15:00 | 694.00 | 2025-10-03 13:15:00 | 702.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-24 10:00:00 | 696.00 | 2025-10-03 13:15:00 | 702.35 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-14 09:45:00 | 694.00 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-16 09:30:00 | 695.25 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-23 13:45:00 | 693.50 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-23 15:00:00 | 690.20 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-24 10:15:00 | 692.50 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-24 12:15:00 | 692.65 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-28 13:30:00 | 694.00 | 2025-10-28 15:15:00 | 697.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-10-28 14:30:00 | 693.20 | 2025-10-28 15:15:00 | 697.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-10-31 13:30:00 | 693.85 | 2025-11-03 15:15:00 | 697.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-31 14:45:00 | 694.00 | 2025-11-03 15:15:00 | 697.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-11-04 10:30:00 | 689.95 | 2025-11-12 11:15:00 | 701.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-11-04 11:00:00 | 689.55 | 2025-11-12 11:15:00 | 701.75 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-13 14:00:00 | 690.35 | 2025-12-08 12:15:00 | 655.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 14:00:00 | 690.35 | 2026-01-07 09:15:00 | 682.00 | STOP_HIT | 0.50 | 1.21% |
