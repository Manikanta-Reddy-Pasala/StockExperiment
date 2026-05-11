# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 670.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 41 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 34
- **Target hits / Stop hits / Partials:** 5 / 36 / 4
- **Avg / median % per leg:** -0.41% / -1.61%
- **Sum % (uncompounded):** -18.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 5 | 21.7% | 5 | 18 | 0 | -0.40% | -9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 5 | 21.7% | 5 | 18 | 0 | -0.40% | -9.2% |
| SELL (all) | 22 | 6 | 27.3% | 0 | 18 | 4 | -0.42% | -9.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 0 | 18 | 4 | -0.42% | -9.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 11 | 24.4% | 5 | 36 | 4 | -0.41% | -18.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 545.15 | 527.12 | 527.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 566.40 | 529.51 | 528.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 882.95 | 896.86 | 841.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 10:00:00 | 882.95 | 896.86 | 841.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 861.45 | 894.48 | 842.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 874.60 | 892.44 | 842.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:15:00 | 879.90 | 882.37 | 844.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 14:15:00 | 830.85 | 885.41 | 852.77 | SL hit (close<static) qty=1.00 sl=833.85 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 13:15:00 | 729.45 | 843.48 | 843.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 724.15 | 786.07 | 804.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 732.20 | 712.73 | 747.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 732.20 | 712.73 | 747.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 732.20 | 712.73 | 747.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 745.45 | 712.73 | 747.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 685.00 | 657.51 | 684.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:45:00 | 683.45 | 657.51 | 684.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 683.35 | 657.77 | 684.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:15:00 | 677.60 | 657.77 | 684.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 11:30:00 | 680.50 | 659.16 | 684.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 685.40 | 660.61 | 684.32 | SL hit (close>static) qty=1.00 sl=685.35 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 14:15:00 | 729.85 | 689.79 | 689.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 15:15:00 | 731.90 | 690.21 | 689.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 699.30 | 699.96 | 695.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 699.30 | 699.96 | 695.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 699.30 | 699.96 | 695.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:00:00 | 706.10 | 698.02 | 694.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 688.75 | 697.92 | 695.00 | SL hit (close<static) qty=1.00 sl=692.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 733.20 | 756.52 | 756.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 724.50 | 755.22 | 755.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 744.00 | 749.00 | 752.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 12:15:00 | 706.80 | 734.91 | 742.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 727.50 | 727.07 | 736.17 | SL hit (close>ema200) qty=0.50 sl=727.07 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 756.55 | 731.53 | 731.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 13:15:00 | 761.30 | 731.82 | 731.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:30:00 | 734.30 | 736.04 | 733.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 737.10 | 736.05 | 733.98 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 712.70 | 732.07 | 732.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 707.40 | 731.44 | 731.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 632.00 | 630.53 | 653.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 632.00 | 630.53 | 653.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 643.20 | 626.01 | 644.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 643.20 | 626.01 | 644.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 643.35 | 627.10 | 644.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 642.85 | 627.10 | 644.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 645.55 | 627.28 | 644.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:30:00 | 647.65 | 627.28 | 644.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 647.75 | 627.49 | 644.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:30:00 | 647.70 | 627.49 | 644.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 643.85 | 628.10 | 644.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:15:00 | 642.70 | 628.10 | 644.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 643.20 | 628.25 | 644.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 640.55 | 628.69 | 644.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 641.55 | 629.52 | 643.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 639.85 | 630.25 | 643.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 647.20 | 631.01 | 643.50 | SL hit (close>static) qty=1.00 sl=644.45 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 675.65 | 652.29 | 652.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 677.25 | 652.98 | 652.58 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 522.10 | 2024-05-16 09:15:00 | 547.35 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2024-05-13 13:15:00 | 522.75 | 2024-05-16 09:15:00 | 547.35 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2024-05-14 11:30:00 | 522.25 | 2024-05-16 09:15:00 | 547.35 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-10-09 09:15:00 | 874.60 | 2024-10-22 14:15:00 | 830.85 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2024-10-15 14:15:00 | 879.90 | 2024-10-22 14:15:00 | 830.85 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest2 | 2024-10-25 09:15:00 | 903.75 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -8.19% |
| BUY | retest2 | 2024-10-25 11:30:00 | 873.75 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest2 | 2024-10-29 15:15:00 | 846.00 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-30 09:30:00 | 845.45 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-30 12:30:00 | 845.65 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-30 14:30:00 | 845.40 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-11-04 13:15:00 | 866.00 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-11-11 11:00:00 | 866.90 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2024-11-12 09:15:00 | 869.95 | 2024-11-13 09:15:00 | 829.70 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2025-03-17 13:15:00 | 677.60 | 2025-03-19 11:15:00 | 685.40 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-03-18 11:30:00 | 680.50 | 2025-03-19 11:15:00 | 685.40 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-03-25 10:30:00 | 681.00 | 2025-04-01 09:15:00 | 691.95 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-04-01 11:30:00 | 681.30 | 2025-04-01 12:15:00 | 695.50 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-04 12:15:00 | 676.25 | 2025-04-07 09:15:00 | 642.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 13:30:00 | 671.55 | 2025-04-07 09:15:00 | 637.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 15:15:00 | 673.00 | 2025-04-07 09:15:00 | 639.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 12:15:00 | 676.25 | 2025-04-07 14:15:00 | 675.20 | STOP_HIT | 0.50 | 0.16% |
| SELL | retest2 | 2025-04-04 13:30:00 | 671.55 | 2025-04-07 14:15:00 | 675.20 | STOP_HIT | 0.50 | -0.54% |
| SELL | retest2 | 2025-04-04 15:15:00 | 673.00 | 2025-04-07 14:15:00 | 675.20 | STOP_HIT | 0.50 | -0.33% |
| SELL | retest2 | 2025-04-09 09:15:00 | 671.15 | 2025-04-11 09:15:00 | 695.00 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-05-08 10:00:00 | 706.10 | 2025-05-08 13:15:00 | 688.75 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-05-12 09:15:00 | 709.80 | 2025-05-13 14:15:00 | 690.10 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-05-12 14:30:00 | 705.90 | 2025-05-13 14:15:00 | 690.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-13 09:45:00 | 705.90 | 2025-05-13 14:15:00 | 690.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-16 14:45:00 | 710.60 | 2025-06-10 09:15:00 | 781.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 10:30:00 | 712.30 | 2025-06-10 09:15:00 | 783.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 09:30:00 | 711.50 | 2025-07-23 14:15:00 | 782.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 10:00:00 | 711.75 | 2025-07-23 14:15:00 | 782.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-16 09:15:00 | 734.60 | 2025-07-24 09:15:00 | 808.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 14:00:00 | 730.50 | 2025-08-11 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-07 09:30:00 | 730.15 | 2025-08-11 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-08-07 10:00:00 | 728.75 | 2025-08-11 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-15 09:15:00 | 744.00 | 2025-11-10 12:15:00 | 706.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 09:15:00 | 744.00 | 2025-11-20 09:15:00 | 727.50 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2025-12-16 12:00:00 | 746.10 | 2025-12-29 12:15:00 | 756.55 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-12-16 12:45:00 | 746.95 | 2025-12-29 12:15:00 | 756.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-12-16 13:30:00 | 745.85 | 2025-12-29 12:15:00 | 756.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-10 14:15:00 | 640.55 | 2026-04-17 09:15:00 | 647.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-04-15 10:15:00 | 641.55 | 2026-04-17 09:15:00 | 647.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-16 09:45:00 | 639.85 | 2026-04-17 09:15:00 | 647.20 | STOP_HIT | 1.00 | -1.15% |
