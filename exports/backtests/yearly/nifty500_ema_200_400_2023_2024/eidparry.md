# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 834.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 9 |
| TARGET_HIT | 13 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 25 / 18
- **Target hits / Stop hits / Partials:** 13 / 21 / 9
- **Avg / median % per leg:** 2.22% / 5.00%
- **Sum % (uncompounded):** 95.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 7 | 7 | 0 | 2.12% | 29.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 7 | 50.0% | 7 | 7 | 0 | 2.12% | 29.7% |
| SELL (all) | 29 | 18 | 62.1% | 6 | 14 | 9 | 2.27% | 65.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 18 | 62.1% | 6 | 14 | 9 | 2.27% | 65.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 25 | 58.1% | 13 | 21 | 9 | 2.22% | 95.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 15:15:00 | 516.25 | 480.91 | 480.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 518.95 | 481.29 | 481.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 511.25 | 515.35 | 503.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 12:00:00 | 511.25 | 515.35 | 503.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 505.65 | 516.86 | 506.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:45:00 | 506.20 | 516.86 | 506.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 509.10 | 516.79 | 506.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:30:00 | 510.90 | 516.72 | 506.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 15:00:00 | 509.70 | 516.60 | 506.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 496.80 | 515.55 | 507.18 | SL hit (close<static) qty=1.00 sl=505.30 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 13:15:00 | 467.40 | 500.73 | 500.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 14:15:00 | 466.35 | 500.38 | 500.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 14:15:00 | 492.40 | 490.39 | 494.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 14:15:00 | 492.40 | 490.39 | 494.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 492.40 | 490.39 | 494.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 12:15:00 | 477.20 | 493.80 | 495.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 511.50 | 493.75 | 495.68 | SL hit (close>static) qty=1.00 sl=501.55 alert=retest2 |

### Cycle 3 — BUY (started 2023-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 14:15:00 | 537.40 | 497.81 | 497.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 545.10 | 507.97 | 503.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 563.00 | 563.38 | 546.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 10:00:00 | 563.00 | 563.38 | 546.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 592.75 | 595.21 | 572.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 11:30:00 | 596.50 | 595.23 | 572.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 09:15:00 | 595.85 | 593.62 | 573.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 09:45:00 | 598.60 | 593.53 | 574.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 11:00:00 | 596.75 | 593.56 | 574.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-21 09:15:00 | 656.15 | 597.77 | 578.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 13:15:00 | 551.25 | 583.82 | 583.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 540.00 | 583.01 | 583.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 10:15:00 | 581.50 | 579.98 | 581.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 10:15:00 | 581.50 | 579.98 | 581.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 581.50 | 579.98 | 581.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:00:00 | 581.50 | 579.98 | 581.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 586.10 | 580.04 | 581.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 12:00:00 | 586.10 | 580.04 | 581.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 12:15:00 | 585.00 | 580.09 | 581.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 12:45:00 | 584.80 | 580.09 | 581.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 585.00 | 580.19 | 581.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:15:00 | 583.80 | 580.19 | 581.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 583.65 | 580.23 | 581.87 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 15:15:00 | 620.00 | 583.35 | 583.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 09:15:00 | 629.15 | 583.81 | 583.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 606.40 | 608.42 | 599.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 11:30:00 | 605.05 | 608.42 | 599.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 608.90 | 609.55 | 600.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:30:00 | 607.20 | 609.55 | 600.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 601.45 | 609.42 | 600.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 12:00:00 | 601.45 | 609.42 | 600.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 607.05 | 609.40 | 600.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 611.25 | 609.29 | 600.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 11:30:00 | 611.80 | 609.29 | 600.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 15:15:00 | 611.00 | 609.39 | 601.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-29 11:15:00 | 672.38 | 623.83 | 612.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 796.45 | 857.23 | 857.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 785.15 | 856.51 | 857.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 11:15:00 | 843.30 | 840.44 | 847.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 11:45:00 | 842.60 | 840.44 | 847.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 841.15 | 840.51 | 847.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 835.10 | 840.51 | 847.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 15:00:00 | 839.80 | 840.25 | 847.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 858.45 | 840.40 | 847.30 | SL hit (close>static) qty=1.00 sl=849.90 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 859.65 | 778.54 | 778.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 868.40 | 788.00 | 783.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 948.90 | 949.89 | 905.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 942.75 | 949.89 | 905.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1097.00 | 1146.07 | 1079.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1140.60 | 1146.07 | 1079.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 14:45:00 | 1130.00 | 1152.51 | 1106.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 14:00:00 | 1135.40 | 1149.07 | 1107.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1146.70 | 1148.64 | 1107.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1106.50 | 1147.47 | 1108.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1106.50 | 1147.47 | 1108.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1103.90 | 1147.04 | 1108.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1103.90 | 1147.04 | 1108.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1103.00 | 1146.60 | 1108.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1124.50 | 1146.60 | 1108.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1097.50 | 1141.55 | 1109.43 | SL hit (close<static) qty=1.00 sl=1098.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 1018.90 | 1093.05 | 1093.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 1008.00 | 1054.83 | 1065.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1071.80 | 1053.09 | 1064.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 1071.80 | 1053.09 | 1064.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1055.00 | 1053.11 | 1063.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 1047.90 | 1053.11 | 1063.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:00:00 | 1048.70 | 1052.72 | 1063.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1050.90 | 1050.46 | 1061.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 995.50 | 1035.70 | 1047.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 996.26 | 1035.70 | 1047.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 998.36 | 1035.70 | 1047.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1045.30 | 1033.84 | 1046.10 | SL hit (close>ema200) qty=0.50 sl=1033.84 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-13 12:30:00 | 510.90 | 2023-10-18 14:15:00 | 496.80 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2023-10-13 15:00:00 | 509.70 | 2023-10-18 14:15:00 | 496.80 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2023-11-22 12:15:00 | 477.20 | 2023-11-23 09:15:00 | 511.50 | STOP_HIT | 1.00 | -7.19% |
| BUY | retest2 | 2024-02-12 11:30:00 | 596.50 | 2024-02-21 09:15:00 | 656.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-15 09:15:00 | 595.85 | 2024-02-21 09:15:00 | 655.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-16 09:45:00 | 598.60 | 2024-02-21 09:15:00 | 658.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-16 11:00:00 | 596.75 | 2024-02-21 09:15:00 | 656.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 09:15:00 | 611.25 | 2024-05-29 11:15:00 | 672.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 11:30:00 | 611.80 | 2024-05-29 11:15:00 | 672.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 15:15:00 | 611.00 | 2024-05-29 11:15:00 | 672.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 835.10 | 2025-02-06 09:15:00 | 858.45 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-02-05 15:00:00 | 839.80 | 2025-02-06 09:15:00 | 858.45 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-02-07 12:00:00 | 840.55 | 2025-02-07 14:15:00 | 850.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-02-10 09:15:00 | 837.70 | 2025-02-11 11:15:00 | 795.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 837.70 | 2025-02-14 10:15:00 | 753.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 730.40 | 2025-04-11 09:15:00 | 777.25 | STOP_HIT | 1.00 | -6.41% |
| SELL | retest2 | 2025-04-08 13:30:00 | 764.50 | 2025-04-11 09:15:00 | 777.25 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-04-08 15:00:00 | 761.10 | 2025-04-11 09:15:00 | 777.25 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-08-11 09:15:00 | 1140.60 | 2025-09-08 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-08-28 14:45:00 | 1130.00 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -7.47% |
| BUY | retest2 | 2025-09-01 14:00:00 | 1135.40 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -7.91% |
| BUY | retest2 | 2025-09-02 09:15:00 | 1146.70 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -8.82% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1124.50 | 2025-09-26 09:15:00 | 1045.60 | STOP_HIT | 1.00 | -7.02% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1047.90 | 2025-12-09 09:15:00 | 995.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 11:00:00 | 1048.70 | 2025-12-09 09:15:00 | 996.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 15:00:00 | 1050.90 | 2025-12-09 09:15:00 | 998.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1047.90 | 2025-12-10 10:15:00 | 1045.30 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest2 | 2025-11-13 11:00:00 | 1048.70 | 2025-12-10 10:15:00 | 1045.30 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2025-11-17 15:00:00 | 1050.90 | 2025-12-10 10:15:00 | 1045.30 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1043.30 | 2026-01-07 13:15:00 | 991.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1030.80 | 2026-01-08 10:15:00 | 979.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 10:15:00 | 1035.30 | 2026-01-08 10:15:00 | 983.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1034.40 | 2026-01-08 10:15:00 | 982.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 13:00:00 | 1035.00 | 2026-01-08 10:15:00 | 983.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1043.30 | 2026-01-09 13:15:00 | 938.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1030.80 | 2026-01-12 09:15:00 | 927.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 10:15:00 | 1035.30 | 2026-01-12 09:15:00 | 931.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1034.40 | 2026-01-12 09:15:00 | 930.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 13:00:00 | 1035.00 | 2026-01-12 09:15:00 | 931.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 12:00:00 | 845.00 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2026-04-16 15:00:00 | 850.80 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-04-20 09:30:00 | 848.40 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-04-20 15:15:00 | 850.00 | 2026-04-23 09:15:00 | 884.00 | STOP_HIT | 1.00 | -4.00% |
