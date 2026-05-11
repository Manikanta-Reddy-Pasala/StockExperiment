# Asahi India Glass Ltd. (ASAHIINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 836.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 4 |
| TARGET_HIT | 7 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 7
- **Target hits / Stop hits / Partials:** 7 / 7 / 4
- **Avg / median % per leg:** 3.95% / 5.00%
- **Sum % (uncompounded):** 71.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 3 | 33.3% | 3 | 6 | 0 | 1.72% | 15.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 3 | 33.3% | 3 | 6 | 0 | 1.72% | 15.5% |
| SELL (all) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.18% | 55.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.18% | 55.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 11 | 61.1% | 7 | 7 | 4 | 3.95% | 71.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 648.05 | 697.71 | 697.95 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 757.10 | 695.13 | 695.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 758.80 | 695.76 | 695.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 734.40 | 737.94 | 723.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 734.40 | 737.94 | 723.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 726.00 | 737.78 | 723.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 716.55 | 737.78 | 723.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 726.90 | 737.67 | 723.71 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 651.10 | 713.14 | 713.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 647.25 | 704.29 | 708.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 675.75 | 671.99 | 688.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 675.75 | 671.99 | 688.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 662.10 | 659.26 | 675.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 662.10 | 659.26 | 675.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 670.20 | 659.58 | 674.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:00:00 | 670.20 | 659.58 | 674.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 676.80 | 659.75 | 674.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:30:00 | 677.55 | 659.75 | 674.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 683.50 | 659.99 | 674.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:00:00 | 683.50 | 659.99 | 674.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 683.00 | 660.22 | 675.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:30:00 | 685.05 | 660.44 | 675.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 689.90 | 660.73 | 675.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:00:00 | 689.90 | 660.73 | 675.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 675.55 | 664.75 | 676.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 675.55 | 664.75 | 676.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 687.95 | 665.05 | 676.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 692.30 | 665.05 | 676.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 682.40 | 665.23 | 676.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 682.40 | 665.23 | 676.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 677.70 | 665.44 | 676.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 677.70 | 665.44 | 676.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 684.65 | 665.63 | 676.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:00:00 | 684.65 | 665.63 | 676.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 677.25 | 666.81 | 676.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:30:00 | 674.95 | 666.92 | 676.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 674.65 | 667.04 | 676.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 641.20 | 666.29 | 676.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 640.92 | 666.29 | 676.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 607.46 | 663.11 | 674.05 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 703.40 | 649.19 | 649.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-25 11:15:00 | 709.00 | 650.25 | 649.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 710.00 | 714.50 | 691.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 13:00:00 | 710.00 | 714.50 | 691.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 722.45 | 737.13 | 718.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 719.65 | 737.13 | 718.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 710.75 | 736.68 | 717.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 710.75 | 736.68 | 717.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 708.90 | 736.40 | 717.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 704.55 | 736.40 | 717.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 711.65 | 728.93 | 715.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:30:00 | 721.80 | 728.86 | 716.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 15:00:00 | 717.50 | 728.55 | 716.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-01 12:15:00 | 793.98 | 731.98 | 718.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 935.30 | 962.19 | 962.32 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 981.90 | 962.53 | 962.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 993.90 | 963.07 | 962.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 961.90 | 970.26 | 966.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 961.90 | 970.26 | 966.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 964.00 | 970.20 | 966.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 959.90 | 970.20 | 966.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 964.50 | 969.96 | 966.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 965.00 | 969.96 | 966.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 967.90 | 969.94 | 966.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:00:00 | 973.60 | 969.98 | 966.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 975.90 | 970.05 | 966.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 974.00 | 970.05 | 966.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 950.90 | 969.90 | 966.73 | SL hit (close<static) qty=1.00 sl=962.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 934.50 | 964.87 | 965.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 931.20 | 964.03 | 964.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 914.00 | 894.04 | 921.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:00:00 | 914.00 | 894.04 | 921.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 946.30 | 894.56 | 921.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 946.30 | 894.56 | 921.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 885.00 | 894.46 | 921.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:15:00 | 876.60 | 894.56 | 921.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:45:00 | 866.95 | 894.26 | 920.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 15:15:00 | 832.77 | 891.07 | 918.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 09:15:00 | 823.60 | 890.50 | 917.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 788.94 | 885.30 | 914.30 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 11:15:00 | 586.00 | 2024-06-04 10:15:00 | 569.00 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-06-04 10:00:00 | 585.50 | 2024-06-04 10:15:00 | 569.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-06-06 09:15:00 | 585.40 | 2024-06-18 10:15:00 | 643.94 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-25 14:30:00 | 674.95 | 2025-02-28 09:15:00 | 641.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 674.65 | 2025-02-28 09:15:00 | 640.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:30:00 | 674.95 | 2025-03-03 09:15:00 | 607.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 674.65 | 2025-03-03 09:15:00 | 607.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-15 09:30:00 | 675.00 | 2025-04-22 09:15:00 | 704.60 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-06-26 11:30:00 | 721.80 | 2025-07-01 12:15:00 | 793.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 15:00:00 | 717.50 | 2025-07-01 12:15:00 | 789.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-05 13:00:00 | 973.60 | 2026-02-06 09:15:00 | 950.90 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-05 13:30:00 | 975.90 | 2026-02-06 09:15:00 | 950.90 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-02-05 15:15:00 | 974.00 | 2026-02-06 09:15:00 | 950.90 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-02-06 13:00:00 | 974.80 | 2026-02-13 09:15:00 | 959.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-03-18 14:15:00 | 876.60 | 2026-03-19 15:15:00 | 832.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:45:00 | 866.95 | 2026-03-20 09:15:00 | 823.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:15:00 | 876.60 | 2026-03-23 09:15:00 | 788.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-18 14:45:00 | 866.95 | 2026-03-24 10:15:00 | 780.26 | TARGET_HIT | 0.50 | 10.00% |
