# CG Power and Industrial Solutions Ltd. (CGPOWER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 875.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 75 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 5 |
| TARGET_HIT | 14 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 38
- **Target hits / Stop hits / Partials:** 14 / 39 / 5
- **Avg / median % per leg:** 1.70% / -1.18%
- **Sum % (uncompounded):** 98.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 12 | 27.9% | 12 | 31 | 0 | 1.65% | 70.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 43 | 12 | 27.9% | 12 | 31 | 0 | 1.65% | 70.8% |
| SELL (all) | 15 | 8 | 53.3% | 2 | 8 | 5 | 1.85% | 27.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 8 | 53.3% | 2 | 8 | 5 | 1.85% | 27.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 20 | 34.5% | 14 | 39 | 5 | 1.70% | 98.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 09:15:00 | 707.70 | 734.73 | 734.74 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 13:15:00 | 757.00 | 734.52 | 734.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 765.75 | 735.40 | 734.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 736.05 | 736.69 | 735.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 11:15:00 | 736.05 | 736.69 | 735.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 736.05 | 736.69 | 735.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 734.60 | 736.69 | 735.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 738.30 | 736.70 | 735.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 736.05 | 736.70 | 735.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 730.60 | 736.70 | 735.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 730.60 | 736.70 | 735.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 732.00 | 736.65 | 735.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 734.50 | 736.65 | 735.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-09 11:15:00 | 807.95 | 746.35 | 740.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 705.95 | 743.68 | 743.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 695.50 | 743.20 | 743.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 611.40 | 610.83 | 645.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:00:00 | 611.40 | 610.83 | 645.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 619.70 | 606.62 | 634.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 11:00:00 | 613.75 | 606.69 | 634.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 12:00:00 | 611.95 | 606.75 | 634.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 583.06 | 606.42 | 633.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 606.60 | 606.10 | 632.65 | SL hit (close>ema200) qty=0.50 sl=606.10 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 667.75 | 625.75 | 625.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 678.30 | 627.48 | 626.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 15:15:00 | 673.55 | 674.09 | 658.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 09:15:00 | 665.90 | 674.09 | 658.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 667.55 | 674.03 | 658.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 669.65 | 674.03 | 658.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 669.25 | 673.96 | 658.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:00:00 | 671.65 | 673.87 | 658.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 673.85 | 673.80 | 658.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 667.65 | 677.19 | 666.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:45:00 | 666.75 | 677.19 | 666.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 667.00 | 677.09 | 666.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:15:00 | 664.25 | 677.09 | 666.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 668.75 | 677.00 | 666.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 665.90 | 677.00 | 666.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 665.80 | 676.89 | 666.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 665.80 | 676.89 | 666.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 665.60 | 676.78 | 666.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 670.75 | 676.78 | 666.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 672.00 | 676.73 | 666.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:45:00 | 673.05 | 676.71 | 666.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 675.15 | 676.35 | 666.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:00:00 | 673.50 | 676.27 | 666.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 674.00 | 676.24 | 666.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 670.30 | 676.38 | 667.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 670.30 | 676.38 | 667.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 666.50 | 676.28 | 667.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 666.50 | 676.28 | 667.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 670.65 | 676.23 | 667.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 666.10 | 676.23 | 667.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 667.35 | 675.53 | 667.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 667.80 | 675.53 | 667.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 668.15 | 675.46 | 667.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:15:00 | 666.50 | 675.46 | 667.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 666.50 | 675.37 | 667.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 669.10 | 675.37 | 667.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 671.90 | 675.33 | 667.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 674.25 | 675.33 | 667.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 674.95 | 675.24 | 667.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:30:00 | 672.55 | 675.99 | 669.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 673.00 | 675.91 | 669.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 668.05 | 675.83 | 669.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 668.05 | 675.83 | 669.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 667.00 | 675.74 | 669.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 671.00 | 675.74 | 669.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 670.15 | 676.73 | 670.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 670.15 | 676.73 | 670.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 666.00 | 676.62 | 670.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 666.00 | 676.62 | 670.43 | SL hit (close<static) qty=1.00 sl=666.20 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 689.50 | 728.25 | 728.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 14:15:00 | 688.60 | 727.48 | 728.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 613.35 | 607.14 | 639.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:30:00 | 613.25 | 607.14 | 639.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 653.00 | 606.59 | 637.32 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 713.40 | 655.22 | 655.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 719.20 | 657.55 | 656.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 10:15:00 | 684.20 | 692.81 | 678.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:30:00 | 683.80 | 692.81 | 678.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 680.45 | 694.14 | 681.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 680.45 | 694.14 | 681.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 681.00 | 694.01 | 681.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 670.55 | 694.01 | 681.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 650.35 | 693.58 | 681.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 650.35 | 693.58 | 681.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 661.80 | 690.77 | 680.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:30:00 | 661.30 | 690.77 | 680.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 675.90 | 689.47 | 680.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 674.75 | 689.47 | 680.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 673.05 | 689.30 | 680.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 671.10 | 689.30 | 680.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 675.05 | 686.29 | 679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 675.05 | 686.29 | 679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 684.45 | 686.27 | 679.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:45:00 | 677.20 | 686.27 | 679.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 678.55 | 686.18 | 679.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 678.30 | 686.18 | 679.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 679.50 | 686.12 | 679.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 661.05 | 686.12 | 679.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 660.95 | 685.87 | 679.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 660.95 | 685.87 | 679.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 681.90 | 685.10 | 679.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 674.20 | 685.10 | 679.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 681.35 | 685.06 | 679.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 691.00 | 684.79 | 679.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:00:00 | 689.90 | 684.79 | 679.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:30:00 | 690.50 | 684.87 | 679.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 09:15:00 | 758.89 | 698.62 | 687.78 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:45:00 | 617.10 | 2024-06-12 10:15:00 | 678.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 14:45:00 | 619.30 | 2024-06-14 13:15:00 | 681.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:00:00 | 625.50 | 2024-06-14 13:15:00 | 688.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:15:00 | 734.50 | 2024-12-09 11:15:00 | 807.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-23 11:00:00 | 733.25 | 2024-12-23 13:15:00 | 727.55 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-12-26 12:45:00 | 736.20 | 2024-12-31 12:15:00 | 727.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-12-31 09:30:00 | 733.15 | 2024-12-31 12:15:00 | 727.50 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-03-10 11:00:00 | 613.75 | 2025-03-11 09:15:00 | 583.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 11:00:00 | 613.75 | 2025-03-11 14:15:00 | 606.60 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2025-03-10 12:00:00 | 611.95 | 2025-03-19 09:15:00 | 647.00 | STOP_HIT | 1.00 | -5.73% |
| SELL | retest2 | 2025-04-01 11:30:00 | 613.55 | 2025-04-04 13:15:00 | 582.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 15:15:00 | 612.90 | 2025-04-04 14:15:00 | 582.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:30:00 | 613.55 | 2025-04-07 09:15:00 | 552.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 15:15:00 | 612.90 | 2025-04-07 09:15:00 | 551.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-30 14:00:00 | 615.20 | 2025-04-30 14:15:00 | 626.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-05-02 12:15:00 | 615.50 | 2025-05-02 13:15:00 | 623.45 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-05-06 14:30:00 | 601.80 | 2025-05-08 09:15:00 | 626.30 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-05-07 10:00:00 | 615.10 | 2025-05-08 09:15:00 | 626.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-05-08 13:15:00 | 625.65 | 2025-05-09 09:15:00 | 594.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:30:00 | 625.45 | 2025-05-09 09:15:00 | 594.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:15:00 | 625.65 | 2025-05-12 09:15:00 | 637.45 | STOP_HIT | 0.50 | -1.89% |
| SELL | retest2 | 2025-05-08 14:30:00 | 625.45 | 2025-05-12 09:15:00 | 637.45 | STOP_HIT | 0.50 | -1.92% |
| BUY | retest2 | 2025-06-13 10:15:00 | 669.65 | 2025-07-24 11:15:00 | 666.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-13 11:15:00 | 669.25 | 2025-07-24 11:15:00 | 666.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-06-13 13:00:00 | 671.65 | 2025-07-24 11:15:00 | 666.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-16 10:15:00 | 673.85 | 2025-07-24 11:15:00 | 666.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-03 10:45:00 | 673.05 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-04 09:15:00 | 675.15 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-04 13:00:00 | 673.50 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-07 09:45:00 | 674.00 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-14 10:15:00 | 674.25 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-14 13:30:00 | 674.95 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-18 10:30:00 | 672.55 | 2025-07-25 13:15:00 | 664.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-07-18 13:30:00 | 673.00 | 2025-08-11 09:15:00 | 657.90 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-07-24 14:30:00 | 685.70 | 2025-08-14 11:15:00 | 668.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-07-25 10:30:00 | 670.80 | 2025-08-14 11:15:00 | 668.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-07-25 11:15:00 | 670.50 | 2025-08-14 11:15:00 | 668.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-08-05 09:15:00 | 670.45 | 2025-08-28 09:15:00 | 662.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-13 14:30:00 | 672.20 | 2025-08-28 09:15:00 | 662.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-14 09:15:00 | 675.05 | 2025-08-28 09:15:00 | 662.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-14 10:00:00 | 671.80 | 2025-08-28 09:15:00 | 662.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-08-18 09:15:00 | 676.05 | 2025-09-02 10:15:00 | 736.62 | TARGET_HIT | 1.00 | 8.96% |
| BUY | retest2 | 2025-08-18 14:00:00 | 673.65 | 2025-09-02 10:15:00 | 736.18 | TARGET_HIT | 1.00 | 9.28% |
| BUY | retest2 | 2025-08-19 10:45:00 | 674.00 | 2025-09-02 10:15:00 | 738.82 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2025-08-19 14:30:00 | 674.90 | 2025-09-02 10:15:00 | 741.24 | TARGET_HIT | 1.00 | 9.83% |
| BUY | retest2 | 2025-08-29 10:00:00 | 675.75 | 2025-09-03 09:15:00 | 743.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-29 15:00:00 | 753.35 | 2025-11-19 09:15:00 | 728.90 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-10-30 10:45:00 | 738.00 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-10-30 14:00:00 | 737.10 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-10-31 09:15:00 | 736.30 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-31 14:30:00 | 739.45 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-11-03 10:45:00 | 737.10 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-11-06 10:45:00 | 736.65 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-10 13:45:00 | 736.95 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-11-12 14:45:00 | 735.90 | 2025-11-19 10:15:00 | 722.65 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-04-07 13:30:00 | 691.00 | 2026-04-16 09:15:00 | 758.89 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2026-04-07 14:00:00 | 689.90 | 2026-04-16 09:15:00 | 759.55 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2026-04-07 14:30:00 | 690.50 | 2026-04-16 10:15:00 | 760.10 | TARGET_HIT | 1.00 | 10.08% |
