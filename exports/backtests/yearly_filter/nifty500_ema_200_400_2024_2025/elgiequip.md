# Elgi Equipments Ltd. (ELGIEQUIP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 561.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 1 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 47 |
| PARTIAL | 23 |
| TARGET_HIT | 13 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 28
- **Target hits / Stop hits / Partials:** 13 / 38 / 23
- **Avg / median % per leg:** 2.68% / 3.12%
- **Sum % (uncompounded):** 198.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.24% | -22.4% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.15% | -8.6% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.30% | -13.8% |
| SELL (all) | 64 | 46 | 71.9% | 13 | 28 | 23 | 3.45% | 220.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 64 | 46 | 71.9% | 13 | 28 | 23 | 3.45% | 220.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.15% | -8.6% |
| retest2 (combined) | 70 | 46 | 65.7% | 13 | 34 | 23 | 2.95% | 206.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 13:15:00 | 593.00 | 623.71 | 623.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 14:15:00 | 589.05 | 623.37 | 623.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 10:15:00 | 622.05 | 618.36 | 620.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 10:15:00 | 622.05 | 618.36 | 620.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 622.05 | 618.36 | 620.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 622.05 | 618.36 | 620.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 640.35 | 618.58 | 620.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 12:00:00 | 640.35 | 618.58 | 620.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 645.85 | 618.85 | 621.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:00:00 | 645.85 | 618.85 | 621.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 13:15:00 | 688.95 | 623.21 | 623.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 10:15:00 | 693.05 | 625.69 | 624.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 697.55 | 698.67 | 673.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:45:00 | 695.10 | 698.67 | 673.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 665.85 | 697.66 | 675.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 665.85 | 697.66 | 675.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 664.25 | 697.33 | 675.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 664.25 | 697.33 | 675.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 678.60 | 695.00 | 675.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:30:00 | 677.50 | 695.00 | 675.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 681.05 | 694.70 | 675.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 677.95 | 694.70 | 675.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 701.55 | 694.64 | 675.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 691.90 | 694.64 | 675.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 676.00 | 694.41 | 675.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 676.00 | 694.41 | 675.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 670.00 | 694.17 | 675.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 669.25 | 694.17 | 675.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 670.40 | 693.94 | 675.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 670.35 | 693.94 | 675.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 680.75 | 693.07 | 675.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:45:00 | 678.30 | 693.07 | 675.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 687.10 | 697.68 | 681.80 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 598.10 | 670.15 | 670.41 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 721.35 | 668.02 | 667.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 730.40 | 687.32 | 679.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 698.60 | 705.13 | 691.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:45:00 | 707.15 | 705.12 | 691.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:45:00 | 707.50 | 705.17 | 691.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 13:15:00 | 706.55 | 705.18 | 691.94 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 14:00:00 | 707.60 | 705.20 | 692.01 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 692.00 | 704.95 | 692.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 692.00 | 704.95 | 692.22 | SL hit (close<ema400) qty=1.00 sl=692.22 alert=retest1 |

### Cycle 5 — SELL (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 14:15:00 | 670.20 | 684.60 | 684.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 659.95 | 682.13 | 683.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 656.40 | 650.23 | 664.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 15:00:00 | 656.40 | 650.23 | 664.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 665.75 | 650.39 | 664.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 643.85 | 650.63 | 664.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:45:00 | 653.80 | 650.05 | 662.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 656.40 | 650.20 | 662.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:15:00 | 623.58 | 649.01 | 661.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 13:15:00 | 621.11 | 648.51 | 661.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 11:15:00 | 611.66 | 647.01 | 659.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 09:15:00 | 588.42 | 644.96 | 658.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 535.00 | 480.58 | 480.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 536.35 | 481.13 | 480.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 508.65 | 510.42 | 499.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 508.65 | 510.42 | 499.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 543.10 | 552.93 | 537.15 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 495.65 | 527.03 | 527.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 489.10 | 526.36 | 526.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 496.35 | 492.38 | 505.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 15:00:00 | 496.35 | 492.38 | 505.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 511.85 | 492.64 | 505.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 511.85 | 492.64 | 505.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 506.60 | 492.77 | 505.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:30:00 | 505.05 | 492.90 | 505.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:30:00 | 503.55 | 493.01 | 505.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 503.95 | 493.37 | 505.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 505.10 | 494.05 | 505.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 499.40 | 494.11 | 505.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:00:00 | 496.80 | 494.26 | 505.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 496.60 | 494.33 | 505.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 497.55 | 494.54 | 504.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 498.30 | 494.64 | 504.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 504.25 | 494.73 | 504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 504.25 | 494.73 | 504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 501.30 | 494.80 | 504.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 498.10 | 495.02 | 504.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 479.80 | 494.86 | 504.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 479.85 | 494.86 | 504.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 478.37 | 494.70 | 504.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 478.75 | 494.70 | 504.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 473.38 | 493.76 | 503.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 473.19 | 493.76 | 503.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 491.95 | 501.35 | SL hit (close>ema200) qty=0.50 sl=491.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 508.10 | 494.41 | 494.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 509.95 | 494.99 | 494.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 496.70 | 496.76 | 495.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:45:00 | 496.40 | 496.76 | 495.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 493.20 | 496.72 | 495.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 493.20 | 496.72 | 495.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 491.50 | 496.67 | 495.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 497.00 | 496.67 | 495.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 493.70 | 496.80 | 495.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 487.50 | 496.70 | 495.67 | SL hit (close<static) qty=1.00 sl=489.85 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 484.35 | 494.74 | 494.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 483.75 | 494.63 | 494.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 483.25 | 480.27 | 485.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 09:45:00 | 483.10 | 480.27 | 485.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 481.10 | 480.43 | 485.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 477.75 | 480.36 | 485.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 476.90 | 480.36 | 485.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 15:00:00 | 477.75 | 480.33 | 485.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 453.86 | 477.45 | 483.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 453.05 | 477.45 | 483.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 453.86 | 477.45 | 483.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-16 15:15:00 | 429.98 | 466.64 | 476.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 515.00 | 472.01 | 471.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 516.95 | 475.95 | 473.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 501.30 | 504.19 | 490.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 11:00:00 | 501.30 | 504.19 | 490.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 500.75 | 505.39 | 492.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 516.35 | 505.37 | 493.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 485.00 | 506.23 | 494.55 | SL hit (close<static) qty=1.00 sl=492.55 alert=retest2 |

### Cycle 11 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 476.85 | 487.29 | 487.33 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 15:15:00 | 500.50 | 487.45 | 487.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 502.35 | 487.60 | 487.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-24 10:45:00 | 707.15 | 2024-09-25 11:15:00 | 692.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest1 | 2024-09-24 11:45:00 | 707.50 | 2024-09-25 11:15:00 | 692.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest1 | 2024-09-24 13:15:00 | 706.55 | 2024-09-25 11:15:00 | 692.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest1 | 2024-09-24 14:00:00 | 707.60 | 2024-09-25 11:15:00 | 692.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-09-27 09:15:00 | 696.00 | 2024-09-27 09:15:00 | 683.95 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-10-03 10:15:00 | 694.65 | 2024-10-03 11:15:00 | 688.15 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-11-04 09:15:00 | 643.85 | 2024-11-11 11:15:00 | 623.58 | PARTIAL | 0.50 | 3.15% |
| SELL | retest2 | 2024-11-07 09:45:00 | 653.80 | 2024-11-11 13:15:00 | 621.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 656.40 | 2024-11-12 11:15:00 | 611.66 | PARTIAL | 0.50 | 6.82% |
| SELL | retest2 | 2024-11-04 09:15:00 | 643.85 | 2024-11-13 09:15:00 | 588.42 | TARGET_HIT | 0.50 | 8.61% |
| SELL | retest2 | 2024-11-07 09:45:00 | 653.80 | 2024-11-13 09:15:00 | 590.76 | TARGET_HIT | 0.50 | 9.64% |
| SELL | retest2 | 2024-11-07 13:15:00 | 656.40 | 2024-11-13 11:15:00 | 579.47 | TARGET_HIT | 0.50 | 11.72% |
| SELL | retest2 | 2024-11-26 09:30:00 | 653.00 | 2024-12-03 12:15:00 | 660.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-27 09:15:00 | 633.40 | 2024-12-03 12:15:00 | 660.00 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2024-11-28 09:45:00 | 643.95 | 2024-12-03 12:15:00 | 660.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-12-02 10:00:00 | 644.00 | 2024-12-10 15:15:00 | 620.35 | PARTIAL | 0.50 | 3.67% |
| SELL | retest2 | 2024-12-06 09:45:00 | 644.80 | 2024-12-12 09:15:00 | 612.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 10:00:00 | 644.00 | 2024-12-23 15:15:00 | 587.70 | TARGET_HIT | 0.50 | 8.74% |
| SELL | retest2 | 2024-12-06 09:45:00 | 644.80 | 2024-12-27 11:15:00 | 580.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-18 09:15:00 | 492.05 | 2025-04-01 15:15:00 | 474.86 | PARTIAL | 0.50 | 3.49% |
| SELL | retest2 | 2025-03-20 09:45:00 | 499.85 | 2025-04-01 15:15:00 | 474.52 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-03-20 10:30:00 | 494.65 | 2025-04-02 09:15:00 | 474.00 | PARTIAL | 0.50 | 4.17% |
| SELL | retest2 | 2025-03-24 09:30:00 | 499.50 | 2025-04-04 09:15:00 | 467.45 | PARTIAL | 0.50 | 6.42% |
| SELL | retest2 | 2025-03-25 10:00:00 | 498.95 | 2025-04-04 09:15:00 | 469.92 | PARTIAL | 0.50 | 5.82% |
| SELL | retest2 | 2025-03-18 09:15:00 | 492.05 | 2025-04-04 12:15:00 | 449.87 | TARGET_HIT | 0.50 | 8.57% |
| SELL | retest2 | 2025-03-20 09:45:00 | 499.85 | 2025-04-04 12:15:00 | 449.55 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2025-03-20 10:30:00 | 494.65 | 2025-04-04 12:15:00 | 449.06 | TARGET_HIT | 0.50 | 9.22% |
| SELL | retest2 | 2025-03-24 09:30:00 | 499.50 | 2025-04-04 13:15:00 | 442.85 | TARGET_HIT | 0.50 | 11.34% |
| SELL | retest2 | 2025-03-25 10:00:00 | 498.95 | 2025-04-04 13:15:00 | 445.19 | TARGET_HIT | 0.50 | 10.78% |
| SELL | retest2 | 2025-05-28 13:15:00 | 503.55 | 2025-05-30 09:15:00 | 524.65 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-05-28 15:00:00 | 502.90 | 2025-05-30 09:15:00 | 524.65 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-05-29 09:15:00 | 501.70 | 2025-05-30 09:15:00 | 524.65 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-09-18 11:30:00 | 505.05 | 2025-09-26 10:15:00 | 479.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 12:30:00 | 503.55 | 2025-09-26 10:15:00 | 479.85 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-19 09:15:00 | 503.95 | 2025-09-26 11:15:00 | 478.37 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-09-22 09:30:00 | 505.10 | 2025-09-26 11:15:00 | 478.75 | PARTIAL | 0.50 | 5.22% |
| SELL | retest2 | 2025-09-22 14:00:00 | 496.80 | 2025-09-29 14:15:00 | 473.38 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-23 09:15:00 | 496.60 | 2025-09-29 14:15:00 | 473.19 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-18 11:30:00 | 505.05 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2025-09-18 12:30:00 | 503.55 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-09-19 09:15:00 | 503.95 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-09-22 09:30:00 | 505.10 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-09-22 14:00:00 | 496.80 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2025-09-23 09:15:00 | 496.60 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 0.50 | 0.93% |
| SELL | retest2 | 2025-09-24 09:15:00 | 497.55 | 2025-10-13 10:15:00 | 471.96 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-09-24 10:30:00 | 498.30 | 2025-10-13 10:15:00 | 471.77 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-09-25 09:30:00 | 498.10 | 2025-10-13 10:15:00 | 472.67 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-10-07 11:15:00 | 497.50 | 2025-10-13 10:15:00 | 472.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 497.55 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2025-09-24 10:30:00 | 498.30 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2025-09-25 09:30:00 | 498.10 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2025-10-07 11:15:00 | 497.50 | 2025-10-24 13:15:00 | 482.75 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-10-29 11:30:00 | 499.85 | 2025-10-29 14:15:00 | 505.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-30 10:00:00 | 499.80 | 2025-10-31 09:15:00 | 507.45 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-04 11:15:00 | 488.55 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-11-11 11:00:00 | 489.40 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-12 09:45:00 | 489.35 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-11-12 10:45:00 | 488.65 | 2025-11-13 09:15:00 | 499.25 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-11-21 15:00:00 | 492.25 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-11-24 09:30:00 | 492.00 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-11-24 10:15:00 | 491.95 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-24 10:45:00 | 493.05 | 2025-11-24 14:15:00 | 498.40 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-11-25 09:30:00 | 489.00 | 2025-11-26 09:15:00 | 504.05 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-25 10:45:00 | 477.15 | 2025-11-26 09:15:00 | 504.05 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest2 | 2025-12-05 09:15:00 | 497.00 | 2025-12-08 10:15:00 | 487.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-12-08 10:15:00 | 493.70 | 2025-12-08 10:15:00 | 487.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-09 15:00:00 | 494.20 | 2025-12-10 10:15:00 | 484.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-01-05 13:30:00 | 477.75 | 2026-01-09 09:15:00 | 453.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 476.90 | 2026-01-09 09:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 477.75 | 2026-01-09 09:15:00 | 453.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:30:00 | 477.75 | 2026-01-16 15:15:00 | 429.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 476.90 | 2026-01-16 15:15:00 | 429.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 477.75 | 2026-01-16 15:15:00 | 429.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-10 09:15:00 | 516.35 | 2026-03-12 09:15:00 | 485.00 | STOP_HIT | 1.00 | -6.07% |
