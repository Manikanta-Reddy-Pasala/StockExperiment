# HINDALCO (HINDALCO)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1044.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 5 |
| TARGET_HIT | 14 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 26
- **Target hits / Stop hits / Partials:** 14 / 34 / 5
- **Avg / median % per leg:** 1.82% / 0.10%
- **Sum % (uncompounded):** 96.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 17 | 65.4% | 13 | 13 | 0 | 3.71% | 96.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 17 | 65.4% | 13 | 13 | 0 | 3.71% | 96.4% |
| SELL (all) | 27 | 10 | 37.0% | 1 | 21 | 5 | 0.00% | 0.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 10 | 37.0% | 1 | 21 | 5 | 0.00% | 0.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 27 | 50.9% | 14 | 34 | 5 | 1.82% | 96.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 11:15:00 | 410.20 | 425.64 | 425.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 406.50 | 423.81 | 424.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 09:15:00 | 422.85 | 417.15 | 420.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 09:15:00 | 422.85 | 417.15 | 420.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 422.85 | 417.15 | 420.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 10:00:00 | 422.85 | 417.15 | 420.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 423.05 | 417.20 | 420.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 10:30:00 | 423.40 | 417.20 | 420.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 419.70 | 417.39 | 420.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 416.15 | 417.42 | 420.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-06 09:45:00 | 418.10 | 416.34 | 419.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-06 10:15:00 | 417.25 | 416.34 | 419.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-07 14:15:00 | 422.55 | 416.38 | 419.43 | SL hit (close>static) qty=1.00 sl=421.40 alert=retest2 |

### Cycle 2 — BUY (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 11:15:00 | 428.05 | 420.67 | 420.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 436.55 | 421.63 | 421.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 09:15:00 | 441.30 | 447.71 | 438.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-16 10:00:00 | 441.30 | 447.71 | 438.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 439.45 | 446.97 | 439.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 13:00:00 | 439.45 | 446.97 | 439.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 440.20 | 446.86 | 439.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 440.20 | 446.86 | 439.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 439.20 | 446.79 | 439.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 442.35 | 446.79 | 439.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-04 09:15:00 | 486.59 | 452.62 | 444.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 11:15:00 | 505.45 | 545.46 | 545.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 502.10 | 540.08 | 542.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 10:15:00 | 533.40 | 532.93 | 538.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 11:00:00 | 533.40 | 532.93 | 538.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 539.70 | 533.06 | 538.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:30:00 | 540.05 | 533.06 | 538.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 537.75 | 533.10 | 538.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 10:45:00 | 535.60 | 533.22 | 538.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 12:00:00 | 535.80 | 533.25 | 538.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 14:15:00 | 535.50 | 533.33 | 538.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 15:00:00 | 533.20 | 533.33 | 538.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 12:15:00 | 508.82 | 532.40 | 537.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 12:15:00 | 509.01 | 532.40 | 537.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 12:15:00 | 508.72 | 532.40 | 537.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 14:15:00 | 506.54 | 531.89 | 537.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-15 14:15:00 | 532.65 | 530.60 | 535.99 | SL hit (close>ema200) qty=0.50 sl=530.60 alert=retest2 |

### Cycle 4 — BUY (started 2024-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 15:15:00 | 568.45 | 539.20 | 539.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 13:15:00 | 569.65 | 540.53 | 539.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 636.90 | 656.14 | 625.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 636.90 | 656.14 | 625.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 636.00 | 655.66 | 625.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 611.95 | 655.66 | 625.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 644.00 | 655.36 | 625.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 10:45:00 | 648.30 | 655.33 | 625.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 654.70 | 682.93 | 666.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:30:00 | 645.10 | 679.28 | 665.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:00:00 | 645.15 | 677.84 | 665.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 661.70 | 676.93 | 664.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:00:00 | 661.70 | 676.93 | 664.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 666.45 | 676.83 | 664.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 14:15:00 | 667.80 | 676.83 | 664.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 15:00:00 | 668.25 | 676.74 | 664.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 656.85 | 675.98 | 665.12 | SL hit (close<static) qty=1.00 sl=661.20 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 625.10 | 657.82 | 657.94 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 11:15:00 | 688.50 | 656.97 | 656.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 10:15:00 | 695.65 | 658.74 | 657.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 671.70 | 672.77 | 665.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 671.70 | 672.77 | 665.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 671.70 | 672.77 | 665.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 674.65 | 672.09 | 666.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 656.20 | 671.79 | 666.13 | SL hit (close<static) qty=1.00 sl=661.25 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 654.00 | 693.93 | 694.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 651.50 | 693.51 | 693.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 672.10 | 669.43 | 678.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 14:00:00 | 672.10 | 669.43 | 678.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 677.15 | 669.27 | 677.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 677.15 | 669.27 | 677.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 674.90 | 669.33 | 677.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:15:00 | 676.65 | 669.33 | 677.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 676.00 | 669.39 | 677.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 675.30 | 669.43 | 677.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:15:00 | 641.53 | 667.06 | 674.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 12:15:00 | 607.77 | 650.23 | 663.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 15:15:00 | 680.45 | 622.89 | 622.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 691.25 | 623.57 | 623.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 668.05 | 668.42 | 651.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 11:00:00 | 668.05 | 668.42 | 651.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 652.85 | 667.37 | 652.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 650.10 | 667.37 | 652.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 651.10 | 667.20 | 652.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 651.85 | 667.20 | 652.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 656.10 | 667.09 | 652.05 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 10:15:00 | 610.15 | 639.84 | 639.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 09:15:00 | 610.00 | 638.36 | 639.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 10:15:00 | 630.95 | 630.94 | 634.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-30 10:45:00 | 631.55 | 630.94 | 634.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 647.65 | 630.80 | 634.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 649.85 | 630.80 | 634.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 634.40 | 630.83 | 634.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 628.80 | 630.83 | 634.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:15:00 | 632.40 | 631.11 | 634.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 14:15:00 | 633.40 | 631.14 | 634.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 14:45:00 | 633.40 | 631.16 | 634.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 634.35 | 631.19 | 634.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 637.85 | 631.19 | 634.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 635.15 | 631.23 | 634.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 631.50 | 631.23 | 634.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:30:00 | 633.10 | 631.28 | 634.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 14:00:00 | 633.20 | 631.30 | 634.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:45:00 | 631.80 | 631.26 | 634.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 635.20 | 631.30 | 634.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 634.65 | 631.30 | 634.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 636.60 | 631.35 | 634.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 638.45 | 631.35 | 634.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 635.85 | 631.53 | 634.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 629.40 | 631.53 | 634.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 644.00 | 630.65 | 633.61 | SL hit (close>static) qty=1.00 sl=642.70 alert=retest2 |

### Cycle 10 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 657.50 | 636.11 | 636.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.15 | 636.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 637.20 | 645.55 | 641.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 637.55 | 645.47 | 641.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 636.70 | 645.47 | 641.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 637.05 | 643.08 | 640.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 636.60 | 643.08 | 640.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 635.70 | 642.58 | 640.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 635.70 | 642.58 | 640.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 642.95 | 642.45 | 640.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 645.15 | 642.49 | 640.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 645.40 | 642.49 | 640.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 645.30 | 645.47 | 642.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 645.05 | 645.47 | 642.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 638.65 | 645.57 | 642.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 638.65 | 645.57 | 642.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 640.40 | 645.52 | 642.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 641.50 | 645.52 | 642.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 15:00:00 | 642.20 | 645.49 | 642.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 642.60 | 645.37 | 642.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:15:00 | 641.50 | 645.24 | 642.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 639.50 | 645.15 | 642.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 640.60 | 645.15 | 642.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 644.20 | 645.38 | 643.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 650.95 | 645.49 | 643.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 11:15:00 | 705.65 | 660.85 | 652.07 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-30 09:15:00 | 416.15 | 2023-06-07 14:15:00 | 422.55 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2023-06-06 09:45:00 | 418.10 | 2023-06-07 14:15:00 | 422.55 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-06-06 10:15:00 | 417.25 | 2023-06-07 14:15:00 | 422.55 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-06-08 14:45:00 | 418.25 | 2023-06-13 12:15:00 | 423.55 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-06-23 09:15:00 | 411.80 | 2023-07-03 09:15:00 | 424.35 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2023-06-27 11:15:00 | 414.45 | 2023-07-03 09:15:00 | 424.35 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2023-08-21 09:15:00 | 442.35 | 2023-09-04 09:15:00 | 486.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-11 10:45:00 | 535.60 | 2024-03-13 12:15:00 | 508.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 12:00:00 | 535.80 | 2024-03-13 12:15:00 | 509.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 14:15:00 | 535.50 | 2024-03-13 12:15:00 | 508.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 15:00:00 | 533.20 | 2024-03-13 14:15:00 | 506.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 10:45:00 | 535.60 | 2024-03-15 14:15:00 | 532.65 | STOP_HIT | 0.50 | 0.55% |
| SELL | retest2 | 2024-03-11 12:00:00 | 535.80 | 2024-03-15 14:15:00 | 532.65 | STOP_HIT | 0.50 | 0.59% |
| SELL | retest2 | 2024-03-11 14:15:00 | 535.50 | 2024-03-15 14:15:00 | 532.65 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2024-03-11 15:00:00 | 533.20 | 2024-03-15 14:15:00 | 532.65 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2024-03-20 09:15:00 | 525.15 | 2024-03-21 09:15:00 | 541.80 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-06-05 10:45:00 | 648.30 | 2024-07-30 09:15:00 | 656.85 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2024-07-23 12:30:00 | 654.70 | 2024-07-30 09:15:00 | 656.85 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-07-25 10:30:00 | 645.10 | 2024-08-02 09:15:00 | 658.35 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2024-07-25 15:00:00 | 645.15 | 2024-08-02 09:15:00 | 658.35 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2024-07-26 14:15:00 | 667.80 | 2024-08-09 13:15:00 | 625.10 | STOP_HIT | 1.00 | -6.39% |
| BUY | retest2 | 2024-07-26 15:00:00 | 668.25 | 2024-08-09 13:15:00 | 625.10 | STOP_HIT | 1.00 | -6.46% |
| BUY | retest2 | 2024-07-31 09:15:00 | 667.45 | 2024-08-09 13:15:00 | 625.10 | STOP_HIT | 1.00 | -6.35% |
| BUY | retest2 | 2024-07-31 14:15:00 | 669.40 | 2024-08-09 13:15:00 | 625.10 | STOP_HIT | 1.00 | -6.62% |
| BUY | retest2 | 2024-09-06 12:15:00 | 674.65 | 2024-09-09 09:15:00 | 656.20 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-09-12 14:45:00 | 674.70 | 2024-09-27 09:15:00 | 742.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-16 09:15:00 | 682.50 | 2024-09-27 09:15:00 | 750.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 12:00:00 | 676.40 | 2024-09-27 09:15:00 | 744.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-28 10:15:00 | 693.15 | 2024-10-29 10:15:00 | 677.25 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-10-28 12:45:00 | 694.15 | 2024-10-29 10:15:00 | 677.25 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-10-29 15:00:00 | 693.55 | 2024-11-04 09:15:00 | 673.75 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-10-30 09:30:00 | 696.55 | 2024-11-04 09:15:00 | 673.75 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-12-11 12:30:00 | 675.30 | 2024-12-17 11:15:00 | 641.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:30:00 | 675.30 | 2024-12-30 12:15:00 | 607.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 628.80 | 2025-05-12 09:15:00 | 644.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-05-05 13:15:00 | 632.40 | 2025-05-12 09:15:00 | 644.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-05-05 14:15:00 | 633.40 | 2025-05-12 09:15:00 | 644.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-05-05 14:45:00 | 633.40 | 2025-05-12 09:15:00 | 644.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-05-06 11:00:00 | 631.50 | 2025-05-12 09:15:00 | 644.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-05-06 12:30:00 | 633.10 | 2025-05-12 13:15:00 | 648.75 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-05-06 14:00:00 | 633.20 | 2025-05-12 13:15:00 | 648.75 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-05-07 09:45:00 | 631.80 | 2025-05-12 13:15:00 | 648.75 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-05-08 09:15:00 | 629.40 | 2025-05-12 13:15:00 | 648.75 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-05-13 13:45:00 | 632.85 | 2025-05-14 09:15:00 | 650.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-06 12:30:00 | 645.15 | 2025-07-02 11:15:00 | 705.65 | TARGET_HIT | 1.00 | 9.38% |
| BUY | retest2 | 2025-06-06 13:00:00 | 645.40 | 2025-07-02 11:15:00 | 705.65 | TARGET_HIT | 1.00 | 9.34% |
| BUY | retest2 | 2025-06-16 10:45:00 | 645.30 | 2025-07-03 09:15:00 | 706.42 | TARGET_HIT | 1.00 | 9.47% |
| BUY | retest2 | 2025-06-16 11:15:00 | 645.05 | 2025-07-03 09:15:00 | 706.86 | TARGET_HIT | 1.00 | 9.58% |
| BUY | retest2 | 2025-06-17 14:15:00 | 641.50 | 2025-08-18 09:15:00 | 709.67 | TARGET_HIT | 1.00 | 10.63% |
| BUY | retest2 | 2025-06-17 15:00:00 | 642.20 | 2025-08-18 09:15:00 | 709.94 | TARGET_HIT | 1.00 | 10.55% |
| BUY | retest2 | 2025-06-19 11:00:00 | 642.60 | 2025-08-18 09:15:00 | 709.83 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2025-06-19 14:15:00 | 641.50 | 2025-08-18 09:15:00 | 709.56 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2025-06-23 11:30:00 | 650.95 | 2025-08-18 13:15:00 | 716.05 | TARGET_HIT | 1.00 | 10.00% |
