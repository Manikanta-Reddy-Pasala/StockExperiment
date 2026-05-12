# Jubilant Ingrevia Ltd. (JUBLINGREA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 743.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 63 |
| PARTIAL | 20 |
| TARGET_HIT | 11 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 38
- **Target hits / Stop hits / Partials:** 11 / 52 / 20
- **Avg / median % per leg:** 1.72% / 1.65%
- **Sum % (uncompounded):** 142.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 5 | 17.2% | 5 | 24 | 0 | -0.90% | -26.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 5 | 17.2% | 5 | 24 | 0 | -0.90% | -26.1% |
| SELL (all) | 54 | 40 | 74.1% | 6 | 28 | 20 | 3.13% | 168.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 40 | 74.1% | 6 | 28 | 20 | 3.13% | 168.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 83 | 45 | 54.2% | 11 | 52 | 20 | 1.72% | 142.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 426.90 | 462.80 | 462.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 11:15:00 | 419.85 | 462.38 | 462.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 436.30 | 434.89 | 444.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 09:30:00 | 436.85 | 434.89 | 444.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 436.30 | 433.31 | 442.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 11:15:00 | 434.95 | 433.66 | 442.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 09:15:00 | 444.25 | 433.93 | 441.90 | SL hit (close>static) qty=1.00 sl=442.95 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 13:15:00 | 481.20 | 447.07 | 447.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 12:15:00 | 483.20 | 449.08 | 448.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 476.00 | 477.68 | 465.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 10:00:00 | 476.00 | 477.68 | 465.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 465.10 | 480.19 | 469.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 461.95 | 480.19 | 469.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 466.20 | 480.05 | 469.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:45:00 | 467.65 | 479.94 | 469.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 472.80 | 479.38 | 469.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 462.70 | 478.48 | 470.10 | SL hit (close<static) qty=1.00 sl=463.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 428.65 | 464.19 | 464.30 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 488.30 | 463.33 | 463.23 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 11:15:00 | 437.55 | 465.98 | 466.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 12:15:00 | 433.10 | 465.65 | 465.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 13:15:00 | 458.75 | 457.77 | 461.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-22 13:30:00 | 457.25 | 457.77 | 461.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 460.30 | 457.79 | 461.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 09:15:00 | 453.35 | 457.80 | 461.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-01 11:15:00 | 462.75 | 456.25 | 460.11 | SL hit (close>static) qty=1.00 sl=462.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 09:15:00 | 494.50 | 463.20 | 463.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 11:15:00 | 523.70 | 465.98 | 464.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 13:15:00 | 519.00 | 519.43 | 500.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 13:45:00 | 519.60 | 519.43 | 500.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 529.90 | 518.85 | 502.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 10:15:00 | 531.50 | 518.85 | 502.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:15:00 | 530.85 | 521.80 | 505.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 536.00 | 522.18 | 505.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 497.10 | 518.91 | 507.51 | SL hit (close<static) qty=1.00 sl=500.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 10:15:00 | 690.30 | 753.48 | 753.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 11:15:00 | 686.95 | 752.82 | 753.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 734.00 | 726.03 | 737.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 10:15:00 | 734.00 | 726.03 | 737.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 734.00 | 726.03 | 737.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 734.00 | 726.03 | 737.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 743.30 | 726.20 | 737.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:00:00 | 743.30 | 726.20 | 737.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 732.05 | 726.26 | 737.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 12:30:00 | 734.00 | 726.26 | 737.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 735.85 | 725.84 | 737.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 735.85 | 725.84 | 737.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 742.15 | 726.00 | 737.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 743.10 | 726.00 | 737.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 742.75 | 726.17 | 737.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:15:00 | 744.00 | 726.17 | 737.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 734.45 | 727.02 | 737.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:45:00 | 727.25 | 727.25 | 737.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 740.00 | 727.30 | 736.98 | SL hit (close>static) qty=1.00 sl=739.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 687.00 | 681.07 | 681.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 692.50 | 681.26 | 681.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 676.50 | 682.88 | 682.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 676.50 | 682.88 | 682.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 675.50 | 682.81 | 681.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 674.60 | 682.81 | 681.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 678.95 | 682.69 | 681.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 678.95 | 682.69 | 681.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 672.20 | 682.59 | 681.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 672.20 | 682.59 | 681.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 682.95 | 682.35 | 681.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:30:00 | 681.90 | 682.35 | 681.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 677.10 | 682.30 | 681.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 675.55 | 682.30 | 681.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 677.40 | 682.25 | 681.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 677.40 | 682.25 | 681.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 680.95 | 682.16 | 681.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 681.50 | 682.16 | 681.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 677.45 | 682.11 | 681.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:15:00 | 678.00 | 682.11 | 681.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 679.05 | 682.08 | 681.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:45:00 | 685.30 | 682.11 | 681.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 14:00:00 | 683.00 | 682.14 | 681.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 675.65 | 682.24 | 681.78 | SL hit (close<static) qty=1.00 sl=676.05 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 701.00 | 744.97 | 745.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 690.10 | 723.49 | 731.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 686.00 | 680.94 | 701.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:30:00 | 683.70 | 680.94 | 701.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 699.55 | 681.19 | 701.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 699.55 | 681.19 | 701.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 699.35 | 681.37 | 701.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 708.15 | 681.37 | 701.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 704.50 | 681.60 | 701.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 704.50 | 681.60 | 701.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 703.65 | 681.82 | 701.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 705.00 | 681.82 | 701.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 701.90 | 682.02 | 701.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 690.45 | 682.02 | 701.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 697.40 | 682.91 | 701.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:45:00 | 692.25 | 683.25 | 701.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 699.05 | 685.22 | 701.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 695.55 | 685.33 | 701.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 694.65 | 685.33 | 701.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:45:00 | 694.95 | 685.61 | 701.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 690.90 | 685.75 | 701.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 693.75 | 684.53 | 697.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 697.50 | 685.30 | 697.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 694.50 | 685.76 | 697.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 694.80 | 685.83 | 697.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:45:00 | 693.00 | 685.99 | 697.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 693.90 | 686.18 | 697.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 14:15:00 | 664.10 | 685.89 | 696.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 15:15:00 | 662.53 | 685.66 | 695.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 655.93 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 657.64 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.92 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 660.20 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 656.35 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.06 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.77 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 660.06 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 658.35 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 13:15:00 | 659.20 | 684.58 | 695.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 681.55 | 680.58 | 691.53 | SL hit (close>ema200) qty=0.50 sl=680.58 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 707.60 | 698.41 | 698.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 719.00 | 698.70 | 698.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 694.40 | 700.02 | 699.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:45:00 | 695.05 | 700.02 | 699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 690.00 | 699.92 | 699.22 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 683.30 | 698.44 | 698.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 680.55 | 698.26 | 698.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 694.65 | 692.36 | 695.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 694.65 | 692.36 | 695.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 693.45 | 692.37 | 695.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 692.55 | 693.91 | 695.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 692.65 | 693.90 | 695.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:45:00 | 692.10 | 693.88 | 695.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 692.60 | 693.86 | 695.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 698.70 | 692.78 | 695.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 698.70 | 692.78 | 695.04 | SL hit (close>static) qty=1.00 sl=696.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 14:15:00 | 710.60 | 697.10 | 697.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 722.70 | 697.49 | 697.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 697.50 | 699.42 | 698.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 697.50 | 699.42 | 698.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 698.15 | 699.41 | 698.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:00:00 | 701.05 | 699.42 | 698.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 694.05 | 699.37 | 698.25 | SL hit (close<static) qty=1.00 sl=695.85 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 669.80 | 699.89 | 699.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 667.45 | 699.57 | 699.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 674.05 | 668.58 | 681.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:00:00 | 674.05 | 668.58 | 681.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.50 | 661.35 | 675.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 664.95 | 661.43 | 675.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 12:00:00 | 663.95 | 661.43 | 675.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 662.90 | 662.04 | 675.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 664.45 | 662.10 | 675.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 669.25 | 662.21 | 675.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 673.80 | 662.21 | 675.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 672.75 | 662.31 | 675.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 665.15 | 662.33 | 675.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 631.70 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 630.75 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 629.75 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 631.23 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 631.89 | 660.94 | 673.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-20 12:15:00 | 598.46 | 650.68 | 666.55 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 14 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 717.05 | 623.93 | 623.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 10:15:00 | 738.45 | 625.07 | 624.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-12-05 11:15:00 | 434.95 | 2023-12-06 09:15:00 | 444.25 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-01-18 11:45:00 | 467.65 | 2024-01-23 11:15:00 | 462.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-01-19 09:15:00 | 472.80 | 2024-01-23 11:15:00 | 462.70 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-01-23 13:30:00 | 468.20 | 2024-01-24 09:15:00 | 463.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-01-24 11:45:00 | 467.75 | 2024-01-24 15:15:00 | 463.25 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-01-25 09:15:00 | 467.80 | 2024-01-25 10:15:00 | 459.95 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-03-26 09:15:00 | 453.35 | 2024-04-01 11:15:00 | 462.75 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-04-03 14:00:00 | 457.85 | 2024-04-04 10:15:00 | 463.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-05-15 10:15:00 | 531.50 | 2024-05-29 09:15:00 | 497.10 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest2 | 2024-05-17 10:15:00 | 530.85 | 2024-05-29 09:15:00 | 497.10 | STOP_HIT | 1.00 | -6.36% |
| BUY | retest2 | 2024-05-18 09:15:00 | 536.00 | 2024-05-29 09:15:00 | 497.10 | STOP_HIT | 1.00 | -7.26% |
| BUY | retest2 | 2024-06-19 12:30:00 | 532.85 | 2024-07-09 12:15:00 | 586.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-22 11:15:00 | 712.90 | 2024-10-24 10:15:00 | 680.45 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2024-10-23 13:30:00 | 714.70 | 2024-10-24 10:15:00 | 680.45 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2024-10-30 12:45:00 | 713.00 | 2024-11-11 10:15:00 | 698.60 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-10-31 09:30:00 | 713.30 | 2024-11-11 10:15:00 | 698.60 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-11-05 14:45:00 | 727.30 | 2024-11-12 14:15:00 | 682.00 | STOP_HIT | 1.00 | -6.23% |
| BUY | retest2 | 2024-11-06 09:15:00 | 738.40 | 2024-11-12 14:15:00 | 682.00 | STOP_HIT | 1.00 | -7.64% |
| BUY | retest2 | 2024-11-26 13:45:00 | 730.00 | 2024-12-04 10:15:00 | 803.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-16 09:15:00 | 728.90 | 2025-01-20 09:15:00 | 697.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-02-06 13:45:00 | 727.25 | 2025-02-07 10:15:00 | 740.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-02-10 09:15:00 | 724.00 | 2025-02-11 09:15:00 | 687.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 724.00 | 2025-02-14 09:15:00 | 651.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-13 10:30:00 | 728.65 | 2025-05-13 14:15:00 | 692.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-13 10:30:00 | 728.65 | 2025-05-13 14:15:00 | 702.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2025-05-13 12:45:00 | 728.75 | 2025-05-13 14:15:00 | 692.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-13 12:45:00 | 728.75 | 2025-05-13 14:15:00 | 702.00 | STOP_HIT | 0.50 | 3.67% |
| BUY | retest2 | 2025-05-23 11:45:00 | 685.30 | 2025-05-27 09:15:00 | 675.65 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-05-23 14:00:00 | 683.00 | 2025-05-27 09:15:00 | 675.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-28 09:15:00 | 691.50 | 2025-06-11 13:15:00 | 674.80 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-06-11 12:00:00 | 682.80 | 2025-06-11 13:15:00 | 674.80 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-23 10:00:00 | 715.80 | 2025-07-03 13:15:00 | 787.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 13:00:00 | 715.15 | 2025-08-11 09:15:00 | 685.75 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-08-19 09:15:00 | 716.45 | 2025-08-19 15:15:00 | 701.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-19 09:45:00 | 719.35 | 2025-08-19 15:15:00 | 701.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-10-14 09:15:00 | 690.45 | 2025-11-06 14:15:00 | 664.10 | PARTIAL | 0.50 | 3.82% |
| SELL | retest2 | 2025-10-15 09:15:00 | 697.40 | 2025-11-06 15:15:00 | 662.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 11:45:00 | 692.25 | 2025-11-07 13:15:00 | 655.93 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2025-10-17 10:00:00 | 699.05 | 2025-11-07 13:15:00 | 657.64 | PARTIAL | 0.50 | 5.92% |
| SELL | retest2 | 2025-10-17 11:15:00 | 694.65 | 2025-11-07 13:15:00 | 659.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 14:45:00 | 694.95 | 2025-11-07 13:15:00 | 660.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 09:15:00 | 690.90 | 2025-11-07 13:15:00 | 656.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:45:00 | 693.75 | 2025-11-07 13:15:00 | 659.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:45:00 | 694.50 | 2025-11-07 13:15:00 | 659.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:30:00 | 694.80 | 2025-11-07 13:15:00 | 660.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 09:45:00 | 693.00 | 2025-11-07 13:15:00 | 658.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:15:00 | 693.90 | 2025-11-07 13:15:00 | 659.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-14 09:15:00 | 690.45 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-10-15 09:15:00 | 697.40 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2025-10-15 11:45:00 | 692.25 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-10-17 10:00:00 | 699.05 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-10-17 11:15:00 | 694.65 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-10-17 14:45:00 | 694.95 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-10-20 09:15:00 | 690.90 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-10-29 09:45:00 | 693.75 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-10-31 09:45:00 | 694.50 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-10-31 10:30:00 | 694.80 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2025-11-03 09:45:00 | 693.00 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-11-03 14:15:00 | 693.90 | 2025-11-13 14:15:00 | 681.55 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2025-12-17 10:30:00 | 692.55 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-17 12:00:00 | 692.65 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-17 13:45:00 | 692.10 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-17 14:30:00 | 692.60 | 2025-12-19 09:15:00 | 698.70 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-19 12:30:00 | 697.05 | 2025-12-19 14:15:00 | 705.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-12-19 13:45:00 | 697.35 | 2025-12-19 14:15:00 | 705.65 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-19 14:30:00 | 696.25 | 2025-12-19 15:15:00 | 707.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-30 14:00:00 | 701.05 | 2025-12-30 14:15:00 | 694.05 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-31 09:15:00 | 700.40 | 2026-01-05 10:15:00 | 770.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 10:15:00 | 704.45 | 2026-01-05 10:15:00 | 774.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-09 09:45:00 | 702.80 | 2026-01-09 11:15:00 | 692.25 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-10 11:30:00 | 664.95 | 2026-02-16 09:15:00 | 631.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 12:00:00 | 663.95 | 2026-02-16 09:15:00 | 630.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 662.90 | 2026-02-16 09:15:00 | 629.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 664.45 | 2026-02-16 09:15:00 | 631.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 14:30:00 | 665.15 | 2026-02-16 09:15:00 | 631.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:30:00 | 664.95 | 2026-02-20 12:15:00 | 598.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-10 12:00:00 | 663.95 | 2026-02-20 12:15:00 | 597.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 662.90 | 2026-02-20 12:15:00 | 598.01 | TARGET_HIT | 0.50 | 9.79% |
| SELL | retest2 | 2026-02-12 11:15:00 | 664.45 | 2026-02-20 12:15:00 | 598.63 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2026-02-12 14:30:00 | 665.15 | 2026-02-20 13:15:00 | 596.61 | TARGET_HIT | 0.50 | 10.30% |
| SELL | retest2 | 2026-04-15 14:15:00 | 668.70 | 2026-04-22 09:15:00 | 680.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-04-16 09:45:00 | 668.50 | 2026-04-22 09:15:00 | 680.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-16 11:45:00 | 668.80 | 2026-04-22 09:15:00 | 680.80 | STOP_HIT | 1.00 | -1.79% |
