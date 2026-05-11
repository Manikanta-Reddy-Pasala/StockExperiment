# UPL (UPL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 644.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 7 |
| TARGET_HIT | 15 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 20
- **Target hits / Stop hits / Partials:** 15 / 21 / 7
- **Avg / median % per leg:** 3.16% / 5.00%
- **Sum % (uncompounded):** 135.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 9 | 47.4% | 9 | 10 | 0 | 3.38% | 64.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 9 | 47.4% | 9 | 10 | 0 | 3.38% | 64.2% |
| SELL (all) | 24 | 14 | 58.3% | 6 | 11 | 7 | 2.98% | 71.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 14 | 58.3% | 6 | 11 | 7 | 2.98% | 71.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 23 | 53.5% | 15 | 21 | 7 | 3.16% | 135.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 496.46 | 477.53 | 477.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 505.95 | 480.36 | 479.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 474.73 | 486.10 | 482.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 474.73 | 486.10 | 482.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 474.73 | 486.10 | 482.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 474.73 | 486.10 | 482.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 473.43 | 485.97 | 482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 473.43 | 485.97 | 482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 480.72 | 485.92 | 482.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 474.54 | 485.92 | 482.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 479.33 | 485.78 | 482.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 479.33 | 485.78 | 482.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 520.68 | 532.96 | 519.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 514.11 | 532.96 | 519.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 526.05 | 532.89 | 519.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 528.49 | 532.89 | 519.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 14:15:00 | 526.72 | 532.72 | 519.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 11:00:00 | 529.65 | 532.39 | 519.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 506.43 | 530.35 | 519.54 | SL hit (close<static) qty=1.00 sl=511.13 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 15:15:00 | 510.56 | 553.16 | 553.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 501.40 | 549.92 | 551.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 539.81 | 538.66 | 544.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 11:45:00 | 540.29 | 538.66 | 544.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 544.13 | 538.77 | 544.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:15:00 | 544.37 | 538.77 | 544.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 544.37 | 538.83 | 544.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 547.63 | 538.83 | 544.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 548.06 | 538.92 | 544.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:45:00 | 543.65 | 539.06 | 544.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 543.03 | 539.12 | 544.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 543.36 | 539.27 | 544.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:45:00 | 543.46 | 539.30 | 544.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 539.09 | 539.30 | 544.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 538.18 | 539.30 | 544.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 13:30:00 | 536.70 | 539.22 | 544.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 516.47 | 538.87 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 515.88 | 538.87 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 516.19 | 538.87 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 516.29 | 538.87 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 511.27 | 538.87 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 509.87 | 538.87 | 544.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-11 14:15:00 | 489.28 | 537.53 | 543.47 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 552.35 | 542.79 | 542.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 563.90 | 544.33 | 543.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 543.85 | 545.07 | 543.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 543.85 | 545.07 | 543.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 543.85 | 545.07 | 543.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 543.85 | 545.07 | 543.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 546.75 | 545.09 | 543.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 14:00:00 | 547.90 | 545.12 | 544.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 15:00:00 | 548.65 | 545.15 | 544.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 538.70 | 545.11 | 544.01 | SL hit (close<static) qty=1.00 sl=542.95 alert=retest2 |

### Cycle 4 — SELL (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 14:15:00 | 518.45 | 543.00 | 543.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 512.35 | 542.46 | 542.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 12:15:00 | 529.00 | 524.31 | 532.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 529.00 | 524.31 | 532.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 529.80 | 524.43 | 532.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 532.30 | 524.43 | 532.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 530.95 | 524.55 | 532.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 522.15 | 524.59 | 532.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 12:15:00 | 523.25 | 524.59 | 532.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:15:00 | 522.70 | 524.59 | 532.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 09:15:00 | 539.65 | 524.61 | 531.87 | SL hit (close>static) qty=1.00 sl=535.45 alert=retest2 |

### Cycle 5 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 554.50 | 536.56 | 536.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 556.75 | 536.94 | 536.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 11:30:00 | 537.70 | 538.63 | 537.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 540.55 | 538.65 | 537.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 14:00:00 | 541.70 | 538.68 | 537.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 11:45:00 | 543.30 | 538.82 | 537.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 09:30:00 | 544.30 | 539.03 | 537.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-31 14:15:00 | 595.87 | 544.22 | 540.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 650.95 | 688.12 | 688.12 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 685.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 741.00 | 724.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 736.25 | 741.00 | 724.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.21 | 757.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:00:00 | 763.70 | 775.21 | 757.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 755.75 | 775.02 | 757.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 756.55 | 775.02 | 757.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 747.35 | 774.74 | 757.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 747.35 | 774.74 | 757.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 740.55 | 774.40 | 757.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 740.55 | 774.40 | 757.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 680.75 | 744.85 | 745.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 663.85 | 744.04 | 744.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:00:00 | 745.45 | 739.26 | 742.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 742.95 | 739.30 | 742.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 746.40 | 739.30 | 742.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 757.70 | 739.45 | 742.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 757.70 | 739.45 | 742.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 756.30 | 739.62 | 742.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 757.65 | 739.62 | 742.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 740.65 | 740.53 | 742.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 09:45:00 | 734.70 | 740.79 | 742.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 745.15 | 740.85 | 742.58 | SL hit (close>static) qty=1.00 sl=743.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-22 10:15:00 | 528.49 | 2024-07-25 09:15:00 | 506.43 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2024-07-22 14:15:00 | 526.72 | 2024-07-25 09:15:00 | 506.43 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2024-07-23 11:00:00 | 529.65 | 2024-07-25 09:15:00 | 506.43 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2024-07-29 09:15:00 | 530.03 | 2024-08-05 12:15:00 | 509.07 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-08-07 15:00:00 | 523.70 | 2024-08-30 12:15:00 | 576.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 12:45:00 | 523.32 | 2024-08-30 12:15:00 | 575.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 11:45:00 | 522.74 | 2024-08-30 12:15:00 | 575.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 13:15:00 | 523.32 | 2024-08-30 12:15:00 | 575.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 14:45:00 | 530.41 | 2024-09-04 10:15:00 | 583.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 09:15:00 | 533.24 | 2024-09-05 09:15:00 | 586.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-17 14:15:00 | 531.95 | 2024-10-21 15:15:00 | 521.68 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-10-18 11:30:00 | 530.85 | 2024-10-21 15:15:00 | 521.68 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-11-07 11:45:00 | 543.65 | 2024-11-11 09:15:00 | 516.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 543.03 | 2024-11-11 09:15:00 | 515.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 543.36 | 2024-11-11 09:15:00 | 516.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:45:00 | 543.46 | 2024-11-11 09:15:00 | 516.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 538.18 | 2024-11-11 09:15:00 | 511.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 13:30:00 | 536.70 | 2024-11-11 09:15:00 | 509.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 11:45:00 | 543.65 | 2024-11-11 14:15:00 | 489.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 543.03 | 2024-11-13 09:15:00 | 488.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 543.36 | 2024-11-13 09:15:00 | 489.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 09:45:00 | 543.46 | 2024-11-13 09:15:00 | 489.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 538.18 | 2024-11-13 10:15:00 | 484.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 13:30:00 | 536.70 | 2024-11-19 12:15:00 | 531.66 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2024-11-21 12:30:00 | 535.11 | 2024-11-25 09:15:00 | 550.46 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-11-21 14:30:00 | 535.79 | 2024-11-25 09:15:00 | 550.46 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-12-12 14:00:00 | 547.90 | 2024-12-13 09:15:00 | 538.70 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-12-12 15:00:00 | 548.65 | 2024-12-13 09:15:00 | 538.70 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-12-13 14:45:00 | 549.60 | 2024-12-17 12:15:00 | 542.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-12-16 12:15:00 | 547.85 | 2024-12-17 12:15:00 | 542.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-01-06 11:45:00 | 522.15 | 2025-01-07 09:15:00 | 539.65 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-01-06 12:15:00 | 523.25 | 2025-01-07 09:15:00 | 539.65 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-01-06 13:15:00 | 522.70 | 2025-01-07 09:15:00 | 539.65 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-01-27 14:00:00 | 541.70 | 2025-01-31 14:15:00 | 595.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-28 11:45:00 | 543.30 | 2025-01-31 14:15:00 | 597.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-29 09:30:00 | 544.30 | 2025-01-31 14:15:00 | 598.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-09 09:45:00 | 734.70 | 2026-02-09 11:15:00 | 745.15 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-13 09:15:00 | 732.55 | 2026-02-18 09:15:00 | 749.65 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-02-13 10:45:00 | 734.95 | 2026-02-18 09:15:00 | 749.65 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-02-13 11:15:00 | 734.70 | 2026-02-18 09:15:00 | 749.65 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-18 12:15:00 | 744.95 | 2026-02-19 09:15:00 | 754.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-02-23 09:15:00 | 679.80 | 2026-02-23 10:15:00 | 645.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:15:00 | 679.80 | 2026-03-04 09:15:00 | 611.82 | TARGET_HIT | 0.50 | 10.00% |
