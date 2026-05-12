# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3043.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 142 |
| ALERT2 | 137 |
| ALERT2_SKIP | 77 |
| ALERT3 | 339 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 129 |
| PARTIAL | 22 |
| TARGET_HIT | 25 |
| STOP_HIT | 112 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 156 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 88
- **Target hits / Stop hits / Partials:** 25 / 109 / 22
- **Avg / median % per leg:** 1.28% / -0.39%
- **Sum % (uncompounded):** 199.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 21 | 30.4% | 17 | 52 | 0 | 1.11% | 76.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.75% | -3.0% |
| BUY @ 3rd Alert (retest2) | 65 | 21 | 32.3% | 17 | 48 | 0 | 1.23% | 79.9% |
| SELL (all) | 87 | 47 | 54.0% | 8 | 57 | 22 | 1.41% | 123.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.37% | -3.4% |
| SELL @ 3rd Alert (retest2) | 86 | 47 | 54.7% | 8 | 56 | 22 | 1.47% | 126.4% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.28% | -6.4% |
| retest2 (combined) | 151 | 68 | 45.0% | 25 | 104 | 22 | 1.37% | 206.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 514.60 | 514.91 | 514.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 505.60 | 512.81 | 513.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 15:15:00 | 481.50 | 480.81 | 487.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-24 09:15:00 | 476.20 | 480.81 | 487.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 484.00 | 482.07 | 485.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 14:15:00 | 482.50 | 482.07 | 485.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 15:15:00 | 480.00 | 482.35 | 485.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:15:00 | 458.38 | 476.86 | 482.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-25 09:15:00 | 456.00 | 476.86 | 482.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-05-26 10:15:00 | 465.35 | 462.20 | 469.74 | SL hit (close>ema200) qty=0.50 sl=462.20 alert=retest2 |

### Cycle 2 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 462.00 | 458.42 | 458.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 12:15:00 | 466.05 | 460.51 | 459.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 14:15:00 | 504.50 | 505.38 | 497.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 15:00:00 | 504.50 | 505.38 | 497.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 500.55 | 504.02 | 499.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 500.55 | 504.02 | 499.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 499.70 | 503.16 | 499.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:45:00 | 499.00 | 503.16 | 499.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 13:15:00 | 498.40 | 502.21 | 499.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 14:00:00 | 498.40 | 502.21 | 499.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 498.10 | 501.39 | 499.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 498.10 | 501.39 | 499.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 497.30 | 500.57 | 499.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:15:00 | 495.15 | 500.57 | 499.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 496.65 | 499.78 | 498.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 492.85 | 499.78 | 498.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 497.50 | 499.33 | 498.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 11:15:00 | 509.30 | 499.33 | 498.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-12 14:15:00 | 560.23 | 537.60 | 521.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 588.00 | 588.96 | 589.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 577.75 | 586.72 | 587.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 574.90 | 567.51 | 573.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 574.90 | 567.51 | 573.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 574.90 | 567.51 | 573.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 574.90 | 567.51 | 573.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 580.55 | 570.12 | 574.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 580.55 | 570.12 | 574.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 580.00 | 572.10 | 574.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:30:00 | 580.60 | 572.10 | 574.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 583.35 | 574.72 | 575.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:00:00 | 583.35 | 574.72 | 575.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 589.55 | 577.69 | 576.71 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 12:15:00 | 576.55 | 579.77 | 580.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 12:15:00 | 572.60 | 577.57 | 578.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 581.05 | 577.22 | 578.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 581.05 | 577.22 | 578.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 581.05 | 577.22 | 578.13 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 12:15:00 | 579.75 | 578.81 | 578.71 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 13:15:00 | 577.60 | 578.56 | 578.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 15:15:00 | 575.80 | 577.79 | 578.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 568.90 | 568.14 | 571.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 568.90 | 568.14 | 571.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 568.90 | 568.14 | 571.98 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 15:15:00 | 574.50 | 569.89 | 569.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 10:15:00 | 586.55 | 573.23 | 571.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 14:15:00 | 610.60 | 612.58 | 603.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 15:00:00 | 610.60 | 612.58 | 603.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 607.10 | 612.98 | 607.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 13:45:00 | 609.00 | 612.98 | 607.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 601.70 | 610.72 | 607.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 15:00:00 | 601.70 | 610.72 | 607.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 15:15:00 | 605.70 | 609.72 | 606.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:15:00 | 597.70 | 609.72 | 606.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 594.65 | 604.34 | 604.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 590.00 | 599.82 | 602.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 09:15:00 | 609.00 | 597.43 | 600.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 09:15:00 | 609.00 | 597.43 | 600.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 609.00 | 597.43 | 600.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:00:00 | 609.00 | 597.43 | 600.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 615.20 | 600.98 | 601.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:15:00 | 617.60 | 600.98 | 601.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 614.95 | 603.78 | 602.74 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 601.60 | 603.63 | 603.73 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 13:15:00 | 605.50 | 604.01 | 603.89 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 09:15:00 | 597.00 | 602.69 | 603.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 15:15:00 | 591.90 | 597.07 | 599.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 615.90 | 600.83 | 601.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 615.90 | 600.83 | 601.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 615.90 | 600.83 | 601.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:00:00 | 615.90 | 600.83 | 601.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 10:15:00 | 615.80 | 603.83 | 602.64 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 599.25 | 604.26 | 604.80 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 630.65 | 609.32 | 606.49 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 15:15:00 | 608.80 | 609.27 | 609.29 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 10:15:00 | 612.80 | 609.77 | 609.50 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 14:15:00 | 606.70 | 609.37 | 609.48 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 619.00 | 611.41 | 610.39 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 610.60 | 618.47 | 618.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 12:15:00 | 607.40 | 616.25 | 617.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 609.35 | 607.19 | 611.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 609.35 | 607.19 | 611.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 606.50 | 607.05 | 611.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 11:30:00 | 614.00 | 607.05 | 611.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 605.45 | 605.03 | 608.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:30:00 | 603.50 | 604.73 | 607.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:15:00 | 602.85 | 604.89 | 607.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 10:15:00 | 630.70 | 599.59 | 596.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 10:15:00 | 630.70 | 599.59 | 596.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 11:15:00 | 648.00 | 609.28 | 601.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 15:15:00 | 775.50 | 776.49 | 738.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-18 09:15:00 | 761.80 | 776.49 | 738.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 746.50 | 763.25 | 743.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:45:00 | 748.40 | 763.25 | 743.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 752.85 | 761.17 | 744.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 13:30:00 | 745.30 | 761.17 | 744.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 748.60 | 758.65 | 744.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 748.60 | 758.65 | 744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 753.90 | 757.70 | 745.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:15:00 | 745.05 | 757.70 | 745.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 755.50 | 757.26 | 746.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 11:15:00 | 768.85 | 757.78 | 747.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 15:15:00 | 770.20 | 757.23 | 750.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-22 09:45:00 | 767.00 | 760.43 | 753.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 774.00 | 761.48 | 757.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 789.95 | 788.26 | 782.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 796.00 | 788.26 | 782.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 832.35 | 787.27 | 784.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-28 09:15:00 | 843.70 | 793.00 | 787.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 837.60 | 881.65 | 883.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 789.80 | 829.72 | 852.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 09:15:00 | 826.90 | 802.91 | 813.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 826.90 | 802.91 | 813.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 826.90 | 802.91 | 813.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 10:00:00 | 826.90 | 802.91 | 813.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 841.40 | 810.61 | 816.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 10:45:00 | 851.60 | 810.61 | 816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 12:15:00 | 842.90 | 822.47 | 820.86 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 811.50 | 827.72 | 829.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 12:15:00 | 803.60 | 817.60 | 824.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 829.85 | 805.78 | 810.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 829.85 | 805.78 | 810.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 829.85 | 805.78 | 810.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 829.85 | 805.78 | 810.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 837.25 | 812.07 | 812.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:45:00 | 836.45 | 812.07 | 812.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 832.05 | 816.07 | 814.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 13:15:00 | 859.00 | 826.16 | 819.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 14:15:00 | 833.95 | 834.41 | 828.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 14:45:00 | 833.40 | 834.41 | 828.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 832.50 | 834.29 | 829.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:30:00 | 827.15 | 834.29 | 829.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 830.00 | 833.43 | 829.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 830.00 | 833.43 | 829.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 831.85 | 833.11 | 829.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:15:00 | 829.50 | 833.11 | 829.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 826.55 | 831.80 | 829.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:00:00 | 826.55 | 831.80 | 829.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 828.50 | 831.14 | 829.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 849.25 | 830.47 | 829.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:30:00 | 833.95 | 838.45 | 835.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 11:00:00 | 832.95 | 837.35 | 835.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 12:00:00 | 831.70 | 836.22 | 835.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 830.00 | 834.53 | 834.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:00:00 | 830.00 | 834.53 | 834.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-03 14:15:00 | 830.65 | 833.75 | 834.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 14:15:00 | 830.65 | 833.75 | 834.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 813.55 | 829.10 | 831.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 830.35 | 816.70 | 822.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 830.35 | 816.70 | 822.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 830.35 | 816.70 | 822.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 09:30:00 | 835.25 | 816.70 | 822.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 10:15:00 | 827.05 | 818.77 | 822.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 13:00:00 | 821.00 | 820.37 | 822.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 14:00:00 | 821.80 | 820.66 | 822.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 14:45:00 | 821.70 | 820.47 | 822.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 10:15:00 | 805.30 | 804.39 | 804.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 805.30 | 804.39 | 804.33 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 802.75 | 804.01 | 804.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 13:15:00 | 798.50 | 802.91 | 803.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 802.10 | 799.48 | 800.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 802.10 | 799.48 | 800.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 802.10 | 799.48 | 800.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:30:00 | 803.60 | 799.48 | 800.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 797.55 | 799.09 | 800.48 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 14:15:00 | 810.25 | 802.19 | 801.61 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 11:15:00 | 799.40 | 804.09 | 804.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 12:15:00 | 793.90 | 802.05 | 803.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 12:15:00 | 695.85 | 683.74 | 704.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 12:30:00 | 696.10 | 683.74 | 704.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 717.00 | 690.39 | 705.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:00:00 | 717.00 | 690.39 | 705.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 737.45 | 699.81 | 708.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:30:00 | 752.80 | 699.81 | 708.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 09:15:00 | 769.90 | 721.85 | 717.47 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 11:15:00 | 738.50 | 741.68 | 741.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 733.75 | 740.10 | 741.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 739.60 | 738.15 | 739.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 739.60 | 738.15 | 739.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 739.60 | 738.15 | 739.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 12:00:00 | 737.65 | 738.45 | 739.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 15:15:00 | 741.50 | 740.03 | 740.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 15:15:00 | 741.50 | 740.03 | 740.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 743.55 | 740.74 | 740.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 743.00 | 743.20 | 741.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 14:15:00 | 743.00 | 743.20 | 741.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 743.00 | 743.20 | 741.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 15:00:00 | 743.00 | 743.20 | 741.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 742.90 | 743.14 | 742.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 745.75 | 743.14 | 742.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 11:30:00 | 762.50 | 747.17 | 744.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 10:15:00 | 769.45 | 775.85 | 776.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 10:15:00 | 769.45 | 775.85 | 776.04 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 13:15:00 | 795.15 | 776.42 | 774.90 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 776.00 | 780.44 | 780.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 15:15:00 | 774.45 | 778.84 | 780.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 10:15:00 | 785.65 | 778.99 | 779.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 10:15:00 | 785.65 | 778.99 | 779.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 785.65 | 778.99 | 779.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:45:00 | 793.75 | 778.99 | 779.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 11:15:00 | 786.65 | 780.52 | 780.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 10:15:00 | 807.45 | 788.06 | 784.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 11:15:00 | 806.10 | 806.40 | 798.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-22 11:45:00 | 806.65 | 806.40 | 798.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 799.65 | 805.05 | 798.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 821.10 | 804.31 | 799.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 09:15:00 | 819.35 | 828.38 | 828.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 819.35 | 828.38 | 828.54 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 839.00 | 830.51 | 829.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 843.05 | 833.01 | 830.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 09:15:00 | 838.50 | 839.19 | 835.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 10:00:00 | 838.50 | 839.19 | 835.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 837.80 | 838.91 | 835.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 10:30:00 | 835.00 | 838.91 | 835.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 834.00 | 837.93 | 835.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 11:30:00 | 834.60 | 837.93 | 835.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 836.55 | 837.65 | 835.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 842.20 | 836.68 | 835.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 13:45:00 | 838.35 | 838.51 | 837.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:15:00 | 842.40 | 839.17 | 837.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 832.70 | 838.38 | 837.74 | SL hit (close<static) qty=1.00 sl=833.55 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 14:15:00 | 836.25 | 837.36 | 837.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 09:15:00 | 832.70 | 836.15 | 836.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 10:15:00 | 840.40 | 837.00 | 837.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 10:15:00 | 840.40 | 837.00 | 837.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 840.40 | 837.00 | 837.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 10:30:00 | 845.50 | 837.00 | 837.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 11:15:00 | 841.45 | 837.89 | 837.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 12:15:00 | 857.00 | 841.71 | 839.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 14:15:00 | 848.45 | 852.99 | 848.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 14:15:00 | 848.45 | 852.99 | 848.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 848.45 | 852.99 | 848.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 15:00:00 | 848.45 | 852.99 | 848.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 15:15:00 | 847.00 | 851.79 | 848.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 09:15:00 | 861.50 | 851.79 | 848.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 10:15:00 | 844.50 | 849.94 | 848.38 | SL hit (close<static) qty=1.00 sl=845.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 836.50 | 846.83 | 847.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 10:15:00 | 827.15 | 833.10 | 835.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-18 09:15:00 | 833.75 | 830.64 | 832.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 833.75 | 830.64 | 832.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 833.75 | 830.64 | 832.65 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 12:15:00 | 840.00 | 835.03 | 834.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 09:15:00 | 860.05 | 839.82 | 836.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 15:15:00 | 848.75 | 850.29 | 844.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:15:00 | 854.55 | 850.29 | 844.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 850.55 | 849.87 | 845.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-20 11:15:00 | 844.90 | 848.87 | 845.35 | SL hit (close<ema400) qty=1.00 sl=845.35 alert=retest1 |

### Cycle 45 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 807.45 | 837.22 | 840.45 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 858.10 | 828.48 | 827.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 874.25 | 837.64 | 831.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 864.05 | 869.50 | 860.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 10:00:00 | 864.05 | 869.50 | 860.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 861.50 | 867.90 | 860.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:00:00 | 861.50 | 867.90 | 860.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 859.95 | 866.31 | 860.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 11:30:00 | 860.85 | 866.31 | 860.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 12:15:00 | 862.60 | 865.57 | 860.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 12:45:00 | 862.50 | 865.57 | 860.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 863.30 | 865.12 | 860.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:30:00 | 861.00 | 865.12 | 860.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 862.90 | 864.67 | 860.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 14:30:00 | 862.55 | 864.67 | 860.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 856.05 | 862.95 | 860.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 871.05 | 862.95 | 860.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:00:00 | 865.15 | 871.69 | 870.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:45:00 | 864.05 | 869.34 | 869.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 11:15:00 | 864.05 | 869.34 | 869.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 11:15:00 | 863.70 | 868.21 | 868.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 863.70 | 868.21 | 868.65 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 15:15:00 | 869.50 | 867.91 | 867.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 876.60 | 869.65 | 868.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 10:15:00 | 863.85 | 868.49 | 868.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 10:15:00 | 863.85 | 868.49 | 868.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 863.85 | 868.49 | 868.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:00:00 | 863.85 | 868.49 | 868.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 11:15:00 | 862.45 | 867.28 | 867.64 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 12:15:00 | 871.35 | 867.70 | 867.28 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 861.60 | 866.48 | 866.76 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 876.20 | 867.77 | 866.96 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 861.40 | 865.99 | 866.57 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 11:15:00 | 893.90 | 871.72 | 868.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 14:15:00 | 902.30 | 880.61 | 873.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 888.85 | 891.64 | 883.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-11 14:00:00 | 888.85 | 891.64 | 883.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 15:15:00 | 884.00 | 889.25 | 884.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 09:15:00 | 889.00 | 889.25 | 884.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 12:15:00 | 876.95 | 884.50 | 883.27 | SL hit (close<static) qty=1.00 sl=880.25 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 14:15:00 | 875.05 | 881.79 | 882.20 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 10:15:00 | 898.95 | 884.87 | 883.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 921.00 | 898.11 | 891.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 884.95 | 897.67 | 893.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 884.95 | 897.67 | 893.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 884.95 | 897.67 | 893.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 884.95 | 897.67 | 893.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 887.85 | 895.71 | 892.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 09:15:00 | 896.60 | 891.58 | 891.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 879.00 | 893.58 | 893.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 879.00 | 893.58 | 893.95 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 09:15:00 | 904.05 | 890.13 | 889.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 10:15:00 | 930.50 | 898.21 | 892.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 11:15:00 | 916.35 | 920.25 | 910.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 12:00:00 | 916.35 | 920.25 | 910.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 913.35 | 918.41 | 911.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:45:00 | 913.15 | 918.41 | 911.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 895.80 | 913.89 | 909.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:30:00 | 892.80 | 913.89 | 909.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 895.00 | 910.11 | 908.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:15:00 | 895.45 | 910.11 | 908.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 09:15:00 | 896.50 | 907.39 | 907.42 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 10:15:00 | 916.90 | 906.95 | 905.74 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 12:15:00 | 905.80 | 906.55 | 906.64 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 09:15:00 | 910.05 | 907.04 | 906.81 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 900.90 | 906.12 | 906.65 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 922.35 | 909.36 | 908.08 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 11:15:00 | 909.35 | 917.33 | 917.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 12:15:00 | 905.40 | 914.95 | 916.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 875.75 | 874.78 | 883.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 869.70 | 874.29 | 879.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 869.70 | 874.29 | 879.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:30:00 | 874.70 | 874.29 | 879.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 894.50 | 877.89 | 879.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 12:45:00 | 896.00 | 877.89 | 879.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 13:15:00 | 898.80 | 882.07 | 881.57 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 863.65 | 879.59 | 880.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 815.05 | 860.87 | 870.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 806.85 | 804.77 | 829.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:30:00 | 805.00 | 804.77 | 829.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 817.85 | 804.13 | 817.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 817.85 | 804.13 | 817.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 810.45 | 805.40 | 817.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:30:00 | 816.00 | 805.40 | 817.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 810.50 | 808.74 | 815.22 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 826.10 | 817.41 | 817.16 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 10:15:00 | 814.60 | 816.85 | 816.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 11:15:00 | 803.50 | 814.18 | 815.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 09:15:00 | 807.05 | 803.01 | 808.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 807.05 | 803.01 | 808.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 807.05 | 803.01 | 808.65 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 14:15:00 | 838.60 | 813.23 | 811.57 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 13:15:00 | 809.10 | 816.33 | 817.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 808.80 | 814.82 | 816.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 816.70 | 813.12 | 814.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 11:15:00 | 816.70 | 813.12 | 814.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 816.70 | 813.12 | 814.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:45:00 | 816.60 | 813.12 | 814.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 814.75 | 813.45 | 814.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 14:00:00 | 808.25 | 812.41 | 814.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 15:15:00 | 817.00 | 813.90 | 814.64 | SL hit (close>static) qty=1.00 sl=816.85 alert=retest2 |

### Cycle 72 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 827.25 | 816.57 | 815.79 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 810.95 | 816.69 | 817.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 802.00 | 812.12 | 814.65 | Break + close below crossover candle low |

### Cycle 74 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 847.10 | 815.94 | 815.47 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 803.30 | 816.22 | 816.38 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 822.75 | 815.45 | 814.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 09:15:00 | 831.45 | 822.47 | 820.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 13:15:00 | 824.55 | 824.70 | 822.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 13:15:00 | 824.55 | 824.70 | 822.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 824.55 | 824.70 | 822.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:45:00 | 823.95 | 824.70 | 822.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 834.25 | 826.61 | 823.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:30:00 | 833.15 | 826.61 | 823.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 812.00 | 825.03 | 823.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 812.00 | 825.03 | 823.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 803.20 | 820.67 | 821.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 798.95 | 816.32 | 819.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 819.65 | 813.38 | 816.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 819.65 | 813.38 | 816.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 819.65 | 813.38 | 816.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 819.65 | 813.38 | 816.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 817.00 | 814.11 | 816.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:00:00 | 812.70 | 813.83 | 816.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 14:15:00 | 811.25 | 814.10 | 815.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 772.07 | 792.65 | 801.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 11:15:00 | 770.69 | 787.77 | 798.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 10:15:00 | 731.43 | 762.63 | 780.46 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 78 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 761.00 | 742.15 | 740.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 768.45 | 757.57 | 755.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 10:15:00 | 785.55 | 787.08 | 780.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 11:00:00 | 785.55 | 787.08 | 780.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 781.85 | 785.74 | 782.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:45:00 | 775.45 | 785.74 | 782.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 777.00 | 783.99 | 781.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:30:00 | 782.30 | 782.79 | 781.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 11:15:00 | 775.00 | 780.00 | 780.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 11:15:00 | 775.00 | 780.00 | 780.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 13:15:00 | 769.55 | 776.90 | 778.79 | Break + close below crossover candle low |

### Cycle 80 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 803.10 | 779.20 | 779.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 817.15 | 791.91 | 785.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 09:15:00 | 870.50 | 870.54 | 847.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-04 09:45:00 | 869.60 | 870.54 | 847.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 11:15:00 | 900.30 | 907.93 | 898.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:00:00 | 900.30 | 907.93 | 898.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 898.75 | 906.09 | 898.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:30:00 | 896.40 | 906.09 | 898.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 888.70 | 902.61 | 897.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 888.70 | 902.61 | 897.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 895.50 | 901.19 | 897.62 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 883.70 | 894.65 | 895.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 15:15:00 | 880.75 | 886.57 | 890.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 892.10 | 887.68 | 890.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 892.10 | 887.68 | 890.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 892.10 | 887.68 | 890.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 851.50 | 886.73 | 889.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 12:15:00 | 891.75 | 881.11 | 881.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 12:15:00 | 891.75 | 881.11 | 881.01 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 862.70 | 878.98 | 880.82 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 915.35 | 878.90 | 876.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 10:15:00 | 937.00 | 890.52 | 881.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 11:15:00 | 1005.95 | 1011.61 | 989.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 11:45:00 | 1005.80 | 1011.61 | 989.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 996.45 | 1006.62 | 992.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 996.45 | 1006.62 | 992.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 997.40 | 1004.51 | 994.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 990.10 | 1004.51 | 994.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 999.55 | 1002.92 | 997.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:45:00 | 998.00 | 1002.92 | 997.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1004.85 | 1003.30 | 998.03 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 990.75 | 995.66 | 995.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 15:15:00 | 986.20 | 993.77 | 995.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 10:15:00 | 996.55 | 994.18 | 995.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 10:15:00 | 996.55 | 994.18 | 995.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 996.55 | 994.18 | 995.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:15:00 | 1001.25 | 994.18 | 995.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 1010.00 | 997.35 | 996.37 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 988.40 | 996.19 | 996.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 977.65 | 989.45 | 992.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 1003.30 | 989.33 | 991.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 14:15:00 | 1003.30 | 989.33 | 991.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 1003.30 | 989.33 | 991.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 1003.30 | 989.33 | 991.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 1001.95 | 991.86 | 992.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 987.50 | 991.86 | 992.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 902.90 | 911.06 | 919.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 907.65 | 911.06 | 919.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 951.95 | 916.08 | 916.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 956.55 | 916.08 | 916.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 960.95 | 925.05 | 920.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 974.25 | 944.66 | 931.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 975.50 | 977.54 | 966.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:45:00 | 977.05 | 977.54 | 966.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1376.00 | 1416.60 | 1391.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 1376.00 | 1416.60 | 1391.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1377.00 | 1408.68 | 1390.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:00:00 | 1377.00 | 1408.68 | 1390.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 1373.20 | 1401.58 | 1388.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:15:00 | 1368.45 | 1401.58 | 1388.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1413.15 | 1399.25 | 1391.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 10:30:00 | 1430.00 | 1403.45 | 1393.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 1430.90 | 1403.45 | 1393.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:15:00 | 1433.00 | 1427.86 | 1413.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 09:15:00 | 1367.00 | 1408.49 | 1410.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 1367.00 | 1408.49 | 1410.40 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 1416.80 | 1401.96 | 1400.83 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 1366.00 | 1399.80 | 1400.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 1275.65 | 1374.97 | 1389.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1295.00 | 1228.69 | 1270.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1295.00 | 1228.69 | 1270.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1295.00 | 1228.69 | 1270.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1291.25 | 1228.69 | 1270.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1298.15 | 1242.58 | 1273.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1298.15 | 1242.58 | 1273.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 1356.70 | 1294.15 | 1288.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 13:15:00 | 1442.40 | 1381.39 | 1360.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 1770.00 | 1781.68 | 1711.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 1770.00 | 1781.68 | 1711.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1674.75 | 1751.00 | 1736.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 1674.75 | 1751.00 | 1736.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 1671.70 | 1735.14 | 1730.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:00:00 | 1671.70 | 1735.14 | 1730.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 1667.05 | 1721.52 | 1724.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 1643.80 | 1689.48 | 1707.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 1789.20 | 1702.78 | 1710.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 1789.20 | 1702.78 | 1710.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1789.20 | 1702.78 | 1710.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 1816.25 | 1702.78 | 1710.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1763.60 | 1724.73 | 1719.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 1813.20 | 1759.87 | 1740.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 09:15:00 | 2098.00 | 2110.54 | 2048.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:00:00 | 2098.00 | 2110.54 | 2048.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 2085.75 | 2105.58 | 2051.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 2066.00 | 2105.58 | 2051.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 2606.60 | 2678.28 | 2606.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 2606.60 | 2678.28 | 2606.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 2590.10 | 2660.65 | 2604.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 2590.10 | 2660.65 | 2604.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 2604.00 | 2649.32 | 2604.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:15:00 | 2654.00 | 2628.29 | 2605.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 2560.00 | 2614.63 | 2600.98 | SL hit (close<static) qty=1.00 sl=2585.15 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 2503.95 | 2575.72 | 2584.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 2468.20 | 2530.42 | 2557.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 2584.05 | 2511.39 | 2531.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 2584.05 | 2511.39 | 2531.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 2584.05 | 2511.39 | 2531.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 2584.05 | 2511.39 | 2531.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 2589.40 | 2526.99 | 2536.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:15:00 | 2594.75 | 2526.99 | 2536.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 2594.75 | 2551.38 | 2546.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 2614.10 | 2580.86 | 2563.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 2586.00 | 2590.74 | 2573.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 2586.00 | 2590.74 | 2573.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 2586.00 | 2590.74 | 2573.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 2585.40 | 2590.74 | 2573.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 2590.00 | 2589.43 | 2576.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 2515.85 | 2589.43 | 2576.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2526.15 | 2576.77 | 2572.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2506.00 | 2576.77 | 2572.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 10:15:00 | 2507.85 | 2562.99 | 2566.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 2500.75 | 2541.02 | 2555.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 2521.10 | 2520.73 | 2539.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 2521.10 | 2520.73 | 2539.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2521.10 | 2520.73 | 2539.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 2540.40 | 2520.73 | 2539.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 2563.55 | 2510.37 | 2525.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 2563.55 | 2510.37 | 2525.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 2575.00 | 2523.29 | 2530.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 2517.40 | 2523.29 | 2530.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 2528.50 | 2531.66 | 2533.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:15:00 | 2579.85 | 2531.66 | 2533.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 11:15:00 | 2565.00 | 2538.32 | 2535.99 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 2495.50 | 2534.96 | 2536.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 2447.60 | 2506.26 | 2521.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2523.00 | 2502.21 | 2516.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2523.00 | 2502.21 | 2516.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2523.00 | 2502.21 | 2516.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 2523.00 | 2502.21 | 2516.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2565.60 | 2514.89 | 2520.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 2577.00 | 2514.89 | 2520.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 2568.95 | 2525.70 | 2525.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 12:15:00 | 2577.50 | 2536.06 | 2529.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 11:15:00 | 2494.00 | 2553.82 | 2544.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 2494.00 | 2553.82 | 2544.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 2494.00 | 2553.82 | 2544.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:45:00 | 2511.40 | 2553.82 | 2544.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 2462.10 | 2535.48 | 2537.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 09:15:00 | 2421.55 | 2478.54 | 2506.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 2294.00 | 2254.58 | 2302.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 10:00:00 | 2294.00 | 2254.58 | 2302.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 2318.80 | 2267.42 | 2303.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:15:00 | 2323.35 | 2267.42 | 2303.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 2323.35 | 2278.61 | 2305.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:30:00 | 2323.35 | 2278.61 | 2305.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 2428.00 | 2329.62 | 2321.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 10:15:00 | 2439.50 | 2351.59 | 2332.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 2415.00 | 2424.75 | 2398.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 2415.00 | 2424.75 | 2398.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 2399.35 | 2417.30 | 2401.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 2379.75 | 2417.30 | 2401.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 2376.00 | 2409.04 | 2398.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 2376.00 | 2409.04 | 2398.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 2378.40 | 2402.91 | 2396.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 2369.00 | 2402.91 | 2396.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 2343.45 | 2384.63 | 2389.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 2329.30 | 2360.38 | 2375.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 2128.00 | 2123.64 | 2180.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:45:00 | 2131.05 | 2123.64 | 2180.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 2173.90 | 2142.65 | 2175.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:30:00 | 2215.00 | 2142.65 | 2175.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 2177.00 | 2149.52 | 2175.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 2165.65 | 2149.52 | 2175.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 2156.70 | 2150.96 | 2173.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:30:00 | 2170.00 | 2150.96 | 2173.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 2162.15 | 2153.20 | 2172.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:45:00 | 2164.10 | 2153.20 | 2172.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 2175.95 | 2157.75 | 2173.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:45:00 | 2162.00 | 2157.60 | 2171.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 2130.40 | 2149.74 | 2166.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:15:00 | 2053.90 | 2118.17 | 2146.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 2023.88 | 2055.45 | 2096.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-13 14:15:00 | 1945.80 | 1986.48 | 2023.48 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 104 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 1865.10 | 1782.79 | 1777.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 1891.90 | 1834.09 | 1814.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 09:15:00 | 1905.15 | 1939.10 | 1908.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1905.15 | 1939.10 | 1908.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1905.15 | 1939.10 | 1908.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 1910.95 | 1939.10 | 1908.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1912.00 | 1933.68 | 1909.01 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1834.50 | 1896.89 | 1900.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1789.50 | 1837.83 | 1863.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 1795.95 | 1793.68 | 1814.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 13:45:00 | 1799.95 | 1793.68 | 1814.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1784.30 | 1760.65 | 1771.08 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1788.65 | 1777.09 | 1776.61 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 1757.55 | 1773.08 | 1774.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 1750.00 | 1765.90 | 1771.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1743.55 | 1723.02 | 1732.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1743.55 | 1723.02 | 1732.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1743.55 | 1723.02 | 1732.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:45:00 | 1722.25 | 1720.61 | 1730.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 10:15:00 | 1833.95 | 1740.12 | 1732.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1833.95 | 1740.12 | 1732.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 1844.10 | 1760.92 | 1742.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 1816.00 | 1816.92 | 1782.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 14:15:00 | 1802.45 | 1818.53 | 1796.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1802.45 | 1818.53 | 1796.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 15:00:00 | 1802.45 | 1818.53 | 1796.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1802.00 | 1815.22 | 1797.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 1772.00 | 1815.22 | 1797.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1759.85 | 1804.15 | 1793.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 1759.85 | 1804.15 | 1793.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1764.10 | 1796.14 | 1791.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:15:00 | 1759.90 | 1796.14 | 1791.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 1764.15 | 1784.49 | 1786.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 1757.70 | 1779.13 | 1783.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 15:15:00 | 1710.80 | 1709.63 | 1729.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 1718.80 | 1709.63 | 1729.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1715.75 | 1710.44 | 1726.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 1722.05 | 1710.44 | 1726.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1695.15 | 1708.37 | 1720.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1674.05 | 1707.89 | 1719.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:00:00 | 1691.70 | 1695.22 | 1708.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 14:15:00 | 1728.60 | 1705.66 | 1711.37 | SL hit (close>static) qty=1.00 sl=1722.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 1679.95 | 1628.51 | 1627.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 10:15:00 | 1721.00 | 1665.31 | 1650.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1717.50 | 1737.75 | 1718.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 1717.50 | 1737.75 | 1718.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1717.50 | 1737.75 | 1718.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 1716.90 | 1737.75 | 1718.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1713.55 | 1732.91 | 1717.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 1708.00 | 1732.91 | 1717.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1705.05 | 1727.34 | 1716.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:00:00 | 1705.05 | 1727.34 | 1716.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 1698.55 | 1710.44 | 1710.81 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 1720.00 | 1712.35 | 1711.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 1815.85 | 1733.05 | 1721.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 1766.00 | 1766.26 | 1747.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 1757.65 | 1764.13 | 1749.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1757.65 | 1764.13 | 1749.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 1755.00 | 1764.13 | 1749.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1760.00 | 1767.32 | 1756.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1760.00 | 1767.32 | 1756.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1758.95 | 1765.65 | 1756.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 1751.85 | 1765.65 | 1756.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1755.00 | 1763.52 | 1756.72 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 1741.00 | 1752.32 | 1753.00 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 1768.25 | 1753.35 | 1753.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 1817.20 | 1766.12 | 1759.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 14:15:00 | 1767.00 | 1776.12 | 1766.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 1767.00 | 1776.12 | 1766.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1767.00 | 1776.12 | 1766.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 1769.15 | 1776.12 | 1766.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1755.05 | 1771.90 | 1765.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1861.00 | 1771.90 | 1765.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 1716.15 | 1785.07 | 1781.89 | SL hit (close<static) qty=1.00 sl=1755.05 alert=retest2 |

### Cycle 115 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 1674.50 | 1762.96 | 1772.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 1659.75 | 1742.32 | 1761.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 10:15:00 | 1621.00 | 1620.66 | 1657.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 10:15:00 | 1621.00 | 1620.66 | 1657.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1621.00 | 1620.66 | 1657.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:45:00 | 1643.45 | 1620.66 | 1657.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1545.10 | 1532.93 | 1552.55 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1594.05 | 1559.65 | 1555.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1624.00 | 1578.81 | 1565.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 1592.40 | 1592.77 | 1580.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:30:00 | 1586.00 | 1592.77 | 1580.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1587.00 | 1591.61 | 1580.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 1582.95 | 1591.61 | 1580.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1584.45 | 1590.18 | 1581.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:30:00 | 1586.80 | 1590.18 | 1581.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1620.00 | 1596.15 | 1584.62 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 1564.00 | 1584.67 | 1584.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 1548.30 | 1574.21 | 1579.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1562.35 | 1542.09 | 1556.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 1562.35 | 1542.09 | 1556.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1562.35 | 1542.09 | 1556.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:30:00 | 1555.05 | 1542.09 | 1556.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1551.00 | 1543.87 | 1556.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 1549.00 | 1543.87 | 1556.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 1567.40 | 1549.40 | 1556.77 | SL hit (close>static) qty=1.00 sl=1563.75 alert=retest2 |

### Cycle 118 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 1597.65 | 1567.56 | 1563.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 1607.35 | 1581.81 | 1571.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 1574.45 | 1584.55 | 1577.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 14:15:00 | 1574.45 | 1584.55 | 1577.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1574.45 | 1584.55 | 1577.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 1574.60 | 1584.55 | 1577.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1575.95 | 1582.83 | 1577.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1554.95 | 1582.83 | 1577.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1558.00 | 1577.87 | 1575.60 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1547.90 | 1571.87 | 1573.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1537.30 | 1564.96 | 1569.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1432.75 | 1416.00 | 1448.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1432.75 | 1416.00 | 1448.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1432.75 | 1416.00 | 1448.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 1371.10 | 1412.43 | 1430.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 14:45:00 | 1374.70 | 1385.26 | 1395.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 1434.00 | 1399.72 | 1399.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1434.00 | 1399.72 | 1399.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 1442.50 | 1417.75 | 1408.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 1675.05 | 1688.54 | 1635.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1687.55 | 1680.14 | 1655.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1687.55 | 1680.14 | 1655.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 13:15:00 | 1724.30 | 1675.15 | 1664.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1771.90 | 1688.94 | 1674.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 1721.65 | 1754.55 | 1758.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 1721.65 | 1754.55 | 1758.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 1715.45 | 1736.90 | 1748.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 1738.85 | 1735.47 | 1745.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 1738.85 | 1735.47 | 1745.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1766.00 | 1741.57 | 1747.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:45:00 | 1736.65 | 1743.06 | 1746.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 15:15:00 | 1731.00 | 1743.06 | 1746.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 1789.00 | 1750.32 | 1748.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 1789.00 | 1750.32 | 1748.70 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1734.00 | 1751.15 | 1751.72 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 1761.65 | 1739.07 | 1738.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 12:15:00 | 1776.85 | 1748.80 | 1743.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 15:15:00 | 1745.00 | 1748.90 | 1744.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 09:15:00 | 1732.95 | 1748.90 | 1744.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1720.55 | 1743.23 | 1742.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 1720.55 | 1743.23 | 1742.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 1714.00 | 1737.38 | 1740.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 1704.70 | 1727.10 | 1734.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1596.95 | 1573.65 | 1601.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 1596.95 | 1573.65 | 1601.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 1572.80 | 1568.18 | 1582.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 15:15:00 | 1552.00 | 1566.55 | 1577.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 13:15:00 | 1681.00 | 1582.75 | 1578.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 1681.00 | 1582.75 | 1578.35 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1604.05 | 1643.11 | 1643.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1572.35 | 1628.96 | 1636.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1586.00 | 1574.28 | 1598.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 1586.00 | 1574.28 | 1598.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1594.65 | 1578.36 | 1598.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 14:30:00 | 1574.05 | 1584.42 | 1596.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1495.35 | 1529.98 | 1550.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 10:15:00 | 1416.64 | 1469.72 | 1504.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 1446.00 | 1440.19 | 1439.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 1459.35 | 1444.02 | 1441.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1560.80 | 1564.15 | 1528.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 1560.80 | 1564.15 | 1528.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1545.00 | 1557.83 | 1541.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1496.20 | 1557.83 | 1541.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1499.65 | 1546.19 | 1537.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 1487.50 | 1546.19 | 1537.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1485.05 | 1533.96 | 1532.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 1485.05 | 1533.96 | 1532.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 1470.80 | 1521.33 | 1526.96 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 1554.95 | 1529.78 | 1528.18 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 1527.80 | 1532.01 | 1532.47 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 14:15:00 | 1534.00 | 1532.40 | 1532.35 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 1531.70 | 1532.26 | 1532.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1461.30 | 1518.07 | 1525.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1518.90 | 1463.98 | 1485.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 1518.90 | 1463.98 | 1485.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1518.90 | 1463.98 | 1485.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 1518.90 | 1463.98 | 1485.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1539.10 | 1479.00 | 1489.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 1544.00 | 1479.00 | 1489.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1541.90 | 1503.41 | 1499.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1583.85 | 1539.36 | 1521.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1545.00 | 1556.96 | 1537.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 1553.40 | 1556.96 | 1537.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1516.55 | 1548.88 | 1535.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 1516.55 | 1548.88 | 1535.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 1520.00 | 1543.10 | 1534.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 1534.00 | 1543.10 | 1534.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-01 09:15:00 | 1687.40 | 1617.55 | 1583.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1502.45 | 1568.89 | 1577.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1477.00 | 1495.92 | 1509.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 10:15:00 | 1514.75 | 1499.69 | 1509.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 10:15:00 | 1514.75 | 1499.69 | 1509.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1514.75 | 1499.69 | 1509.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1514.75 | 1499.69 | 1509.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1539.05 | 1507.56 | 1512.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 1539.05 | 1507.56 | 1512.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1518.05 | 1509.66 | 1512.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:45:00 | 1523.80 | 1509.66 | 1512.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 1495.30 | 1506.79 | 1511.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 15:15:00 | 1484.65 | 1505.84 | 1510.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1410.42 | 1448.64 | 1475.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 1428.90 | 1424.32 | 1446.78 | SL hit (close>ema200) qty=0.50 sl=1424.32 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1355.00 | 1312.30 | 1311.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 1359.30 | 1335.21 | 1323.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 1341.00 | 1343.51 | 1332.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 09:15:00 | 1391.65 | 1343.51 | 1332.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1319.95 | 1338.80 | 1331.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1319.95 | 1338.80 | 1331.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1328.80 | 1336.80 | 1331.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 1345.40 | 1337.17 | 1332.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 1341.50 | 1337.17 | 1332.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 1316.00 | 1329.19 | 1329.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 1316.00 | 1329.19 | 1329.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1300.85 | 1323.52 | 1327.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 10:15:00 | 1337.00 | 1326.22 | 1327.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 10:15:00 | 1337.00 | 1326.22 | 1327.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1337.00 | 1326.22 | 1327.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:30:00 | 1346.55 | 1326.22 | 1327.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 11:15:00 | 1358.00 | 1332.57 | 1330.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 10:15:00 | 1382.45 | 1349.19 | 1340.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 11:15:00 | 1343.65 | 1348.09 | 1340.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 12:00:00 | 1343.65 | 1348.09 | 1340.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 1339.50 | 1346.37 | 1340.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:45:00 | 1336.15 | 1346.37 | 1340.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 1335.75 | 1344.24 | 1340.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 1335.75 | 1344.24 | 1340.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 1329.60 | 1341.32 | 1339.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:30:00 | 1332.60 | 1341.32 | 1339.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1327.00 | 1338.45 | 1338.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 1326.25 | 1338.45 | 1338.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1306.95 | 1332.15 | 1335.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 1294.15 | 1314.48 | 1325.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 1315.70 | 1242.18 | 1257.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 1315.70 | 1242.18 | 1257.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1315.70 | 1242.18 | 1257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1315.70 | 1242.18 | 1257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1274.75 | 1248.69 | 1258.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 1264.10 | 1249.82 | 1258.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 1279.60 | 1264.28 | 1262.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1279.60 | 1264.28 | 1262.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 1291.50 | 1278.19 | 1270.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 1293.00 | 1298.02 | 1287.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 15:15:00 | 1293.00 | 1298.02 | 1287.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 1293.00 | 1298.02 | 1287.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 1322.80 | 1298.02 | 1287.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 11:15:00 | 1315.25 | 1338.25 | 1338.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 1315.25 | 1338.25 | 1338.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 1306.45 | 1331.89 | 1336.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1342.60 | 1311.54 | 1317.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 1342.60 | 1311.54 | 1317.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1342.60 | 1311.54 | 1317.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 1342.60 | 1311.54 | 1317.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1353.05 | 1319.84 | 1320.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 1345.90 | 1319.84 | 1320.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 11:15:00 | 1332.85 | 1322.44 | 1321.94 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 1315.90 | 1321.14 | 1321.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 1302.35 | 1317.38 | 1319.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1321.00 | 1316.61 | 1318.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1321.00 | 1316.61 | 1318.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1321.00 | 1316.61 | 1318.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1325.00 | 1316.61 | 1318.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1321.45 | 1317.57 | 1319.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 1346.00 | 1317.57 | 1319.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1343.25 | 1322.71 | 1321.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1356.80 | 1335.72 | 1328.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 15:15:00 | 1697.95 | 1698.60 | 1644.49 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 09:15:00 | 1728.80 | 1698.60 | 1644.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 15:15:00 | 1705.00 | 1709.70 | 1675.05 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 11:45:00 | 1713.00 | 1714.50 | 1688.40 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1729.00 | 1719.56 | 1701.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 1740.00 | 1719.56 | 1701.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:00:00 | 1734.80 | 1721.25 | 1704.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 12:45:00 | 1744.65 | 1720.32 | 1706.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 1704.75 | 1716.01 | 1706.47 | SL hit (close<ema400) qty=1.00 sl=1706.47 alert=retest1 |

### Cycle 145 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 1693.00 | 1704.95 | 1705.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 1685.00 | 1699.29 | 1702.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 1689.00 | 1688.36 | 1694.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 1689.00 | 1688.36 | 1694.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1689.00 | 1688.36 | 1694.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:30:00 | 1690.70 | 1688.36 | 1694.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 1697.00 | 1690.67 | 1694.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 1651.00 | 1690.67 | 1694.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1666.05 | 1685.75 | 1692.02 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1722.15 | 1700.30 | 1697.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1742.55 | 1710.94 | 1702.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1642.10 | 1708.66 | 1708.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1642.10 | 1708.66 | 1708.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1642.10 | 1708.66 | 1708.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1642.10 | 1708.66 | 1708.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1637.10 | 1694.35 | 1701.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1463.95 | 1614.93 | 1656.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1538.00 | 1534.13 | 1588.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1597.20 | 1534.13 | 1588.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1561.00 | 1539.51 | 1585.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1536.40 | 1541.34 | 1582.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:00:00 | 1548.70 | 1541.34 | 1582.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1532.55 | 1556.29 | 1575.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1640.15 | 1572.06 | 1570.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1640.15 | 1572.06 | 1570.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1707.00 | 1623.81 | 1600.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1707.10 | 1716.03 | 1688.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1707.10 | 1716.03 | 1688.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1713.00 | 1732.45 | 1728.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1713.00 | 1732.45 | 1728.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1722.50 | 1730.46 | 1727.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1708.00 | 1730.46 | 1727.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1727.20 | 1729.81 | 1727.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 1737.60 | 1730.19 | 1727.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 1735.20 | 1730.19 | 1727.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:15:00 | 1761.90 | 1734.71 | 1730.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:30:00 | 1744.40 | 1739.20 | 1734.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1736.10 | 1738.58 | 1734.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 1734.90 | 1738.58 | 1734.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1737.00 | 1738.26 | 1734.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:30:00 | 1734.00 | 1738.26 | 1734.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1730.00 | 1736.61 | 1734.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 1730.00 | 1736.61 | 1734.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1725.40 | 1734.37 | 1733.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 1694.00 | 1734.37 | 1733.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1645.30 | 1716.55 | 1725.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1645.30 | 1716.55 | 1725.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1612.50 | 1695.74 | 1715.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1748.40 | 1664.98 | 1685.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 1748.40 | 1664.98 | 1685.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1748.40 | 1664.98 | 1685.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 1748.40 | 1664.98 | 1685.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1759.80 | 1683.94 | 1691.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 1759.80 | 1683.94 | 1691.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 1769.80 | 1701.11 | 1699.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1827.20 | 1751.32 | 1726.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1871.60 | 1895.07 | 1827.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 10:00:00 | 1871.60 | 1895.07 | 1827.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1885.00 | 1904.08 | 1883.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 1887.90 | 1904.08 | 1883.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1880.00 | 1899.26 | 1883.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 1875.90 | 1899.26 | 1883.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1884.60 | 1896.33 | 1883.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:00:00 | 1911.10 | 1897.42 | 1886.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 1862.90 | 1889.96 | 1891.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1862.90 | 1889.96 | 1891.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1836.90 | 1879.35 | 1886.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 1837.50 | 1819.91 | 1843.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 1837.50 | 1819.91 | 1843.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1837.50 | 1819.91 | 1843.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 1789.80 | 1816.32 | 1834.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:45:00 | 1774.50 | 1808.94 | 1829.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 10:00:00 | 1793.90 | 1797.91 | 1820.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 10:30:00 | 1788.90 | 1796.81 | 1817.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1815.50 | 1803.09 | 1817.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:30:00 | 1815.00 | 1803.09 | 1817.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1802.50 | 1802.97 | 1815.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 14:15:00 | 1788.00 | 1802.97 | 1815.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 15:15:00 | 1823.70 | 1811.29 | 1812.38 | SL hit (close>static) qty=1.00 sl=1823.20 alert=retest2 |

### Cycle 152 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 1876.60 | 1824.35 | 1818.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1904.80 | 1840.44 | 1826.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 2439.70 | 2443.90 | 2327.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:45:00 | 2414.50 | 2443.90 | 2327.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2417.00 | 2474.78 | 2406.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 2587.00 | 2502.65 | 2461.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-23 09:15:00 | 2845.70 | 2687.90 | 2589.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 3280.00 | 3292.43 | 3292.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 10:15:00 | 3231.30 | 3280.20 | 3287.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 3266.00 | 3264.93 | 3276.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 14:15:00 | 3266.00 | 3264.93 | 3276.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 3266.00 | 3264.93 | 3276.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 3266.00 | 3264.93 | 3276.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 3272.00 | 3266.35 | 3276.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 3262.90 | 3266.35 | 3276.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 3213.50 | 3255.78 | 3270.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:45:00 | 3189.70 | 3232.45 | 3256.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 3176.80 | 3229.36 | 3246.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 3030.21 | 3094.76 | 3148.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 12:15:00 | 3017.96 | 3080.51 | 3137.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 3037.70 | 3034.08 | 3093.80 | SL hit (close>ema200) qty=0.50 sl=3034.08 alert=retest2 |

### Cycle 154 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 3143.10 | 3092.11 | 3087.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 3160.00 | 3105.69 | 3094.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 3187.20 | 3210.78 | 3177.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 3187.20 | 3210.78 | 3177.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 3187.20 | 3210.78 | 3177.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 3170.60 | 3210.78 | 3177.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 3152.20 | 3199.06 | 3174.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 3154.80 | 3199.06 | 3174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 3132.20 | 3185.69 | 3171.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 3132.20 | 3185.69 | 3171.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 3160.00 | 3180.55 | 3170.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 3144.30 | 3180.55 | 3170.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 3131.00 | 3162.27 | 3163.06 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 3192.00 | 3168.21 | 3165.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 3254.20 | 3185.44 | 3173.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 3296.20 | 3395.00 | 3324.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 3296.20 | 3395.00 | 3324.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3296.20 | 3395.00 | 3324.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 3288.10 | 3395.00 | 3324.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3282.10 | 3372.42 | 3320.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 3282.10 | 3372.42 | 3320.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 3173.80 | 3274.42 | 3285.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 3070.50 | 3220.13 | 3258.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 2995.00 | 2992.31 | 3065.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 3017.20 | 2992.31 | 3065.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 3038.10 | 3001.47 | 3063.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 3025.00 | 3001.47 | 3063.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 3025.00 | 3012.02 | 3045.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 3025.00 | 3012.02 | 3045.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 3039.10 | 3017.43 | 3045.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 3006.30 | 3017.43 | 3045.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 3010.00 | 3015.95 | 3041.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:00:00 | 2992.70 | 3012.51 | 3026.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 2991.10 | 3007.36 | 3022.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:45:00 | 2978.00 | 2982.06 | 2983.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:15:00 | 2843.06 | 2900.79 | 2932.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:15:00 | 2841.54 | 2900.79 | 2932.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:15:00 | 2829.10 | 2900.79 | 2932.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 2966.00 | 2875.79 | 2897.79 | SL hit (close>ema200) qty=0.50 sl=2875.79 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2977.10 | 2911.60 | 2911.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 2986.00 | 2926.48 | 2918.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 2892.00 | 2940.13 | 2929.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 2892.00 | 2940.13 | 2929.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2892.00 | 2940.13 | 2929.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 2892.00 | 2940.13 | 2929.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2898.00 | 2931.70 | 2926.48 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 2885.90 | 2917.25 | 2920.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 2826.00 | 2889.36 | 2905.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 2719.00 | 2717.06 | 2773.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:30:00 | 2708.30 | 2717.06 | 2773.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 2653.20 | 2628.11 | 2660.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 2653.20 | 2628.11 | 2660.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2602.10 | 2631.03 | 2652.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 2600.50 | 2631.03 | 2652.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 2583.30 | 2620.82 | 2637.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:30:00 | 2597.50 | 2612.79 | 2629.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 2596.80 | 2598.43 | 2614.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2613.90 | 2601.52 | 2614.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:15:00 | 2615.50 | 2601.52 | 2614.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2621.50 | 2605.52 | 2615.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 2621.50 | 2605.52 | 2615.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 2622.40 | 2608.89 | 2615.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 2622.40 | 2608.89 | 2615.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 2631.20 | 2620.93 | 2620.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 2631.20 | 2620.93 | 2620.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 2645.00 | 2628.48 | 2624.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 2631.00 | 2634.92 | 2629.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:00:00 | 2631.00 | 2634.92 | 2629.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 2630.20 | 2633.98 | 2629.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 2630.20 | 2633.98 | 2629.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 2623.20 | 2631.82 | 2628.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 2623.20 | 2631.82 | 2628.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 2626.80 | 2630.82 | 2628.77 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 2615.60 | 2626.38 | 2627.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2605.90 | 2619.82 | 2623.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 2489.50 | 2484.27 | 2522.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:45:00 | 2488.00 | 2484.27 | 2522.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 2520.00 | 2495.57 | 2518.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 2534.40 | 2495.57 | 2518.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 2543.00 | 2505.05 | 2520.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 2543.00 | 2505.05 | 2520.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 2559.50 | 2515.94 | 2524.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 2561.20 | 2515.94 | 2524.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 2556.30 | 2534.41 | 2531.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 2613.50 | 2553.70 | 2540.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2564.90 | 2593.44 | 2573.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 2564.90 | 2593.44 | 2573.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 2564.90 | 2593.44 | 2573.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 2564.90 | 2593.44 | 2573.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 2560.70 | 2586.89 | 2572.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 2560.70 | 2586.89 | 2572.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 2563.30 | 2582.17 | 2571.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 2553.20 | 2582.17 | 2571.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 2647.70 | 2594.73 | 2581.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 2662.00 | 2619.08 | 2601.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:00:00 | 2661.80 | 2627.62 | 2607.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 2597.90 | 2617.48 | 2618.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 2597.90 | 2617.48 | 2618.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 2585.30 | 2611.04 | 2615.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 2558.00 | 2553.21 | 2577.31 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2493.20 | 2553.21 | 2577.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 2515.00 | 2516.73 | 2547.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 2515.00 | 2516.73 | 2547.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 2526.00 | 2518.58 | 2545.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:45:00 | 2542.00 | 2518.58 | 2545.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2577.10 | 2527.38 | 2544.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 2577.10 | 2527.38 | 2544.49 | SL hit (close>ema400) qty=1.00 sl=2544.49 alert=retest1 |

### Cycle 164 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 2592.00 | 2561.20 | 2558.19 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 2533.40 | 2553.48 | 2556.04 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 2571.40 | 2547.35 | 2545.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 2633.40 | 2564.56 | 2553.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2545.20 | 2583.48 | 2566.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 2545.20 | 2583.48 | 2566.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2545.20 | 2583.48 | 2566.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 2545.20 | 2583.48 | 2566.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2558.00 | 2578.38 | 2566.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 2571.00 | 2576.49 | 2567.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 2589.00 | 2614.19 | 2615.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 2589.00 | 2614.19 | 2615.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 2566.70 | 2595.74 | 2606.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 2585.00 | 2584.92 | 2597.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 2585.00 | 2584.92 | 2597.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 2585.00 | 2584.92 | 2597.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 2585.00 | 2584.92 | 2597.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 2630.50 | 2594.03 | 2600.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 2630.50 | 2594.03 | 2600.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 2611.00 | 2597.43 | 2601.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 2596.80 | 2597.43 | 2601.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 2610.00 | 2590.47 | 2596.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 2466.96 | 2520.52 | 2549.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 2479.50 | 2520.52 | 2549.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-28 14:15:00 | 2337.12 | 2390.58 | 2444.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 2467.30 | 2426.82 | 2422.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 2484.30 | 2438.32 | 2428.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 2508.20 | 2519.46 | 2491.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:45:00 | 2506.70 | 2519.46 | 2491.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2503.90 | 2516.35 | 2492.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 2494.80 | 2516.35 | 2492.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2495.20 | 2512.12 | 2492.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 2496.90 | 2512.12 | 2492.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 2500.90 | 2509.87 | 2493.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:30:00 | 2500.30 | 2509.87 | 2493.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2454.10 | 2498.74 | 2491.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 2454.10 | 2498.74 | 2491.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2437.30 | 2486.45 | 2486.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 2440.90 | 2486.45 | 2486.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 2427.00 | 2474.56 | 2480.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 2418.50 | 2455.63 | 2470.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 2423.00 | 2415.91 | 2434.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 2397.90 | 2415.91 | 2434.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2377.10 | 2384.12 | 2405.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 2389.00 | 2384.12 | 2405.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2385.60 | 2368.95 | 2384.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 2385.60 | 2368.95 | 2384.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2406.70 | 2376.50 | 2386.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 2406.70 | 2376.50 | 2386.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 2427.30 | 2386.66 | 2390.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 2427.30 | 2386.66 | 2390.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 2421.00 | 2393.53 | 2393.22 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 2362.80 | 2392.33 | 2394.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 2356.90 | 2385.25 | 2391.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 2446.60 | 2389.80 | 2391.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 2446.60 | 2389.80 | 2391.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2446.60 | 2389.80 | 2391.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2446.60 | 2389.80 | 2391.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 2462.80 | 2404.40 | 2397.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 2532.10 | 2429.94 | 2409.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 2509.00 | 2511.25 | 2467.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 11:00:00 | 2509.00 | 2511.25 | 2467.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2484.50 | 2498.15 | 2479.65 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 2446.10 | 2471.14 | 2472.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 15:15:00 | 2435.00 | 2463.91 | 2468.64 | Break + close below crossover candle low |

### Cycle 174 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 2561.00 | 2483.33 | 2477.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 2673.70 | 2590.33 | 2545.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 12:15:00 | 2611.30 | 2622.21 | 2593.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 12:45:00 | 2601.30 | 2622.21 | 2593.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2646.00 | 2662.63 | 2640.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 2626.10 | 2662.63 | 2640.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 2641.30 | 2656.84 | 2641.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 2634.00 | 2656.84 | 2641.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 2640.00 | 2653.47 | 2641.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:45:00 | 2639.20 | 2653.47 | 2641.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 2642.70 | 2651.32 | 2641.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:15:00 | 2647.40 | 2651.32 | 2641.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 2680.00 | 2657.06 | 2645.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 2708.00 | 2661.44 | 2648.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 2643.70 | 2692.74 | 2696.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 2643.70 | 2692.74 | 2696.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 2625.00 | 2670.78 | 2685.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 2540.50 | 2528.38 | 2561.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 2592.90 | 2528.38 | 2561.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 2615.10 | 2545.73 | 2566.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 2629.90 | 2545.73 | 2566.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 2657.00 | 2567.98 | 2574.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 2698.40 | 2567.98 | 2574.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 2640.00 | 2582.39 | 2580.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2688.00 | 2633.53 | 2609.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 2720.70 | 2724.84 | 2699.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:15:00 | 2716.90 | 2724.84 | 2699.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2712.30 | 2720.35 | 2703.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:45:00 | 2711.10 | 2720.35 | 2703.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 2708.00 | 2717.88 | 2704.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2724.60 | 2717.88 | 2704.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 2672.60 | 2708.11 | 2701.94 | SL hit (close<static) qty=1.00 sl=2693.30 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 2678.70 | 2694.86 | 2696.69 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 2715.50 | 2698.84 | 2697.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 15:15:00 | 2735.00 | 2711.03 | 2704.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 11:15:00 | 2705.40 | 2714.49 | 2708.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 11:15:00 | 2705.40 | 2714.49 | 2708.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 2705.40 | 2714.49 | 2708.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 2705.40 | 2714.49 | 2708.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2684.00 | 2708.39 | 2705.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 2687.00 | 2708.39 | 2705.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 2677.60 | 2702.23 | 2703.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 2661.00 | 2693.98 | 2699.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 2592.90 | 2578.97 | 2606.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 2592.90 | 2578.97 | 2606.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 2592.90 | 2578.97 | 2606.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 2583.00 | 2578.97 | 2606.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 2599.00 | 2582.98 | 2605.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 2610.00 | 2582.98 | 2605.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2623.60 | 2591.10 | 2607.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 2623.60 | 2591.10 | 2607.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2614.90 | 2595.86 | 2607.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 2600.20 | 2605.96 | 2610.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 2642.10 | 2616.07 | 2612.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 2642.10 | 2616.07 | 2612.82 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 2600.60 | 2612.47 | 2612.61 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 2629.80 | 2614.95 | 2613.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 2644.90 | 2629.74 | 2621.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 2621.00 | 2630.06 | 2623.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 2621.00 | 2630.06 | 2623.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 2621.00 | 2630.06 | 2623.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 2618.70 | 2630.06 | 2623.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 2645.00 | 2633.05 | 2625.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 2645.00 | 2633.05 | 2625.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2626.90 | 2632.81 | 2627.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 2626.90 | 2632.81 | 2627.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2609.80 | 2628.21 | 2625.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2609.80 | 2628.21 | 2625.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 2606.00 | 2623.77 | 2623.94 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 2648.30 | 2628.67 | 2626.16 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 2610.90 | 2623.38 | 2624.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 2568.90 | 2610.77 | 2618.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 2585.80 | 2583.30 | 2597.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 2585.80 | 2583.30 | 2597.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2585.80 | 2583.30 | 2597.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 2605.10 | 2583.30 | 2597.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 2579.00 | 2564.24 | 2573.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 2572.10 | 2564.24 | 2573.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2553.90 | 2562.17 | 2571.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 2548.50 | 2562.17 | 2571.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 2550.10 | 2556.97 | 2563.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:15:00 | 2550.10 | 2557.18 | 2560.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 2580.00 | 2564.28 | 2562.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 2580.00 | 2564.28 | 2562.98 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 2559.60 | 2562.27 | 2562.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 2553.80 | 2560.58 | 2561.62 | Break + close below crossover candle low |

### Cycle 188 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 2582.80 | 2563.22 | 2562.43 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 2545.90 | 2566.68 | 2567.52 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 2572.00 | 2567.93 | 2567.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 2682.40 | 2590.82 | 2578.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 2794.00 | 2808.83 | 2769.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:00:00 | 2794.00 | 2808.83 | 2769.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 2780.30 | 2803.12 | 2770.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 2778.00 | 2803.12 | 2770.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 2758.00 | 2794.10 | 2769.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:00:00 | 2758.00 | 2794.10 | 2769.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 2757.80 | 2786.84 | 2768.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 2757.80 | 2786.84 | 2768.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 2749.00 | 2779.27 | 2766.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 2749.00 | 2779.27 | 2766.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2861.00 | 2887.79 | 2863.66 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 2812.10 | 2852.69 | 2857.59 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2933.40 | 2852.83 | 2852.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 2953.00 | 2883.12 | 2866.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 2862.50 | 2901.00 | 2884.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 2862.50 | 2901.00 | 2884.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2862.50 | 2901.00 | 2884.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 2862.50 | 2901.00 | 2884.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2824.40 | 2885.68 | 2879.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 2824.40 | 2885.68 | 2879.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 2849.00 | 2873.42 | 2874.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 13:15:00 | 2841.60 | 2867.05 | 2871.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 2774.00 | 2756.26 | 2795.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 2774.00 | 2756.26 | 2795.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 2774.00 | 2756.26 | 2795.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 2808.00 | 2756.26 | 2795.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2762.90 | 2743.22 | 2766.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 2762.90 | 2743.22 | 2766.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 2756.40 | 2745.86 | 2765.15 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 2775.20 | 2768.90 | 2768.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 09:15:00 | 2831.80 | 2786.19 | 2776.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 12:15:00 | 2791.60 | 2796.05 | 2784.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 2791.60 | 2796.05 | 2784.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 2782.00 | 2793.24 | 2784.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 2782.00 | 2793.24 | 2784.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2792.60 | 2793.11 | 2785.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 2799.00 | 2793.11 | 2785.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 2749.80 | 2785.45 | 2784.39 | SL hit (close<static) qty=1.00 sl=2781.80 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 2749.20 | 2778.20 | 2781.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 2737.90 | 2767.11 | 2775.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 2416.10 | 2403.96 | 2458.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:00:00 | 2416.10 | 2403.96 | 2458.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 2452.00 | 2422.61 | 2447.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 2452.00 | 2422.61 | 2447.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 2448.00 | 2427.69 | 2447.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 2420.40 | 2427.69 | 2447.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 09:15:00 | 2299.38 | 2341.12 | 2363.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-18 10:15:00 | 2178.36 | 2230.57 | 2272.56 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 2299.00 | 2264.29 | 2261.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2354.70 | 2282.37 | 2269.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 2419.60 | 2420.78 | 2379.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:45:00 | 2422.50 | 2420.78 | 2379.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2473.20 | 2497.81 | 2479.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 2473.20 | 2497.81 | 2479.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2489.00 | 2496.05 | 2480.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2457.60 | 2496.05 | 2480.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2444.10 | 2485.66 | 2477.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 2444.10 | 2485.66 | 2477.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 2412.30 | 2470.99 | 2471.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 2404.20 | 2457.63 | 2465.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 2441.10 | 2437.04 | 2449.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:45:00 | 2436.70 | 2437.04 | 2449.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 2448.40 | 2438.78 | 2447.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 2448.40 | 2438.78 | 2447.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 2445.20 | 2440.06 | 2447.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 2449.70 | 2440.06 | 2447.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 2440.90 | 2440.23 | 2446.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 2440.90 | 2440.23 | 2446.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 2452.00 | 2442.59 | 2447.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 2427.60 | 2442.59 | 2447.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2417.20 | 2437.51 | 2444.64 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 2470.30 | 2448.31 | 2445.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 2491.60 | 2455.38 | 2449.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 2493.00 | 2504.60 | 2486.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 2493.00 | 2504.60 | 2486.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 2511.00 | 2505.88 | 2489.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 2515.60 | 2495.62 | 2490.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 2477.50 | 2494.97 | 2493.23 | SL hit (close<static) qty=1.00 sl=2486.50 alert=retest2 |

### Cycle 199 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 2462.00 | 2488.38 | 2490.39 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 2506.00 | 2493.37 | 2491.74 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 2474.30 | 2489.56 | 2490.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 2455.90 | 2482.83 | 2487.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2474.30 | 2464.41 | 2473.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 2474.30 | 2464.41 | 2473.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 2474.30 | 2464.41 | 2473.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 2474.00 | 2464.41 | 2473.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 2467.40 | 2465.01 | 2473.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 2476.40 | 2465.01 | 2473.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 2472.60 | 2466.52 | 2473.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 2472.60 | 2466.52 | 2473.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 2488.30 | 2470.88 | 2474.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 2472.00 | 2470.88 | 2474.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2470.40 | 2470.78 | 2474.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 2455.00 | 2465.22 | 2471.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2454.40 | 2451.45 | 2460.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 2454.60 | 2455.29 | 2460.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2332.25 | 2374.73 | 2401.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2331.68 | 2374.73 | 2401.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2331.87 | 2374.73 | 2401.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 2297.50 | 2295.70 | 2334.17 | SL hit (close>ema200) qty=0.50 sl=2295.70 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 12:15:00 | 2327.90 | 2296.38 | 2294.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 2384.30 | 2318.34 | 2304.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 2641.50 | 2729.02 | 2650.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 2641.50 | 2729.02 | 2650.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 2641.50 | 2729.02 | 2650.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 2641.50 | 2729.02 | 2650.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 2622.10 | 2707.64 | 2648.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 2574.00 | 2707.64 | 2648.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 2559.10 | 2607.09 | 2612.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 2450.40 | 2517.07 | 2540.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2453.00 | 2422.97 | 2452.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2453.00 | 2422.97 | 2452.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2453.00 | 2422.97 | 2452.90 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 2514.00 | 2472.70 | 2468.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 2529.30 | 2491.11 | 2477.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 2505.10 | 2510.54 | 2495.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 2479.00 | 2510.54 | 2495.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2473.70 | 2503.18 | 2493.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 2473.10 | 2503.18 | 2493.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 2473.20 | 2497.18 | 2491.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 2473.20 | 2497.18 | 2491.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2472.30 | 2487.68 | 2488.13 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2495.80 | 2486.31 | 2485.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 15:15:00 | 2503.00 | 2489.65 | 2487.41 | Break + close above crossover candle high |

### Cycle 207 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 2439.70 | 2479.66 | 2483.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 2435.20 | 2455.41 | 2468.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 2438.00 | 2433.89 | 2447.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:15:00 | 2462.10 | 2433.89 | 2447.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 2458.50 | 2438.81 | 2448.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 2429.60 | 2436.69 | 2444.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 2425.30 | 2433.53 | 2440.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 2429.80 | 2431.50 | 2438.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:45:00 | 2430.60 | 2434.30 | 2439.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 2501.30 | 2447.70 | 2444.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 13:15:00 | 2501.30 | 2447.70 | 2444.84 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 2449.80 | 2466.63 | 2466.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 2415.60 | 2444.85 | 2454.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 2435.00 | 2424.02 | 2435.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 2435.00 | 2424.02 | 2435.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 2435.00 | 2424.02 | 2435.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 2425.50 | 2426.53 | 2435.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 2420.00 | 2423.10 | 2432.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 2462.40 | 2431.05 | 2433.19 | SL hit (close>static) qty=1.00 sl=2448.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2465.70 | 2437.98 | 2436.14 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2430.10 | 2443.23 | 2443.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 10:15:00 | 2421.20 | 2436.82 | 2440.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 2369.70 | 2341.32 | 2369.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 2369.70 | 2341.32 | 2369.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 2369.70 | 2341.32 | 2369.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 2369.70 | 2341.32 | 2369.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 2356.20 | 2344.30 | 2368.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:15:00 | 2377.60 | 2344.30 | 2368.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 2350.80 | 2345.60 | 2366.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:30:00 | 2359.30 | 2345.60 | 2366.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 2482.00 | 2372.88 | 2377.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 2503.70 | 2372.88 | 2377.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 2430.00 | 2384.30 | 2381.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2518.30 | 2419.34 | 2399.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2466.70 | 2495.36 | 2458.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2466.70 | 2495.36 | 2458.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2466.70 | 2495.36 | 2458.40 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 2442.80 | 2474.21 | 2477.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 2359.60 | 2419.31 | 2442.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2299.00 | 2296.57 | 2342.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 2324.40 | 2296.57 | 2342.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2333.10 | 2305.39 | 2338.65 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2381.90 | 2350.48 | 2346.46 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 2332.00 | 2354.91 | 2355.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2304.60 | 2344.85 | 2350.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2333.60 | 2329.31 | 2341.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2333.60 | 2329.31 | 2341.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2333.60 | 2329.31 | 2341.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 2302.70 | 2325.88 | 2337.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 2310.40 | 2320.14 | 2332.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2187.56 | 2282.46 | 2311.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 2194.88 | 2282.46 | 2311.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 2163.00 | 2160.57 | 2213.07 | SL hit (close>ema200) qty=0.50 sl=2160.57 alert=retest2 |

### Cycle 216 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2316.00 | 2126.11 | 2119.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 2355.00 | 2171.89 | 2140.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2218.50 | 2265.63 | 2207.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2218.50 | 2265.63 | 2207.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2218.50 | 2265.63 | 2207.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 2218.50 | 2265.63 | 2207.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2203.60 | 2253.22 | 2207.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 2218.80 | 2248.48 | 2209.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 2238.30 | 2241.18 | 2209.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 2229.00 | 2243.56 | 2221.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 2440.68 | 2317.60 | 2280.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 2805.00 | 2929.10 | 2945.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 2791.00 | 2865.06 | 2909.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 09:15:00 | 2873.80 | 2834.44 | 2877.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 2873.80 | 2834.44 | 2877.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 2873.80 | 2834.44 | 2877.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 2873.80 | 2834.44 | 2877.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 2864.70 | 2840.49 | 2876.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 13:15:00 | 2855.00 | 2854.66 | 2876.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 2910.00 | 2865.19 | 2877.65 | SL hit (close>static) qty=1.00 sl=2893.90 alert=retest2 |

### Cycle 218 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 3014.60 | 2903.68 | 2893.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 3074.20 | 3008.20 | 2961.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 3061.00 | 3065.87 | 3020.40 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 09:30:00 | 516.60 | 2023-05-18 14:15:00 | 514.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-05-16 09:15:00 | 511.00 | 2023-05-18 14:15:00 | 514.60 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2023-05-24 14:15:00 | 482.50 | 2023-05-25 09:15:00 | 458.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-24 15:15:00 | 480.00 | 2023-05-25 09:15:00 | 456.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-24 14:15:00 | 482.50 | 2023-05-26 10:15:00 | 465.35 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2023-05-24 15:15:00 | 480.00 | 2023-05-26 10:15:00 | 465.35 | STOP_HIT | 0.50 | 3.05% |
| BUY | retest2 | 2023-06-09 11:15:00 | 509.30 | 2023-06-12 14:15:00 | 560.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-04 10:30:00 | 603.50 | 2023-08-14 10:15:00 | 630.70 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2023-08-04 13:15:00 | 602.85 | 2023-08-14 10:15:00 | 630.70 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest2 | 2023-08-21 11:15:00 | 768.85 | 2023-08-28 09:15:00 | 843.70 | TARGET_HIT | 1.00 | 9.74% |
| BUY | retest2 | 2023-08-21 15:15:00 | 770.20 | 2023-09-07 09:15:00 | 845.74 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2023-08-22 09:45:00 | 767.00 | 2023-09-07 09:15:00 | 847.22 | TARGET_HIT | 1.00 | 10.46% |
| BUY | retest2 | 2023-08-23 09:15:00 | 774.00 | 2023-09-07 09:15:00 | 851.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-25 09:15:00 | 796.00 | 2023-09-07 09:15:00 | 875.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 09:15:00 | 832.35 | 2023-09-07 09:15:00 | 915.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-29 14:00:00 | 790.20 | 2023-09-07 09:15:00 | 869.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-29 15:00:00 | 794.90 | 2023-09-07 09:15:00 | 874.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-01 09:30:00 | 830.50 | 2023-09-07 09:15:00 | 913.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-05 09:15:00 | 834.25 | 2023-09-07 09:15:00 | 917.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 09:15:00 | 854.10 | 2023-09-07 11:15:00 | 939.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-29 09:15:00 | 849.25 | 2023-10-03 14:15:00 | 830.65 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2023-10-03 09:30:00 | 833.95 | 2023-10-03 14:15:00 | 830.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-10-03 11:00:00 | 832.95 | 2023-10-03 14:15:00 | 830.65 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2023-10-03 12:00:00 | 831.70 | 2023-10-03 14:15:00 | 830.65 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2023-10-05 13:00:00 | 821.00 | 2023-10-12 10:15:00 | 805.30 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2023-10-05 14:00:00 | 821.80 | 2023-10-12 10:15:00 | 805.30 | STOP_HIT | 1.00 | 2.01% |
| SELL | retest2 | 2023-10-05 14:45:00 | 821.70 | 2023-10-12 10:15:00 | 805.30 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest2 | 2023-11-02 12:00:00 | 737.65 | 2023-11-02 15:15:00 | 741.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-11-06 09:15:00 | 745.75 | 2023-11-13 10:15:00 | 769.45 | STOP_HIT | 1.00 | 3.18% |
| BUY | retest2 | 2023-11-06 11:30:00 | 762.50 | 2023-11-13 10:15:00 | 769.45 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2023-11-23 09:15:00 | 821.10 | 2023-11-30 09:15:00 | 819.35 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2023-12-04 09:15:00 | 842.20 | 2023-12-05 11:15:00 | 832.70 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-12-04 13:45:00 | 838.35 | 2023-12-05 11:15:00 | 832.70 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-12-05 09:15:00 | 842.40 | 2023-12-05 11:15:00 | 832.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-12-05 13:00:00 | 838.50 | 2023-12-05 14:15:00 | 836.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-12-08 09:15:00 | 861.50 | 2023-12-08 10:15:00 | 844.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2023-12-08 10:30:00 | 850.15 | 2023-12-08 12:15:00 | 836.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest1 | 2023-12-20 09:15:00 | 854.55 | 2023-12-20 11:15:00 | 844.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-12-29 09:15:00 | 871.05 | 2024-01-02 11:15:00 | 863.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-01-02 10:00:00 | 865.15 | 2024-01-02 11:15:00 | 863.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-01-02 10:45:00 | 864.05 | 2024-01-02 11:15:00 | 863.70 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-01-02 11:15:00 | 864.05 | 2024-01-02 11:15:00 | 863.70 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-01-12 09:15:00 | 889.00 | 2024-01-12 12:15:00 | 876.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-01-17 09:15:00 | 896.60 | 2024-01-18 09:15:00 | 879.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-02-22 14:00:00 | 808.25 | 2024-02-22 15:15:00 | 817.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-03-07 12:00:00 | 812.70 | 2024-03-12 10:15:00 | 772.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 14:15:00 | 811.25 | 2024-03-12 11:15:00 | 770.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 12:00:00 | 812.70 | 2024-03-13 10:15:00 | 731.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 14:15:00 | 811.25 | 2024-03-13 10:15:00 | 730.12 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-28 09:30:00 | 782.30 | 2024-03-28 11:15:00 | 775.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-04-15 09:15:00 | 851.50 | 2024-04-16 12:15:00 | 891.75 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2024-05-29 10:30:00 | 1430.00 | 2024-05-31 09:15:00 | 1367.00 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2024-05-29 11:15:00 | 1430.90 | 2024-05-31 09:15:00 | 1367.00 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2024-05-30 10:15:00 | 1433.00 | 2024-05-31 09:15:00 | 1367.00 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2024-07-09 09:15:00 | 2654.00 | 2024-07-09 09:15:00 | 2560.00 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2024-08-08 12:45:00 | 2162.00 | 2024-08-09 09:15:00 | 2053.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 13:45:00 | 2130.40 | 2024-08-12 09:15:00 | 2023.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 12:45:00 | 2162.00 | 2024-08-13 14:15:00 | 1945.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-08 13:45:00 | 2130.40 | 2024-08-14 09:15:00 | 1917.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-19 10:45:00 | 1722.25 | 2024-09-20 10:15:00 | 1833.95 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1674.05 | 2024-09-30 14:15:00 | 1728.60 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2024-09-30 13:00:00 | 1691.70 | 2024-09-30 14:15:00 | 1728.60 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1687.00 | 2024-10-04 09:15:00 | 1602.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1687.00 | 2024-10-04 10:15:00 | 1669.95 | STOP_HIT | 0.50 | 1.01% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1861.00 | 2024-10-22 09:15:00 | 1716.15 | STOP_HIT | 1.00 | -7.78% |
| SELL | retest2 | 2024-11-05 15:15:00 | 1549.00 | 2024-11-06 09:15:00 | 1567.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-18 09:30:00 | 1371.10 | 2024-11-25 10:15:00 | 1434.00 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2024-11-22 14:45:00 | 1374.70 | 2024-11-25 10:15:00 | 1434.00 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2024-12-03 13:15:00 | 1724.30 | 2024-12-10 09:15:00 | 1721.65 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-12-04 09:15:00 | 1771.90 | 2024-12-10 09:15:00 | 1721.65 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-12-11 14:45:00 | 1736.65 | 2024-12-12 09:15:00 | 1789.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-12-11 15:15:00 | 1731.00 | 2024-12-12 09:15:00 | 1789.00 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2024-12-26 15:15:00 | 1552.00 | 2024-12-27 13:15:00 | 1681.00 | STOP_HIT | 1.00 | -8.31% |
| SELL | retest2 | 2025-01-07 14:30:00 | 1574.05 | 2025-01-10 09:15:00 | 1495.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 14:30:00 | 1574.05 | 2025-01-13 10:15:00 | 1416.64 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 09:15:00 | 1534.00 | 2025-02-01 09:15:00 | 1687.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 15:15:00 | 1484.65 | 2025-02-12 09:15:00 | 1410.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 15:15:00 | 1484.65 | 2025-02-13 09:15:00 | 1428.90 | STOP_HIT | 0.50 | 3.76% |
| BUY | retest2 | 2025-02-21 11:30:00 | 1345.40 | 2025-02-21 15:15:00 | 1316.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-02-21 12:15:00 | 1341.50 | 2025-02-21 15:15:00 | 1316.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-03-04 11:30:00 | 1264.10 | 2025-03-05 10:15:00 | 1279.60 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-03-07 09:15:00 | 1322.80 | 2025-03-11 11:15:00 | 1315.25 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-03-24 09:15:00 | 1728.80 | 2025-03-26 14:15:00 | 1704.75 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest1 | 2025-03-24 15:15:00 | 1705.00 | 2025-03-26 14:15:00 | 1704.75 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest1 | 2025-03-25 11:45:00 | 1713.00 | 2025-03-26 14:15:00 | 1704.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-03-26 10:15:00 | 1740.00 | 2025-03-28 12:15:00 | 1693.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-03-26 12:00:00 | 1734.80 | 2025-03-28 12:15:00 | 1693.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-03-26 12:45:00 | 1744.65 | 2025-03-28 12:15:00 | 1693.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1536.40 | 2025-04-11 09:15:00 | 1640.15 | STOP_HIT | 1.00 | -6.75% |
| SELL | retest2 | 2025-04-08 11:00:00 | 1548.70 | 2025-04-11 09:15:00 | 1640.15 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1532.55 | 2025-04-11 09:15:00 | 1640.15 | STOP_HIT | 1.00 | -7.02% |
| BUY | retest2 | 2025-04-23 12:45:00 | 1737.60 | 2025-04-25 09:15:00 | 1645.30 | STOP_HIT | 1.00 | -5.31% |
| BUY | retest2 | 2025-04-23 13:15:00 | 1735.20 | 2025-04-25 09:15:00 | 1645.30 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest2 | 2025-04-24 09:15:00 | 1761.90 | 2025-04-25 09:15:00 | 1645.30 | STOP_HIT | 1.00 | -6.62% |
| BUY | retest2 | 2025-04-24 11:30:00 | 1744.40 | 2025-04-25 09:15:00 | 1645.30 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2025-05-05 12:00:00 | 1911.10 | 2025-05-06 13:15:00 | 1862.90 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-05-08 13:45:00 | 1789.80 | 2025-05-12 15:15:00 | 1823.70 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1774.50 | 2025-05-13 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -5.75% |
| SELL | retest2 | 2025-05-09 10:00:00 | 1793.90 | 2025-05-13 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-05-09 10:30:00 | 1788.90 | 2025-05-13 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -4.90% |
| SELL | retest2 | 2025-05-09 14:15:00 | 1788.00 | 2025-05-13 09:15:00 | 1876.60 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2025-05-22 09:30:00 | 2587.00 | 2025-05-23 09:15:00 | 2845.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-10 11:45:00 | 3189.70 | 2025-06-12 11:15:00 | 3030.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 09:15:00 | 3176.80 | 2025-06-12 12:15:00 | 3017.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 11:45:00 | 3189.70 | 2025-06-13 09:15:00 | 3037.70 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-06-11 09:15:00 | 3176.80 | 2025-06-13 09:15:00 | 3037.70 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2025-07-01 12:00:00 | 2992.70 | 2025-07-08 09:15:00 | 2843.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2991.10 | 2025-07-08 09:15:00 | 2841.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-04 11:45:00 | 2978.00 | 2025-07-08 09:15:00 | 2829.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 12:00:00 | 2992.70 | 2025-07-09 09:15:00 | 2966.00 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2991.10 | 2025-07-09 09:15:00 | 2966.00 | STOP_HIT | 0.50 | 0.84% |
| SELL | retest2 | 2025-07-04 11:45:00 | 2978.00 | 2025-07-09 09:15:00 | 2966.00 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2025-07-18 10:15:00 | 2600.50 | 2025-07-22 15:15:00 | 2631.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-21 09:15:00 | 2583.30 | 2025-07-22 15:15:00 | 2631.20 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-07-21 11:30:00 | 2597.50 | 2025-07-22 15:15:00 | 2631.20 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-07-22 09:30:00 | 2596.80 | 2025-07-22 15:15:00 | 2631.20 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-05 09:15:00 | 2662.00 | 2025-08-06 14:15:00 | 2597.90 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-08-05 10:00:00 | 2661.80 | 2025-08-06 14:15:00 | 2597.90 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest1 | 2025-08-08 09:15:00 | 2493.20 | 2025-08-11 09:15:00 | 2577.10 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2025-08-14 12:45:00 | 2571.00 | 2025-08-20 10:15:00 | 2589.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-08-21 12:15:00 | 2596.80 | 2025-08-26 09:15:00 | 2466.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 10:30:00 | 2610.00 | 2025-08-26 09:15:00 | 2479.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 2596.80 | 2025-08-28 14:15:00 | 2337.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-22 10:30:00 | 2610.00 | 2025-08-28 14:15:00 | 2349.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-24 09:15:00 | 2708.00 | 2025-09-26 11:15:00 | 2643.70 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-10-08 09:15:00 | 2724.60 | 2025-10-08 10:15:00 | 2672.60 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-10-08 13:30:00 | 2725.00 | 2025-10-08 14:15:00 | 2678.70 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-10-16 10:15:00 | 2600.20 | 2025-10-17 09:15:00 | 2642.10 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-10-30 10:15:00 | 2548.50 | 2025-11-03 15:15:00 | 2580.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-31 11:00:00 | 2550.10 | 2025-11-03 15:15:00 | 2580.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-11-03 12:15:00 | 2550.10 | 2025-11-03 15:15:00 | 2580.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-28 15:15:00 | 2799.00 | 2025-12-01 11:15:00 | 2749.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-10 09:15:00 | 2420.40 | 2025-12-16 09:15:00 | 2299.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 09:15:00 | 2420.40 | 2025-12-18 10:15:00 | 2178.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-08 09:15:00 | 2515.60 | 2026-01-08 13:15:00 | 2477.50 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-13 11:45:00 | 2455.00 | 2026-01-20 10:15:00 | 2332.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2454.40 | 2026-01-20 10:15:00 | 2331.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2454.60 | 2026-01-20 10:15:00 | 2331.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 2455.00 | 2026-01-21 12:15:00 | 2297.50 | STOP_HIT | 0.50 | 6.42% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2454.40 | 2026-01-21 12:15:00 | 2297.50 | STOP_HIT | 0.50 | 6.39% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2454.60 | 2026-01-21 12:15:00 | 2297.50 | STOP_HIT | 0.50 | 6.40% |
| SELL | retest2 | 2026-02-17 14:00:00 | 2429.60 | 2026-02-18 13:15:00 | 2501.30 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2026-02-18 09:30:00 | 2425.30 | 2026-02-18 13:15:00 | 2501.30 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-02-18 12:15:00 | 2429.80 | 2026-02-18 13:15:00 | 2501.30 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-02-18 12:45:00 | 2430.60 | 2026-02-18 13:15:00 | 2501.30 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-02-25 11:45:00 | 2425.50 | 2026-02-26 09:15:00 | 2462.40 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-25 12:30:00 | 2420.00 | 2026-02-26 09:15:00 | 2462.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-03-20 12:15:00 | 2302.70 | 2026-03-23 09:15:00 | 2187.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 2310.40 | 2026-03-23 09:15:00 | 2194.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 2302.70 | 2026-03-24 11:15:00 | 2163.00 | STOP_HIT | 0.50 | 6.07% |
| SELL | retest2 | 2026-03-20 13:30:00 | 2310.40 | 2026-03-24 11:15:00 | 2163.00 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2026-04-01 10:00:00 | 2299.70 | 2026-04-01 10:15:00 | 2316.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-04-02 11:30:00 | 2218.80 | 2026-04-08 09:15:00 | 2440.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:15:00 | 2238.30 | 2026-04-08 09:15:00 | 2462.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:00:00 | 2229.00 | 2026-04-08 09:15:00 | 2451.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-05 13:15:00 | 2855.00 | 2026-05-05 14:15:00 | 2910.00 | STOP_HIT | 1.00 | -1.93% |
