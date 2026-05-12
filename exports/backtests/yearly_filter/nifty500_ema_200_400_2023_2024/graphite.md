# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 752.00
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
| ALERT2_SKIP | 2 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 38 |
| PARTIAL | 8 |
| TARGET_HIT | 11 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 25
- **Target hits / Stop hits / Partials:** 11 / 27 / 8
- **Avg / median % per leg:** 1.79% / -0.93%
- **Sum % (uncompounded):** 82.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 6 | 37.5% | 6 | 10 | 0 | 2.12% | 34.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 6 | 10 | 0 | 2.12% | 34.0% |
| SELL (all) | 30 | 15 | 50.0% | 5 | 17 | 8 | 1.62% | 48.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 15 | 50.0% | 5 | 17 | 8 | 1.62% | 48.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 21 | 45.7% | 11 | 27 | 8 | 1.79% | 82.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 573.15 | 613.94 | 614.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 568.85 | 613.49 | 613.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 12:15:00 | 589.40 | 585.15 | 595.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 13:00:00 | 589.40 | 585.15 | 595.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 597.00 | 585.34 | 595.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 09:45:00 | 589.80 | 585.38 | 595.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 601.95 | 585.58 | 595.61 | SL hit (close>static) qty=1.00 sl=598.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 599.50 | 545.39 | 545.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 604.25 | 555.63 | 550.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 552.75 | 560.03 | 553.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 552.75 | 560.03 | 553.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 552.75 | 560.03 | 553.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 552.75 | 560.03 | 553.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 542.50 | 559.86 | 553.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 542.50 | 559.86 | 553.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 555.00 | 558.52 | 553.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:30:00 | 546.60 | 558.52 | 553.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 553.50 | 558.47 | 553.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 14:45:00 | 555.70 | 558.45 | 553.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 540.65 | 567.33 | 559.87 | SL hit (close<static) qty=1.00 sl=549.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 11:15:00 | 504.25 | 553.56 | 553.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 15:15:00 | 502.00 | 551.62 | 552.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 541.20 | 540.83 | 546.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:30:00 | 542.55 | 540.83 | 546.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 521.25 | 509.26 | 524.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 517.50 | 509.26 | 524.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 10:15:00 | 519.05 | 509.37 | 524.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 15:15:00 | 526.95 | 509.95 | 524.51 | SL hit (close>static) qty=1.00 sl=524.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 10:15:00 | 563.50 | 535.78 | 535.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 15:15:00 | 565.25 | 537.14 | 536.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 550.65 | 554.11 | 547.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 550.65 | 554.11 | 547.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 547.00 | 554.19 | 547.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 547.00 | 554.19 | 547.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 547.30 | 554.12 | 547.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 543.35 | 554.12 | 547.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 547.00 | 554.05 | 547.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 547.05 | 554.05 | 547.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 547.85 | 553.99 | 547.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 13:15:00 | 548.50 | 553.99 | 547.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 10:00:00 | 548.70 | 553.76 | 547.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 527.25 | 553.12 | 547.52 | SL hit (close<static) qty=1.00 sl=543.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 09:15:00 | 515.10 | 542.57 | 542.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 13:15:00 | 511.05 | 541.41 | 542.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 09:15:00 | 513.50 | 504.59 | 518.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 513.50 | 504.59 | 518.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 513.50 | 504.59 | 518.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 516.60 | 504.59 | 518.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 460.75 | 432.56 | 459.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:45:00 | 462.55 | 432.56 | 459.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 460.25 | 432.83 | 459.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 467.15 | 432.83 | 459.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 455.00 | 435.08 | 459.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:30:00 | 462.35 | 435.08 | 459.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 461.15 | 435.95 | 459.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:00:00 | 461.15 | 435.95 | 459.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 458.50 | 436.17 | 459.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 15:15:00 | 455.55 | 436.17 | 459.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 465.85 | 436.66 | 459.11 | SL hit (close>static) qty=1.00 sl=462.05 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 556.20 | 468.36 | 467.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 11:15:00 | 566.45 | 469.34 | 468.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 525.00 | 527.16 | 506.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 09:30:00 | 522.70 | 527.16 | 506.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 545.25 | 563.12 | 545.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 545.25 | 563.12 | 545.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 544.40 | 562.93 | 545.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 547.50 | 562.31 | 544.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 543.00 | 561.76 | 545.03 | SL hit (close<static) qty=1.00 sl=543.10 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 510.60 | 539.93 | 540.07 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 576.00 | 537.64 | 537.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 578.20 | 545.20 | 541.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 545.25 | 547.58 | 543.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 545.25 | 547.58 | 543.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 543.45 | 547.84 | 543.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 543.45 | 547.84 | 543.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 542.45 | 547.78 | 543.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 547.65 | 547.78 | 543.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-08 12:15:00 | 602.42 | 552.38 | 546.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 537.65 | 561.50 | 561.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 536.20 | 561.25 | 561.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 550.15 | 549.67 | 554.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 550.15 | 549.67 | 554.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 565.70 | 549.84 | 554.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 565.70 | 549.84 | 554.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 562.00 | 549.96 | 554.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 559.40 | 550.05 | 554.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 561.85 | 550.26 | 554.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 573.20 | 550.90 | 555.00 | SL hit (close>static) qty=1.00 sl=566.20 alert=retest2 |

### Cycle 10 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 598.00 | 558.62 | 558.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 609.55 | 561.25 | 559.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 600.15 | 601.83 | 584.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 600.15 | 601.83 | 584.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 597.00 | 619.94 | 600.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 618.20 | 616.94 | 600.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 612.10 | 618.98 | 603.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:30:00 | 614.80 | 618.99 | 603.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-10 12:15:00 | 673.31 | 621.85 | 606.03 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-20 09:45:00 | 589.80 | 2024-06-20 11:15:00 | 601.95 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-06-20 15:15:00 | 588.80 | 2024-06-28 14:15:00 | 560.69 | PARTIAL | 0.50 | 4.77% |
| SELL | retest2 | 2024-06-21 10:45:00 | 590.20 | 2024-06-28 15:15:00 | 560.50 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-06-20 15:15:00 | 588.80 | 2024-07-04 09:15:00 | 582.50 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-06-21 10:45:00 | 590.20 | 2024-07-04 09:15:00 | 582.50 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2024-06-21 11:45:00 | 590.00 | 2024-07-08 11:15:00 | 559.36 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2024-06-21 15:00:00 | 588.90 | 2024-07-08 11:15:00 | 559.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 12:15:00 | 588.20 | 2024-07-08 11:15:00 | 558.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 12:45:00 | 588.20 | 2024-07-08 11:15:00 | 558.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-24 13:15:00 | 588.85 | 2024-07-08 11:15:00 | 559.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 11:45:00 | 590.00 | 2024-07-19 15:15:00 | 529.92 | TARGET_HIT | 0.50 | 10.18% |
| SELL | retest2 | 2024-06-21 15:00:00 | 588.90 | 2024-07-19 15:15:00 | 530.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-24 12:15:00 | 588.20 | 2024-07-19 15:15:00 | 529.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-24 12:45:00 | 588.20 | 2024-07-19 15:15:00 | 529.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-24 13:15:00 | 588.85 | 2024-07-19 15:15:00 | 529.97 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-08 14:45:00 | 555.70 | 2024-10-22 09:15:00 | 540.65 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-12-02 09:15:00 | 517.50 | 2024-12-02 15:15:00 | 526.95 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-12-02 10:15:00 | 519.05 | 2024-12-02 15:15:00 | 526.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-01-08 13:15:00 | 548.50 | 2025-01-10 09:15:00 | 527.25 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-01-09 10:00:00 | 548.70 | 2025-01-10 09:15:00 | 527.25 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-03-20 15:15:00 | 455.55 | 2025-03-21 09:15:00 | 465.85 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-04-07 09:15:00 | 448.95 | 2025-04-07 09:15:00 | 426.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 448.95 | 2025-04-08 11:15:00 | 464.75 | STOP_HIT | 0.50 | -3.52% |
| SELL | retest2 | 2025-04-08 10:30:00 | 457.10 | 2025-04-08 11:15:00 | 464.75 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-04-09 09:15:00 | 456.00 | 2025-04-15 09:15:00 | 464.40 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-04-09 13:30:00 | 449.50 | 2025-04-15 11:15:00 | 472.95 | STOP_HIT | 1.00 | -5.22% |
| SELL | retest2 | 2025-04-11 09:45:00 | 447.45 | 2025-04-15 11:15:00 | 472.95 | STOP_HIT | 1.00 | -5.70% |
| SELL | retest2 | 2025-04-11 13:30:00 | 450.00 | 2025-04-15 11:15:00 | 472.95 | STOP_HIT | 1.00 | -5.10% |
| SELL | retest2 | 2025-05-02 11:15:00 | 449.90 | 2025-05-05 12:15:00 | 469.85 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-05-13 09:15:00 | 463.60 | 2025-05-14 09:15:00 | 474.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-05-13 13:45:00 | 464.00 | 2025-05-14 09:15:00 | 474.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-07-28 09:15:00 | 547.50 | 2025-07-28 12:15:00 | 543.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-29 13:45:00 | 549.05 | 2025-08-01 14:15:00 | 527.10 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-08-18 14:15:00 | 550.25 | 2025-08-20 15:15:00 | 542.75 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-21 09:15:00 | 547.85 | 2025-08-21 12:15:00 | 542.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-03 13:15:00 | 547.65 | 2025-10-08 12:15:00 | 602.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 09:30:00 | 546.80 | 2025-11-18 13:15:00 | 601.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 10:00:00 | 545.00 | 2025-12-04 12:15:00 | 538.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-04 09:45:00 | 544.55 | 2025-12-04 12:15:00 | 538.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-22 12:00:00 | 559.40 | 2025-12-23 10:15:00 | 573.20 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-12-22 13:45:00 | 561.85 | 2025-12-23 10:15:00 | 573.20 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-02-03 09:15:00 | 618.20 | 2026-02-10 12:15:00 | 673.31 | TARGET_HIT | 1.00 | 8.91% |
| BUY | retest2 | 2026-02-06 10:30:00 | 612.10 | 2026-02-11 09:15:00 | 676.28 | TARGET_HIT | 1.00 | 10.49% |
| BUY | retest2 | 2026-02-06 13:30:00 | 614.80 | 2026-02-17 09:15:00 | 680.02 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2026-03-16 09:30:00 | 614.05 | 2026-03-23 09:15:00 | 575.50 | STOP_HIT | 1.00 | -6.28% |
| BUY | retest2 | 2026-04-15 10:00:00 | 663.40 | 2026-04-20 10:15:00 | 729.74 | TARGET_HIT | 1.00 | 10.00% |
