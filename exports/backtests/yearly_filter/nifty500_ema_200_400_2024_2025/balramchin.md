# Balrampur Chini Mills Ltd. (BALRAMCHIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 522.00
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
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 2 |
| TARGET_HIT | 5 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 15
- **Target hits / Stop hits / Partials:** 5 / 17 / 2
- **Avg / median % per leg:** 0.86% / -0.59%
- **Sum % (uncompounded):** 20.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 5 | 1 | 0 | 8.36% | 50.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 5 | 1 | 0 | 8.36% | 50.1% |
| SELL (all) | 18 | 3 | 16.7% | 0 | 16 | 2 | -1.64% | -29.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 3 | 16.7% | 0 | 16 | 2 | -1.64% | -29.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 9 | 37.5% | 5 | 17 | 2 | 0.86% | 20.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 404.10 | 381.82 | 381.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 405.10 | 382.05 | 381.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 609.75 | 622.84 | 583.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 609.75 | 622.84 | 583.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 595.00 | 620.84 | 595.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 588.00 | 620.84 | 595.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 584.00 | 620.48 | 595.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 583.05 | 620.48 | 595.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 582.05 | 620.10 | 595.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:15:00 | 582.05 | 620.10 | 595.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 586.30 | 618.30 | 595.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 580.80 | 618.30 | 595.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 555.55 | 579.69 | 579.79 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 609.00 | 579.61 | 579.61 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 538.00 | 580.12 | 580.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 15:15:00 | 529.95 | 579.62 | 579.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 13:15:00 | 470.25 | 468.44 | 496.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 14:00:00 | 470.25 | 468.44 | 496.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 489.15 | 460.92 | 481.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 489.15 | 460.92 | 481.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 477.70 | 461.08 | 481.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 472.80 | 461.08 | 481.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:15:00 | 473.85 | 462.36 | 480.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 501.90 | 464.17 | 480.93 | SL hit (close>static) qty=1.00 sl=497.15 alert=retest2 |

### Cycle 5 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 545.40 | 493.79 | 493.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 546.60 | 494.32 | 493.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 545.15 | 545.90 | 528.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 545.15 | 545.90 | 528.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 539.05 | 547.35 | 530.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:45:00 | 551.35 | 546.21 | 530.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 549.10 | 550.89 | 536.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 549.25 | 550.96 | 537.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-03 09:15:00 | 606.49 | 560.28 | 545.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 550.00 | 581.55 | 581.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 547.20 | 579.92 | 580.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 578.00 | 574.90 | 577.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 577.70 | 574.91 | 577.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 582.70 | 574.99 | 578.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 583.05 | 574.99 | 578.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 582.85 | 575.07 | 578.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 582.85 | 575.07 | 578.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 585.05 | 576.08 | 578.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 585.70 | 576.08 | 578.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 577.70 | 570.81 | 575.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 578.55 | 570.81 | 575.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 578.85 | 570.89 | 575.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 579.65 | 570.89 | 575.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 436.50 | 424.08 | 438.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 433.50 | 424.18 | 438.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 14:15:00 | 411.82 | 423.79 | 437.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 431.00 | 423.82 | 437.45 | SL hit (close>ema200) qty=0.50 sl=423.82 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 454.75 | 444.83 | 444.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 465.30 | 445.63 | 445.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 12:15:00 | 469.30 | 470.94 | 460.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 469.30 | 470.94 | 460.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 452.00 | 472.74 | 463.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 452.00 | 472.74 | 463.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 451.15 | 472.53 | 463.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 446.60 | 472.53 | 463.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 472.15 | 480.03 | 470.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 476.85 | 480.00 | 470.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 475.95 | 479.86 | 470.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 523.55 | 482.66 | 473.55 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 14:00:00 | 378.75 | 2024-05-17 14:15:00 | 384.70 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-05-16 15:15:00 | 378.40 | 2024-05-17 14:15:00 | 384.70 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-21 09:15:00 | 376.90 | 2024-05-24 12:15:00 | 384.25 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-05-21 12:45:00 | 378.20 | 2024-05-24 12:15:00 | 384.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-28 11:15:00 | 380.10 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-05-28 13:30:00 | 379.90 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-05-29 09:15:00 | 379.25 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-05-29 10:00:00 | 380.25 | 2024-05-29 12:15:00 | 382.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-05-30 14:00:00 | 377.90 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-05-31 09:45:00 | 379.15 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-05-31 15:00:00 | 379.00 | 2024-06-03 09:15:00 | 387.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-06-04 09:30:00 | 379.10 | 2024-06-04 11:15:00 | 360.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:30:00 | 379.10 | 2024-06-05 10:15:00 | 387.85 | STOP_HIT | 0.50 | -2.31% |
| SELL | retest2 | 2024-06-04 12:00:00 | 353.10 | 2024-06-05 10:15:00 | 387.85 | STOP_HIT | 1.00 | -9.84% |
| SELL | retest2 | 2025-03-13 09:15:00 | 472.80 | 2025-03-18 14:15:00 | 501.90 | STOP_HIT | 1.00 | -6.15% |
| SELL | retest2 | 2025-03-17 15:15:00 | 473.85 | 2025-03-18 14:15:00 | 501.90 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest2 | 2025-05-12 09:45:00 | 551.35 | 2025-06-03 09:15:00 | 606.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 09:30:00 | 549.10 | 2025-06-03 09:15:00 | 604.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 11:00:00 | 549.25 | 2025-06-03 09:15:00 | 604.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 11:15:00 | 549.25 | 2025-08-12 10:15:00 | 550.00 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-02-01 12:00:00 | 433.50 | 2026-02-02 14:15:00 | 411.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:00:00 | 433.50 | 2026-02-03 09:15:00 | 431.00 | STOP_HIT | 0.50 | 0.58% |
| BUY | retest2 | 2026-04-13 11:00:00 | 476.85 | 2026-04-21 09:15:00 | 523.55 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2026-04-13 15:00:00 | 475.95 | 2026-04-22 13:15:00 | 524.54 | TARGET_HIT | 1.00 | 10.21% |
