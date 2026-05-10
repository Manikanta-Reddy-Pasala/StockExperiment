# Sun TV Network Ltd. (SUNTV)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 572.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 10 / 15
- **Target hits / Stop hits / Partials:** 6 / 17 / 2
- **Avg / median % per leg:** 1.24% / -0.93%
- **Sum % (uncompounded):** 30.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 6 | 31.6% | 6 | 13 | 0 | 1.30% | 24.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 6 | 31.6% | 6 | 13 | 0 | 1.30% | 24.7% |
| SELL (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.03% | 6.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.03% | 6.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 10 | 40.0% | 6 | 17 | 2 | 1.24% | 30.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 574.20 | 560.34 | 560.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 584.35 | 560.58 | 560.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 564.85 | 567.32 | 564.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 564.85 | 567.32 | 564.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 558.45 | 567.23 | 564.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 558.45 | 567.23 | 564.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 559.10 | 567.15 | 564.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 567.10 | 566.83 | 564.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 561.35 | 566.65 | 564.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 557.95 | 566.50 | 563.98 | SL hit (close<static) qty=1.00 sl=558.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 557.95 | 566.50 | 563.98 | SL hit (close<static) qty=1.00 sl=558.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 561.95 | 565.15 | 563.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 556.70 | 565.03 | 563.37 | SL hit (close<static) qty=1.00 sl=558.05 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 540.35 | 561.90 | 561.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 539.25 | 561.06 | 561.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 560.25 | 559.02 | 560.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 561.75 | 559.02 | 560.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 556.05 | 558.99 | 560.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:30:00 | 557.00 | 558.99 | 560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 562.35 | 558.96 | 560.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 562.35 | 558.96 | 560.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 560.00 | 558.97 | 560.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 535.45 | 558.97 | 560.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 557.95 | 550.15 | 554.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 571.45 | 550.72 | 554.86 | SL hit (close>static) qty=1.00 sl=565.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 571.45 | 550.72 | 554.86 | SL hit (close>static) qty=1.00 sl=565.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 610.55 | 558.89 | 558.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 635.55 | 575.72 | 568.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 576.00 | 585.05 | 575.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 576.30 | 585.05 | 575.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 577.00 | 584.97 | 575.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 584.45 | 584.24 | 575.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 585.00 | 584.26 | 575.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 584.60 | 584.26 | 575.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 13:30:00 | 584.85 | 584.21 | 575.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 579.75 | 584.04 | 575.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 585.25 | 584.01 | 575.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 573.95 | 584.08 | 575.91 | SL hit (close<static) qty=1.00 sl=574.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 573.95 | 584.08 | 575.91 | SL hit (close<static) qty=1.00 sl=574.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 573.95 | 584.08 | 575.91 | SL hit (close<static) qty=1.00 sl=574.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 573.95 | 584.08 | 575.91 | SL hit (close<static) qty=1.00 sl=574.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 563.65 | 583.75 | 575.83 | SL hit (close<static) qty=1.00 sl=565.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 587.05 | 580.43 | 574.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 585.80 | 584.37 | 577.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:30:00 | 584.95 | 584.35 | 577.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-25 09:15:00 | 645.75 | 585.72 | 578.46 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-25 09:15:00 | 644.38 | 585.72 | 578.46 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-25 09:15:00 | 643.45 | 585.72 | 578.46 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 582.15 | 587.29 | 579.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 584.95 | 587.29 | 579.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 588.00 | 587.30 | 579.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:30:00 | 600.00 | 585.53 | 579.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 601.50 | 586.22 | 580.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:00:00 | 595.05 | 587.56 | 581.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 597.40 | 587.64 | 581.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 09:15:00 | 660.00 | 599.08 | 588.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-20 09:15:00 | 654.55 | 599.08 | 588.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-20 09:15:00 | 657.14 | 599.08 | 588.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 595.70 | 609.37 | 595.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 595.70 | 609.37 | 595.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 601.40 | 609.29 | 595.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 15:15:00 | 604.00 | 609.29 | 595.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 594.30 | 609.09 | 595.88 | SL hit (close<static) qty=1.00 sl=595.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 10:30:00 | 604.65 | 608.97 | 595.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 603.30 | 608.97 | 595.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 592.40 | 608.34 | 596.20 | SL hit (close<static) qty=1.00 sl=595.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 592.40 | 608.34 | 596.20 | SL hit (close<static) qty=1.00 sl=595.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 605.50 | 606.74 | 596.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 556.45 | 606.20 | 596.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 556.45 | 606.20 | 596.08 | SL hit (close<static) qty=1.00 sl=571.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 556.45 | 606.20 | 596.08 | SL hit (close<static) qty=1.00 sl=595.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 556.45 | 606.20 | 596.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 555.95 | 605.70 | 595.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 564.00 | 605.25 | 595.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-22 11:30:00 | 591.45 | 2025-07-29 10:15:00 | 561.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:30:00 | 591.45 | 2025-08-12 09:15:00 | 577.90 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-10-01 15:15:00 | 588.15 | 2025-10-15 15:15:00 | 558.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 15:15:00 | 588.15 | 2025-10-16 09:15:00 | 570.30 | STOP_HIT | 0.50 | 3.03% |
| BUY | retest2 | 2026-01-08 09:15:00 | 567.10 | 2026-01-09 11:15:00 | 557.95 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-09 10:15:00 | 561.35 | 2026-01-09 11:15:00 | 557.95 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-01-12 15:15:00 | 561.95 | 2026-01-13 09:15:00 | 556.70 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-27 09:15:00 | 535.45 | 2026-02-09 14:15:00 | 571.45 | STOP_HIT | 1.00 | -6.72% |
| SELL | retest2 | 2026-02-09 11:15:00 | 557.95 | 2026-02-09 14:15:00 | 571.45 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-03-11 09:15:00 | 584.45 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-03-11 10:45:00 | 585.00 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-03-11 11:30:00 | 584.60 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-03-11 13:30:00 | 584.85 | 2026-03-13 12:15:00 | 573.95 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-12 12:15:00 | 585.25 | 2026-03-13 14:15:00 | 563.65 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-03-18 10:15:00 | 587.05 | 2026-03-25 09:15:00 | 645.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 10:00:00 | 585.80 | 2026-03-25 09:15:00 | 644.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 11:30:00 | 584.95 | 2026-03-25 09:15:00 | 643.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:30:00 | 600.00 | 2026-04-20 09:15:00 | 660.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 601.50 | 2026-04-20 09:15:00 | 654.55 | TARGET_HIT | 1.00 | 8.82% |
| BUY | retest2 | 2026-04-09 11:00:00 | 595.05 | 2026-04-20 09:15:00 | 657.14 | TARGET_HIT | 1.00 | 10.43% |
| BUY | retest2 | 2026-04-09 11:30:00 | 597.40 | 2026-04-27 09:15:00 | 594.30 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-04-24 15:15:00 | 604.00 | 2026-04-28 13:15:00 | 592.40 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-04-27 10:30:00 | 604.65 | 2026-04-28 13:15:00 | 592.40 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2026-04-27 11:15:00 | 603.30 | 2026-05-04 09:15:00 | 556.45 | STOP_HIT | 1.00 | -7.77% |
| BUY | retest2 | 2026-04-30 12:15:00 | 605.50 | 2026-05-04 09:15:00 | 556.45 | STOP_HIT | 1.00 | -8.10% |
