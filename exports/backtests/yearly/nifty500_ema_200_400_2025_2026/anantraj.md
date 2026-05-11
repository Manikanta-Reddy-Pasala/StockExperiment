# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (2730 bars)
- **Last close:** 561.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 18 / 14
- **Target hits / Stop hits / Partials:** 4 / 18 / 10
- **Avg / median % per leg:** 2.24% / 2.43%
- **Sum % (uncompounded):** 71.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.35% | -9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.35% | -9.4% |
| SELL (all) | 25 | 18 | 72.0% | 4 | 11 | 10 | 3.24% | 80.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 18 | 72.0% | 4 | 11 | 10 | 3.24% | 80.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 18 | 56.2% | 4 | 18 | 10 | 2.24% | 71.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 687.95 | 559.23 | 558.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 703.90 | 572.21 | 565.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 639.65 | 648.66 | 616.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 13:15:00 | 619.50 | 643.48 | 618.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 619.50 | 643.48 | 618.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 619.10 | 643.48 | 618.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 618.00 | 643.22 | 618.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:15:00 | 616.50 | 643.22 | 618.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 616.50 | 642.96 | 618.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 622.30 | 642.96 | 618.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 623.70 | 641.34 | 623.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 623.70 | 641.34 | 623.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 624.40 | 641.18 | 623.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 633.30 | 638.73 | 622.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 626.35 | 638.18 | 623.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 627.20 | 638.07 | 623.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 618.45 | 637.76 | 623.10 | SL hit (close<static) qty=1.00 sl=622.20 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 578.75 | 616.36 | 616.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 572.00 | 615.52 | 615.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 564.60 | 562.70 | 581.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:45:00 | 566.50 | 562.70 | 581.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 575.55 | 558.37 | 575.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 575.05 | 558.37 | 575.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 583.80 | 558.62 | 575.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 583.80 | 558.62 | 575.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 586.00 | 558.90 | 575.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 588.30 | 558.90 | 575.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 579.35 | 565.88 | 577.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 579.35 | 565.88 | 577.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 572.80 | 565.94 | 577.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 577.60 | 565.94 | 577.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 566.60 | 564.60 | 575.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 561.75 | 564.56 | 575.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 560.20 | 564.56 | 575.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 560.45 | 564.51 | 575.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 559.80 | 564.44 | 575.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 533.66 | 561.61 | 572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.19 | 561.61 | 572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.43 | 561.61 | 572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 531.81 | 561.61 | 572.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 13:15:00 | 505.57 | 553.44 | 566.90 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-10 10:00:00 | 582.70 | 2025-06-12 13:15:00 | 553.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-10 10:00:00 | 582.70 | 2025-06-12 15:15:00 | 559.50 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-07-21 09:30:00 | 583.70 | 2025-07-25 09:15:00 | 591.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-21 11:30:00 | 584.40 | 2025-07-25 09:15:00 | 591.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-07-21 14:45:00 | 584.35 | 2025-07-29 09:15:00 | 554.51 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-07-23 13:30:00 | 560.85 | 2025-07-29 09:15:00 | 555.18 | PARTIAL | 0.50 | 1.01% |
| SELL | retest2 | 2025-07-24 15:15:00 | 559.00 | 2025-07-29 09:15:00 | 555.13 | PARTIAL | 0.50 | 0.69% |
| SELL | retest2 | 2025-07-21 14:45:00 | 584.35 | 2025-07-29 10:15:00 | 567.50 | STOP_HIT | 0.50 | 2.88% |
| SELL | retest2 | 2025-07-23 13:30:00 | 560.85 | 2025-07-29 10:15:00 | 567.50 | STOP_HIT | 0.50 | -1.19% |
| SELL | retest2 | 2025-07-24 15:15:00 | 559.00 | 2025-07-29 10:15:00 | 567.50 | STOP_HIT | 0.50 | -1.52% |
| SELL | retest2 | 2025-07-28 14:00:00 | 561.65 | 2025-07-29 13:15:00 | 573.05 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-07-28 15:15:00 | 561.75 | 2025-07-29 13:15:00 | 573.05 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-31 14:15:00 | 569.60 | 2025-08-06 15:15:00 | 541.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:30:00 | 568.10 | 2025-08-07 09:15:00 | 539.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 569.60 | 2025-08-21 10:15:00 | 554.30 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2025-08-05 09:30:00 | 568.10 | 2025-08-21 10:15:00 | 554.30 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-09-15 09:45:00 | 568.25 | 2025-09-15 10:15:00 | 589.00 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2025-11-10 09:15:00 | 633.30 | 2025-11-11 09:15:00 | 618.45 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-11-10 13:45:00 | 626.35 | 2025-11-11 09:15:00 | 618.45 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-10 14:45:00 | 627.20 | 2025-11-11 09:15:00 | 618.45 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-11-17 09:15:00 | 632.25 | 2025-11-18 09:15:00 | 621.35 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-18 12:15:00 | 623.55 | 2025-11-19 09:15:00 | 617.95 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-18 13:45:00 | 623.80 | 2025-11-19 09:15:00 | 617.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-11-20 09:30:00 | 625.10 | 2025-11-20 11:15:00 | 619.75 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-13 11:30:00 | 561.75 | 2026-01-20 09:15:00 | 533.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 560.20 | 2026-01-20 09:15:00 | 532.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 560.45 | 2026-01-20 09:15:00 | 532.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 559.80 | 2026-01-20 09:15:00 | 531.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:30:00 | 561.75 | 2026-01-23 13:15:00 | 505.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 560.20 | 2026-01-23 13:15:00 | 504.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 560.45 | 2026-01-23 13:15:00 | 504.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 559.80 | 2026-01-23 13:15:00 | 503.82 | TARGET_HIT | 0.50 | 10.00% |
