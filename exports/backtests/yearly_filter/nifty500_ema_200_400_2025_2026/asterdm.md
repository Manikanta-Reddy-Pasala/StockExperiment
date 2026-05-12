# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 742.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 0
- **Target hits / Stop hits / Partials:** 4 / 0 / 0
- **Avg / median % per leg:** 10.00% / 10.00%
- **Sum % (uncompounded):** 40.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 10:15:00 | 621.85 | 656.02 | 656.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 610.30 | 653.87 | 654.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 625.35 | 623.60 | 635.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:30:00 | 625.00 | 623.60 | 635.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 606.80 | 575.96 | 596.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 606.80 | 575.96 | 596.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 608.50 | 576.29 | 596.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 608.50 | 576.29 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 600.00 | 577.55 | 596.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 604.95 | 577.55 | 596.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 597.10 | 579.12 | 597.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:30:00 | 599.30 | 579.12 | 597.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 599.55 | 579.32 | 597.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:45:00 | 599.70 | 579.32 | 597.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 600.05 | 579.53 | 597.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 599.95 | 579.53 | 597.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 614.85 | 580.78 | 597.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 614.85 | 580.78 | 597.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 654.70 | 608.75 | 608.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 11:15:00 | 662.90 | 615.07 | 611.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 639.45 | 641.15 | 628.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 639.45 | 641.15 | 628.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 625.40 | 640.64 | 628.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 625.40 | 640.64 | 628.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 627.85 | 640.51 | 628.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 627.85 | 640.51 | 628.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 632.00 | 640.43 | 628.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 617.65 | 640.43 | 628.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 614.40 | 640.17 | 628.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:45:00 | 612.95 | 640.17 | 628.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 632.00 | 638.59 | 628.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:30:00 | 624.80 | 638.59 | 628.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 641.60 | 638.53 | 628.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 644.95 | 638.53 | 628.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-30 15:15:00 | 709.45 | 641.79 | 631.11 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 09:15:00 | 582.05 | 2025-09-02 15:15:00 | 640.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 15:00:00 | 580.30 | 2025-09-02 15:15:00 | 638.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 10:00:00 | 578.75 | 2025-09-02 15:15:00 | 636.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-25 10:15:00 | 644.95 | 2026-03-30 15:15:00 | 709.45 | TARGET_HIT | 1.00 | 10.00% |
