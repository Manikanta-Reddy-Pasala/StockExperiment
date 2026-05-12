# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 598.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 23
- **Target hits / Stop hits / Partials:** 1 / 30 / 8
- **Avg / median % per leg:** 0.21% / -1.09%
- **Sum % (uncompounded):** 8.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 16 | 41.0% | 1 | 30 | 8 | 0.21% | 8.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 16 | 41.0% | 1 | 30 | 8 | 0.21% | 8.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 16 | 41.0% | 1 | 30 | 8 | 0.21% | 8.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 585.00 | 562.00 | 561.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 602.40 | 572.07 | 567.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 632.75 | 632.84 | 612.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:00:00 | 632.75 | 632.84 | 612.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 613.25 | 632.09 | 613.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 612.45 | 632.09 | 613.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 611.15 | 631.88 | 613.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 610.75 | 631.88 | 613.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 608.80 | 631.65 | 613.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 609.80 | 631.65 | 613.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 615.05 | 631.10 | 613.16 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 528.90 | 602.35 | 602.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 516.30 | 597.94 | 600.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 537.30 | 536.57 | 557.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 537.30 | 536.57 | 557.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 535.00 | 526.17 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 535.00 | 526.17 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 535.85 | 526.26 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 535.70 | 526.26 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 537.95 | 526.38 | 536.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 539.55 | 526.38 | 536.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 532.90 | 526.44 | 536.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 529.10 | 526.69 | 536.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:45:00 | 529.15 | 528.62 | 536.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 553.50 | 527.58 | 534.96 | SL hit (close>static) qty=1.00 sl=543.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 14:15:00 | 570.25 | 486.80 | 486.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 577.90 | 488.53 | 487.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 10:15:00 | 602.05 | 2025-05-21 09:15:00 | 571.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-15 10:15:00 | 602.05 | 2025-05-21 09:15:00 | 573.30 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2025-05-16 09:45:00 | 602.70 | 2025-05-21 09:15:00 | 572.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-16 09:45:00 | 602.70 | 2025-05-21 09:15:00 | 573.30 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2025-05-19 09:30:00 | 603.10 | 2025-05-21 09:15:00 | 572.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 09:30:00 | 603.10 | 2025-05-21 09:15:00 | 573.30 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-10-30 13:30:00 | 529.10 | 2025-11-11 09:15:00 | 553.50 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-11-06 09:45:00 | 529.15 | 2025-11-11 09:15:00 | 553.50 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-11-12 09:15:00 | 528.40 | 2025-11-12 11:15:00 | 543.50 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-12-01 09:45:00 | 528.35 | 2025-12-03 14:15:00 | 546.50 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-12-10 14:15:00 | 533.70 | 2025-12-15 15:15:00 | 539.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-16 09:15:00 | 529.40 | 2025-12-23 09:15:00 | 543.15 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-12-22 11:45:00 | 535.55 | 2025-12-23 09:15:00 | 543.15 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-12-22 12:15:00 | 535.80 | 2025-12-23 09:15:00 | 543.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-12-23 11:15:00 | 537.90 | 2025-12-24 10:15:00 | 545.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-23 12:45:00 | 537.30 | 2025-12-24 10:15:00 | 545.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-23 14:15:00 | 537.95 | 2025-12-24 10:15:00 | 545.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-12-30 09:45:00 | 535.80 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-31 13:00:00 | 537.00 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-01 09:15:00 | 537.05 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-05 09:15:00 | 536.10 | 2026-01-06 10:15:00 | 541.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-05 11:15:00 | 537.25 | 2026-01-12 09:15:00 | 509.01 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2026-01-05 11:15:00 | 537.25 | 2026-01-19 13:15:00 | 482.22 | TARGET_HIT | 0.50 | 10.24% |
| SELL | retest2 | 2026-02-04 11:45:00 | 511.95 | 2026-02-11 11:15:00 | 486.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 13:00:00 | 511.50 | 2026-02-11 11:15:00 | 485.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 13:45:00 | 511.00 | 2026-02-11 11:15:00 | 485.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 09:45:00 | 508.00 | 2026-02-12 09:15:00 | 482.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 11:45:00 | 511.95 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-02-09 13:00:00 | 511.50 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-02-09 13:45:00 | 511.00 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-02-10 09:45:00 | 508.00 | 2026-02-26 14:15:00 | 496.40 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-04-09 13:00:00 | 457.60 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.38% |
| SELL | retest2 | 2026-04-10 14:15:00 | 458.30 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.22% |
| SELL | retest2 | 2026-04-10 15:15:00 | 459.10 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.03% |
| SELL | retest2 | 2026-04-13 12:00:00 | 458.85 | 2026-04-15 09:15:00 | 486.80 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2026-04-15 12:30:00 | 473.05 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-16 11:45:00 | 474.15 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-04-16 13:15:00 | 475.45 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-04-16 14:15:00 | 475.00 | 2026-04-17 09:15:00 | 489.75 | STOP_HIT | 1.00 | -3.11% |
