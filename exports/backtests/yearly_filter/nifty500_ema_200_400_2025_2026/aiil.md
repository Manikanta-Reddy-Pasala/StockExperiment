# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2024-04-23 09:15:00 → 2026-05-11 15:15:00 (3542 bars)
- **Last close:** 504.90
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
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 7
- **Target hits / Stop hits / Partials:** 6 / 8 / 2
- **Avg / median % per leg:** 2.01% / 5.00%
- **Sum % (uncompounded):** 32.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 5 | 2 | 0 | 6.33% | 44.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 5 | 71.4% | 5 | 2 | 0 | 6.33% | 44.3% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 6 | 2 | -1.35% | -12.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 1 | 6 | 2 | -1.35% | -12.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 9 | 56.2% | 6 | 8 | 2 | 2.01% | 32.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 11:15:00 | 553.84 | 595.90 | 596.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 544.02 | 590.05 | 592.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 09:15:00 | 543.74 | 541.49 | 559.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:45:00 | 545.74 | 541.49 | 559.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 560.56 | 541.88 | 559.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 557.44 | 541.88 | 559.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 566.20 | 542.12 | 559.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 566.20 | 542.12 | 559.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 562.94 | 542.33 | 559.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 559.62 | 542.67 | 559.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:45:00 | 557.72 | 543.20 | 559.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 572.60 | 544.12 | 558.92 | SL hit (close>static) qty=1.00 sl=566.40 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 609.38 | 568.74 | 568.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 618.60 | 569.62 | 569.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-21 09:15:00 | 576.30 | 600.72 | 587.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 576.30 | 600.72 | 587.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 576.30 | 600.72 | 587.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 576.30 | 600.72 | 587.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 573.90 | 600.45 | 587.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 568.30 | 600.45 | 587.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 513.90 | 577.42 | 577.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 507.50 | 574.85 | 576.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 481.45 | 480.51 | 508.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 481.45 | 480.51 | 508.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 479.85 | 481.25 | 505.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 10:30:00 | 472.75 | 481.30 | 505.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 522.60 | 482.31 | 504.06 | SL hit (close>static) qty=1.00 sl=508.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 13:30:00 | 556.10 | 2025-08-25 09:15:00 | 609.69 | TARGET_HIT | 1.00 | 9.64% |
| BUY | retest2 | 2025-07-29 14:15:00 | 554.26 | 2025-08-25 14:15:00 | 610.94 | TARGET_HIT | 1.00 | 10.23% |
| BUY | retest2 | 2025-07-31 12:45:00 | 555.40 | 2025-08-26 09:15:00 | 611.71 | TARGET_HIT | 1.00 | 10.14% |
| BUY | retest2 | 2025-07-31 14:45:00 | 557.80 | 2025-08-26 09:15:00 | 613.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 13:30:00 | 572.60 | 2025-08-26 15:15:00 | 629.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-04 12:15:00 | 569.10 | 2025-11-12 11:15:00 | 553.84 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-11-04 14:45:00 | 570.96 | 2025-11-12 11:15:00 | 553.84 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-12-17 09:45:00 | 559.62 | 2025-12-19 13:15:00 | 572.60 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-12-17 12:45:00 | 557.72 | 2025-12-19 13:15:00 | 572.60 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-23 10:30:00 | 472.75 | 2026-03-25 09:15:00 | 522.60 | STOP_HIT | 1.00 | -10.54% |
| SELL | retest2 | 2026-03-27 15:15:00 | 473.40 | 2026-03-30 09:15:00 | 449.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 15:15:00 | 473.40 | 2026-03-30 15:15:00 | 426.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-20 13:15:00 | 473.85 | 2026-04-28 09:15:00 | 514.65 | STOP_HIT | 1.00 | -8.61% |
| SELL | retest2 | 2026-04-21 14:15:00 | 473.15 | 2026-04-28 09:15:00 | 514.65 | STOP_HIT | 1.00 | -8.77% |
| SELL | retest2 | 2026-04-30 09:45:00 | 476.65 | 2026-05-05 09:15:00 | 452.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-30 09:45:00 | 476.65 | 2026-05-06 14:15:00 | 472.85 | STOP_HIT | 0.50 | 0.80% |
