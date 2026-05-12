# Elecon Engineering Co. Ltd. (ELECON)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 562.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 1 / 9 / 2
- **Avg / median % per leg:** -0.14% / -0.73%
- **Sum % (uncompounded):** -1.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.15% | -2.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.15% | -2.2% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 8 | 2 | 0.04% | 0.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 4 | 36.4% | 1 | 8 | 2 | 0.04% | 0.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 4 | 33.3% | 1 | 9 | 2 | -0.14% | -1.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 578.90 | 614.33 | 614.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 570.90 | 613.55 | 613.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 581.05 | 580.80 | 593.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:00:00 | 581.05 | 580.80 | 593.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 592.30 | 567.91 | 580.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 594.40 | 567.91 | 580.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 605.45 | 568.29 | 580.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 605.45 | 568.29 | 580.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 605.00 | 589.65 | 589.60 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 575.35 | 589.46 | 589.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 569.50 | 589.26 | 589.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 587.70 | 585.08 | 587.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 582.60 | 585.08 | 587.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 584.20 | 585.07 | 587.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 583.85 | 585.07 | 587.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 585.55 | 585.04 | 587.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 585.55 | 585.04 | 587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 586.70 | 585.05 | 587.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:00:00 | 585.35 | 585.08 | 587.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 583.60 | 585.05 | 587.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 15:00:00 | 585.10 | 585.01 | 587.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 12:15:00 | 589.35 | 585.12 | 587.06 | SL hit (close>static) qty=1.00 sl=589.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 607.40 | 589.00 | 588.92 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 552.20 | 588.66 | 588.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 537.05 | 585.64 | 587.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 569.50 | 569.13 | 576.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 15:00:00 | 569.50 | 569.13 | 576.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 508.60 | 491.24 | 511.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 508.60 | 491.24 | 511.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 500.65 | 491.34 | 511.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 499.40 | 491.43 | 511.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 499.45 | 491.63 | 511.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 499.15 | 491.93 | 511.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 519.25 | 492.10 | 510.19 | SL hit (close>static) qty=1.00 sl=516.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 509.45 | 424.66 | 424.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 510.05 | 426.33 | 425.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-14 09:15:00 | 627.40 | 2025-07-16 09:15:00 | 613.90 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-10-03 10:00:00 | 585.35 | 2025-10-06 12:15:00 | 589.35 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-03 12:00:00 | 583.60 | 2025-10-06 12:15:00 | 589.35 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-03 15:00:00 | 585.10 | 2025-10-06 12:15:00 | 589.35 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-02 13:15:00 | 499.40 | 2026-01-07 11:15:00 | 519.25 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-01-02 14:45:00 | 499.45 | 2026-01-07 11:15:00 | 519.25 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-01-05 11:15:00 | 499.15 | 2026-01-07 11:15:00 | 519.25 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-01-09 09:15:00 | 437.50 | 2026-01-09 11:15:00 | 415.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 437.50 | 2026-01-14 11:15:00 | 393.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-05 09:15:00 | 446.00 | 2026-02-09 10:15:00 | 484.25 | STOP_HIT | 1.00 | -8.58% |
| SELL | retest2 | 2026-02-12 09:15:00 | 452.30 | 2026-02-16 09:15:00 | 429.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 452.30 | 2026-02-16 15:15:00 | 436.90 | STOP_HIT | 0.50 | 3.40% |
