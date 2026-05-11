# Patanjali Foods Ltd. (PATANJALI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 459.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 7
- **Target hits / Stop hits / Partials:** 7 / 7 / 7
- **Avg / median % per leg:** 4.30% / 5.00%
- **Sum % (uncompounded):** 90.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.07% | -9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.07% | -9.2% |
| SELL (all) | 18 | 14 | 77.8% | 7 | 4 | 7 | 5.53% | 99.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 14 | 77.8% | 7 | 4 | 7 | 5.53% | 99.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 14 | 66.7% | 7 | 7 | 7 | 4.30% | 90.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 572.37 | 608.86 | 608.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 554.83 | 594.44 | 600.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 556.17 | 556.13 | 569.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:30:00 | 556.67 | 556.13 | 569.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 568.33 | 556.28 | 568.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 568.33 | 556.28 | 568.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 571.63 | 556.43 | 568.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 571.63 | 556.43 | 568.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 570.23 | 556.57 | 568.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:15:00 | 568.30 | 556.57 | 568.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 580.63 | 556.94 | 568.36 | SL hit (close>static) qty=1.00 sl=572.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 650.53 | 578.28 | 578.05 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 584.00 | 595.72 | 595.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 15:15:00 | 579.00 | 594.57 | 595.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 595.85 | 593.00 | 594.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 595.85 | 593.00 | 594.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 596.85 | 593.03 | 594.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 593.05 | 593.12 | 594.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 590.80 | 593.14 | 594.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 591.95 | 592.18 | 593.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 598.45 | 592.18 | 593.61 | SL hit (close>static) qty=1.00 sl=598.35 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 10:45:00 | 608.70 | 2025-05-16 09:15:00 | 590.07 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-05-12 11:30:00 | 608.33 | 2025-05-16 09:15:00 | 590.07 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-05-13 10:00:00 | 609.20 | 2025-05-16 09:15:00 | 590.07 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-07-15 13:15:00 | 568.30 | 2025-07-15 14:15:00 | 580.63 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-10-23 13:45:00 | 593.05 | 2025-10-29 09:15:00 | 598.45 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-24 09:15:00 | 590.80 | 2025-10-29 09:15:00 | 598.45 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-28 10:00:00 | 591.95 | 2025-10-29 09:15:00 | 598.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-03 09:15:00 | 583.55 | 2025-12-03 14:15:00 | 554.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 580.85 | 2025-12-03 14:15:00 | 551.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 580.50 | 2025-12-03 14:15:00 | 552.71 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2025-11-20 13:45:00 | 581.80 | 2025-12-03 14:15:00 | 552.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 581.80 | 2025-12-04 11:15:00 | 551.48 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2025-11-03 09:15:00 | 583.55 | 2025-12-04 15:15:00 | 525.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 580.85 | 2025-12-04 15:15:00 | 522.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 580.50 | 2025-12-04 15:15:00 | 522.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 13:45:00 | 581.80 | 2025-12-04 15:15:00 | 523.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 581.80 | 2025-12-04 15:15:00 | 523.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 565.60 | 2026-01-16 10:15:00 | 537.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 564.05 | 2026-01-16 12:15:00 | 535.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 565.60 | 2026-01-20 11:15:00 | 509.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 564.05 | 2026-01-20 12:15:00 | 507.64 | TARGET_HIT | 0.50 | 10.00% |
