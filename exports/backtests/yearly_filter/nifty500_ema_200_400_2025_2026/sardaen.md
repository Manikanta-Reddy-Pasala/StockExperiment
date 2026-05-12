# Sarda Energy and Minerals Ltd. (SARDAEN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 591.90
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
| ALERT2_SKIP | 1 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 25
- **Target hits / Stop hits / Partials:** 1 / 26 / 2
- **Avg / median % per leg:** -1.75% / -1.95%
- **Sum % (uncompounded):** -50.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 1 | 6.7% | 1 | 14 | 0 | -1.55% | -23.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 1 | 14 | 0 | -1.55% | -23.2% |
| SELL (all) | 14 | 3 | 21.4% | 0 | 12 | 2 | -1.96% | -27.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 12 | 2 | -1.96% | -27.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 4 | 13.8% | 1 | 26 | 2 | -1.75% | -50.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 541.20 | 451.02 | 450.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 549.90 | 455.48 | 453.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 568.80 | 571.98 | 539.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 568.80 | 571.98 | 539.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 548.50 | 573.69 | 547.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 548.50 | 573.69 | 547.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 547.30 | 573.43 | 547.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:45:00 | 547.30 | 573.43 | 547.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 544.90 | 573.14 | 547.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 544.90 | 573.14 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 540.60 | 572.82 | 547.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 540.60 | 572.82 | 547.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 542.40 | 571.60 | 547.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 542.40 | 571.60 | 547.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 550.80 | 570.62 | 547.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 554.15 | 570.42 | 547.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 546.40 | 569.78 | 552.33 | SL hit (close<static) qty=1.00 sl=546.65 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 518.75 | 544.58 | 544.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 515.95 | 544.30 | 544.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 507.85 | 502.32 | 516.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 11:30:00 | 511.20 | 502.32 | 516.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 506.50 | 502.88 | 516.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 500.55 | 503.83 | 516.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 500.75 | 503.83 | 516.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:00:00 | 501.40 | 503.76 | 515.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 517.80 | 504.21 | 515.76 | SL hit (close>static) qty=1.00 sl=517.30 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 534.00 | 505.57 | 505.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 536.95 | 506.99 | 506.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:15:00 | 510.45 | 524.56 | 516.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 508.30 | 524.40 | 516.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 508.05 | 524.40 | 516.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 505.70 | 524.21 | 516.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 505.70 | 524.21 | 516.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 519.30 | 523.10 | 516.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:30:00 | 522.55 | 523.09 | 516.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 510.75 | 523.10 | 516.92 | SL hit (close<static) qty=1.00 sl=515.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 09:15:00 | 460.95 | 2025-05-21 10:15:00 | 475.60 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-05-26 09:15:00 | 437.65 | 2025-07-14 15:15:00 | 437.38 | PARTIAL | 0.50 | 0.06% |
| SELL | retest2 | 2025-05-26 09:15:00 | 437.65 | 2025-07-15 09:15:00 | 442.20 | STOP_HIT | 0.50 | -1.04% |
| SELL | retest2 | 2025-07-11 10:00:00 | 460.40 | 2025-08-04 09:15:00 | 512.85 | STOP_HIT | 1.00 | -11.39% |
| BUY | retest2 | 2025-10-03 09:15:00 | 554.15 | 2025-10-14 09:15:00 | 546.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-10-16 09:30:00 | 553.00 | 2025-10-16 10:15:00 | 546.15 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-23 09:15:00 | 556.50 | 2025-10-23 09:15:00 | 543.05 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-10-29 09:15:00 | 558.00 | 2025-10-30 10:15:00 | 544.55 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-11-10 14:15:00 | 546.70 | 2025-11-11 09:15:00 | 534.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-11-10 15:00:00 | 543.20 | 2025-11-11 09:15:00 | 534.70 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-11-11 09:45:00 | 542.35 | 2025-11-11 10:15:00 | 531.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-12-17 10:15:00 | 500.55 | 2025-12-19 09:15:00 | 517.80 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-12-18 09:15:00 | 500.75 | 2025-12-19 09:15:00 | 517.80 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-12-18 11:00:00 | 501.40 | 2025-12-19 09:15:00 | 517.80 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2026-01-08 10:00:00 | 501.20 | 2026-01-09 15:15:00 | 476.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 501.20 | 2026-01-30 15:15:00 | 492.00 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2026-02-09 09:45:00 | 507.05 | 2026-02-09 10:15:00 | 513.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-02-11 09:15:00 | 503.90 | 2026-02-16 12:15:00 | 504.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-02-12 12:30:00 | 506.40 | 2026-02-17 13:15:00 | 515.05 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-12 14:15:00 | 507.80 | 2026-02-17 13:15:00 | 515.05 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-16 09:15:00 | 494.40 | 2026-02-17 13:15:00 | 515.05 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-17 14:30:00 | 522.55 | 2026-03-19 13:15:00 | 510.75 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-20 09:15:00 | 522.90 | 2026-03-20 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-20 10:30:00 | 521.55 | 2026-03-20 13:15:00 | 513.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-03-25 09:15:00 | 522.60 | 2026-03-27 09:15:00 | 510.45 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-04-01 09:15:00 | 529.60 | 2026-04-02 09:15:00 | 506.15 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2026-04-01 12:00:00 | 526.95 | 2026-04-02 09:15:00 | 506.15 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2026-04-01 15:00:00 | 526.30 | 2026-04-02 09:15:00 | 506.15 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-04-07 09:15:00 | 527.80 | 2026-04-13 10:15:00 | 580.58 | TARGET_HIT | 1.00 | 10.00% |
