# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 579.65
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 19
- **Target hits / Stop hits / Partials:** 2 / 19 / 0
- **Avg / median % per leg:** -1.29% / -2.08%
- **Sum % (uncompounded):** -27.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 2 | 10.0% | 2 | 18 | 0 | -1.22% | -24.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 2 | 18 | 0 | -1.22% | -24.3% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| retest2 (combined) | 20 | 2 | 10.0% | 2 | 18 | 0 | -1.22% | -24.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 482.00 | 450.23 | 450.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 484.15 | 451.53 | 450.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 489.30 | 490.69 | 477.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 489.30 | 490.69 | 477.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 480.50 | 490.55 | 478.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 476.95 | 490.55 | 478.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 479.15 | 490.22 | 478.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:45:00 | 478.45 | 490.22 | 478.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 480.00 | 490.12 | 478.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 480.25 | 490.12 | 478.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 478.95 | 490.01 | 478.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 485.00 | 489.01 | 478.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 486.85 | 488.87 | 478.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 486.00 | 488.83 | 478.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:00:00 | 486.50 | 488.86 | 479.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 483.90 | 488.78 | 481.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.55 | SL hit (close<static) qty=1.00 sl=477.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.55 | SL hit (close<static) qty=1.00 sl=477.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.55 | SL hit (close<static) qty=1.00 sl=477.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.55 | SL hit (close<static) qty=1.00 sl=477.70 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 487.90 | 485.69 | 481.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 488.40 | 485.85 | 481.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 479.95 | 485.76 | 481.32 | SL hit (close<static) qty=1.00 sl=481.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 479.95 | 485.76 | 481.32 | SL hit (close<static) qty=1.00 sl=481.50 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 455.30 | 477.91 | 477.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 452.40 | 475.45 | 476.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 470.00 | 469.69 | 473.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:15:00 | 464.30 | 469.69 | 473.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 470.80 | 469.06 | 472.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 477.45 | 469.14 | 472.99 | SL hit (close>ema400) qty=1.00 sl=472.99 alert=retest1 |

### Cycle 3 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 500.00 | 476.28 | 476.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 528.40 | 477.03 | 476.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 513.15 | 517.45 | 503.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 513.15 | 517.45 | 503.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 503.60 | 517.17 | 503.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:45:00 | 498.65 | 517.17 | 503.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 503.45 | 517.03 | 503.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 503.65 | 517.03 | 503.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 500.10 | 516.87 | 503.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 500.35 | 516.87 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 501.25 | 516.44 | 503.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:30:00 | 502.70 | 516.44 | 503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 499.00 | 516.27 | 503.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 510.00 | 516.27 | 503.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:00:00 | 503.10 | 515.98 | 503.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 497.65 | 515.56 | 504.06 | SL hit (close<static) qty=1.00 sl=499.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 497.65 | 515.56 | 504.06 | SL hit (close<static) qty=1.00 sl=499.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 11:30:00 | 502.30 | 515.25 | 504.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 13:15:00 | 497.00 | 514.91 | 503.95 | SL hit (close<static) qty=1.00 sl=499.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 508.30 | 514.62 | 503.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 509.25 | 514.52 | 504.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 504.20 | 514.52 | 504.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 506.05 | 514.76 | 504.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 505.75 | 514.76 | 504.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 509.85 | 514.71 | 504.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 512.30 | 514.65 | 504.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 511.40 | 514.62 | 504.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 510.80 | 514.55 | 504.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 501.65 | 514.38 | 504.91 | SL hit (close<static) qty=1.00 sl=502.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 501.65 | 514.38 | 504.91 | SL hit (close<static) qty=1.00 sl=502.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 501.65 | 514.38 | 504.91 | SL hit (close<static) qty=1.00 sl=502.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 490.90 | 513.89 | 504.81 | SL hit (close<static) qty=1.00 sl=499.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 515.80 | 510.50 | 503.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 504.40 | 510.99 | 504.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 502.25 | 510.76 | 504.27 | SL hit (close<static) qty=1.00 sl=502.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 509.90 | 510.23 | 504.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 507.60 | 510.19 | 504.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 509.80 | 510.06 | 504.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 486.40 | 509.82 | 504.13 | SL hit (close<static) qty=1.00 sl=496.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 486.40 | 509.82 | 504.13 | SL hit (close<static) qty=1.00 sl=496.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 486.40 | 509.82 | 504.13 | SL hit (close<static) qty=1.00 sl=496.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 507.85 | 507.25 | 503.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 498.60 | 507.68 | 503.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 498.60 | 507.68 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 492.95 | 507.53 | 503.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-27 10:15:00 | 492.95 | 507.53 | 503.49 | SL hit (close<static) qty=1.00 sl=496.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 492.50 | 507.53 | 503.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 498.90 | 504.94 | 502.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 498.90 | 504.94 | 502.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 501.40 | 504.91 | 502.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:45:00 | 497.75 | 504.91 | 502.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 500.95 | 504.86 | 502.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 499.15 | 504.86 | 502.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 497.35 | 504.79 | 502.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 495.30 | 504.79 | 502.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 503.25 | 503.91 | 502.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 507.50 | 503.91 | 502.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:00:00 | 508.55 | 504.23 | 502.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 13:15:00 | 558.25 | 510.29 | 505.69 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-10 13:15:00 | 559.41 | 510.29 | 505.69 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-11 10:15:00 | 485.00 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-12-12 09:15:00 | 486.85 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-12-12 10:15:00 | 486.00 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-12-15 11:00:00 | 486.50 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-01-02 10:45:00 | 487.90 | 2026-01-05 13:15:00 | 479.95 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-01-05 10:30:00 | 488.40 | 2026-01-05 13:15:00 | 479.95 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest1 | 2026-01-23 09:15:00 | 464.30 | 2026-01-27 10:15:00 | 477.45 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-05 09:15:00 | 510.00 | 2026-03-09 09:15:00 | 497.65 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-03-05 12:00:00 | 503.10 | 2026-03-09 09:15:00 | 497.65 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-03-09 11:30:00 | 502.30 | 2026-03-09 13:15:00 | 497.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-03-10 09:15:00 | 508.30 | 2026-03-13 09:15:00 | 501.65 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-12 12:15:00 | 512.30 | 2026-03-13 09:15:00 | 501.65 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-03-12 13:00:00 | 511.40 | 2026-03-13 09:15:00 | 501.65 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-12 15:00:00 | 510.80 | 2026-03-13 12:15:00 | 490.90 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2026-03-18 09:15:00 | 515.80 | 2026-03-19 12:15:00 | 502.25 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-03-20 11:15:00 | 509.90 | 2026-03-23 09:15:00 | 486.40 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2026-03-20 12:15:00 | 507.60 | 2026-03-23 09:15:00 | 486.40 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-20 15:15:00 | 509.80 | 2026-03-23 09:15:00 | 486.40 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2026-03-25 09:15:00 | 507.85 | 2026-03-27 10:15:00 | 492.95 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-04-06 11:15:00 | 507.50 | 2026-04-10 13:15:00 | 558.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:00:00 | 508.55 | 2026-04-10 13:15:00 | 559.41 | TARGET_HIT | 1.00 | 10.00% |
