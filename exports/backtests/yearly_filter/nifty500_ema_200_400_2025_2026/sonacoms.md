# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 579.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 25 |
| PARTIAL | 3 |
| TARGET_HIT | 5 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 21
- **Target hits / Stop hits / Partials:** 5 / 21 / 3
- **Avg / median % per leg:** 0.34% / -2.06%
- **Sum % (uncompounded):** 9.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 2 | 9.5% | 2 | 19 | 0 | -1.53% | -32.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 2 | 9.5% | 2 | 19 | 0 | -1.53% | -32.2% |
| SELL (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 5.24% | 41.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.40% | 44.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.83% | -2.8% |
| retest2 (combined) | 28 | 8 | 28.6% | 5 | 20 | 3 | 0.45% | 12.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 527.25 | 501.11 | 501.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 531.70 | 502.43 | 501.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 512.30 | 521.20 | 513.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 527.00 | 520.39 | 513.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 524.95 | 520.56 | 513.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 502.55 | 520.01 | 513.70 | SL hit (close<static) qty=1.00 sl=509.25 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 15:15:00 | 483.05 | 508.75 | 508.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 09:15:00 | 480.30 | 508.47 | 508.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 488.45 | 477.59 | 489.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 488.45 | 477.59 | 489.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 478.75 | 477.60 | 489.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 486.20 | 477.60 | 489.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 486.85 | 477.75 | 489.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 479.85 | 477.90 | 489.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 480.30 | 478.75 | 488.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:00:00 | 479.80 | 479.58 | 488.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:15:00 | 455.86 | 477.25 | 485.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:15:00 | 456.28 | 477.25 | 485.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 12:15:00 | 455.81 | 477.25 | 485.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-07 13:15:00 | 431.87 | 465.97 | 478.07 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 481.85 | 449.24 | 449.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 487.35 | 461.00 | 455.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 489.30 | 490.69 | 477.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 489.30 | 490.69 | 477.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 480.50 | 490.55 | 478.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 476.95 | 490.55 | 478.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 479.15 | 490.22 | 478.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:45:00 | 478.45 | 490.22 | 478.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 480.00 | 490.12 | 478.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 480.25 | 490.12 | 478.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 478.95 | 490.01 | 478.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 485.00 | 489.01 | 478.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 486.85 | 488.87 | 478.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 486.00 | 488.83 | 478.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:00:00 | 486.50 | 488.86 | 479.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 482.90 | 488.43 | 481.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 477.45 | 488.43 | 481.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 476.00 | 488.30 | 481.45 | SL hit (close<static) qty=1.00 sl=477.70 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 460.65 | 477.74 | 477.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 452.40 | 475.45 | 476.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 470.00 | 469.69 | 473.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:15:00 | 464.30 | 469.69 | 473.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 470.80 | 469.06 | 472.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 477.45 | 469.14 | 472.94 | SL hit (close>ema400) qty=1.00 sl=472.94 alert=retest1 |

### Cycle 5 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 500.00 | 476.28 | 476.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 528.40 | 477.03 | 476.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 513.15 | 517.45 | 503.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 513.15 | 517.45 | 503.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 503.60 | 517.17 | 503.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:45:00 | 498.65 | 517.17 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 503.45 | 517.03 | 503.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 503.65 | 517.03 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 500.10 | 516.87 | 503.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 500.35 | 516.87 | 503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 501.25 | 516.44 | 503.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:30:00 | 502.70 | 516.44 | 503.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 499.00 | 516.27 | 503.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 510.00 | 516.27 | 503.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 12:00:00 | 503.10 | 515.98 | 503.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 497.65 | 515.56 | 504.04 | SL hit (close<static) qty=1.00 sl=499.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-11 09:15:00 | 527.00 | 2025-06-13 09:15:00 | 502.55 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2025-06-12 11:00:00 | 524.95 | 2025-06-13 09:15:00 | 502.55 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-07-18 10:15:00 | 479.85 | 2025-07-30 12:15:00 | 455.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 480.30 | 2025-07-30 12:15:00 | 456.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 12:00:00 | 479.80 | 2025-07-30 12:15:00 | 455.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:15:00 | 479.85 | 2025-08-07 13:15:00 | 431.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 480.30 | 2025-08-07 13:15:00 | 432.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-25 12:00:00 | 479.80 | 2025-08-07 13:15:00 | 431.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:00:00 | 480.75 | 2025-10-29 09:15:00 | 481.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-12-11 10:15:00 | 485.00 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-12-12 09:15:00 | 486.85 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-12-12 10:15:00 | 486.00 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-12-15 11:00:00 | 486.50 | 2025-12-29 09:15:00 | 476.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-01-02 10:30:00 | 485.15 | 2026-01-06 09:15:00 | 474.10 | STOP_HIT | 1.00 | -2.28% |
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
