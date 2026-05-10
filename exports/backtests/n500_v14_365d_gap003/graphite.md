# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 752.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.78% / -0.93%
- **Sum % (uncompounded):** -7.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.78% | -7.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.78% | -7.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.78% | -7.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 556.20 | 468.36 | 468.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 11:15:00 | 566.45 | 469.34 | 468.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 525.00 | 527.16 | 506.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 09:30:00 | 522.70 | 527.16 | 506.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 545.25 | 563.12 | 545.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 545.25 | 563.12 | 545.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 544.40 | 562.93 | 545.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 547.50 | 562.31 | 544.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 543.00 | 561.76 | 545.04 | SL hit (close<static) qty=1.00 sl=543.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:45:00 | 549.05 | 560.35 | 544.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 527.10 | 561.58 | 547.21 | SL hit (close<static) qty=1.00 sl=543.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 550.25 | 546.89 | 542.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 15:15:00 | 542.75 | 547.40 | 543.31 | SL hit (close<static) qty=1.00 sl=543.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 547.85 | 547.40 | 543.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 542.75 | 547.37 | 543.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 12:15:00 | 542.75 | 547.37 | 543.38 | SL hit (close<static) qty=1.00 sl=543.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 543.30 | 547.37 | 543.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 541.65 | 547.32 | 543.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 543.20 | 547.32 | 543.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 541.15 | 547.26 | 543.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 541.15 | 547.26 | 543.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 540.20 | 547.19 | 543.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 538.10 | 547.19 | 543.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 536.00 | 546.98 | 543.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 536.35 | 546.98 | 543.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| CROSSOVER_SKIP | 2025-08-29 13:15:00 | 510.60 | 539.93 | 540.07 | min_gap filter: gap=0.028% < 0.030% |
| TREND_RESET | 2025-08-29 13:15:00 | 510.60 | 539.93 | 540.07 | EMA inversion without crossover edge (EMA200=539.93 EMA400=540.07) — end cycle |
| CROSSOVER_SKIP | 2025-09-19 13:15:00 | 576.00 | 537.64 | 537.64 | min_gap filter: gap=0.001% < 0.030% |
| CROSSOVER_SKIP | 2025-12-05 14:15:00 | 537.65 | 561.50 | 561.53 | min_gap filter: gap=0.007% < 0.030% |
| CROSSOVER_SKIP | 2025-12-29 15:15:00 | 598.00 | 558.62 | 558.56 | min_gap filter: gap=0.010% < 0.030% |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-28 09:15:00 | 547.50 | 2025-07-28 12:15:00 | 543.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-29 13:45:00 | 549.05 | 2025-08-01 14:15:00 | 527.10 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-08-18 14:15:00 | 550.25 | 2025-08-20 15:15:00 | 542.75 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-21 09:15:00 | 547.85 | 2025-08-21 12:15:00 | 542.75 | STOP_HIT | 1.00 | -0.93% |
