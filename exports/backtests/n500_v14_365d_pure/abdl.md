# Allied Blenders and Distillers Ltd. (ABDL)

## Backtest Summary

- **Window:** 2024-07-02 09:15:00 → 2026-05-08 15:15:00 (3203 bars)
- **Last close:** 594.00
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
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 2
- **Avg / median % per leg:** -0.18% / -1.21%
- **Sum % (uncompounded):** -2.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.95% | -3.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.95% | -3.9% |
| SELL (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | 0.17% | 1.7% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.48% | -25.9% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.61% | 27.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.48% | -25.9% |
| retest2 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.97% | 23.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 401.00 | 335.89 | 335.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 405.00 | 342.92 | 339.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 498.35 | 498.73 | 473.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:45:00 | 498.15 | 498.73 | 473.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 603.20 | 627.03 | 599.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 597.45 | 627.03 | 599.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 603.00 | 624.34 | 604.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 603.00 | 624.34 | 604.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 607.75 | 624.17 | 604.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 611.10 | 624.17 | 604.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 601.25 | 623.17 | 607.50 | SL hit (close<static) qty=1.00 sl=602.50 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 614.70 | 618.91 | 606.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 600.60 | 618.35 | 606.71 | SL hit (close<static) qty=1.00 sl=602.50 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 507.60 | 597.29 | 597.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 504.45 | 596.37 | 597.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 510.80 | 509.40 | 541.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:45:00 | 503.30 | 509.41 | 540.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 12:45:00 | 503.75 | 509.35 | 539.82 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:30:00 | 503.30 | 509.31 | 539.20 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:00:00 | 504.55 | 509.45 | 537.24 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | SL hit (close>ema400) qty=1.00 sl=534.69 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | SL hit (close>ema400) qty=1.00 sl=534.69 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | SL hit (close>ema400) qty=1.00 sl=534.69 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 536.35 | 509.87 | 534.69 | SL hit (close>ema400) qty=1.00 sl=534.69 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 536.35 | 509.87 | 534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 534.10 | 510.12 | 534.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 535.80 | 510.12 | 534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 538.00 | 510.39 | 534.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 538.00 | 510.39 | 534.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 535.40 | 510.64 | 534.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 529.30 | 511.43 | 534.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 532.45 | 512.42 | 534.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 15:15:00 | 502.83 | 514.82 | 532.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 15:15:00 | 505.83 | 514.82 | 532.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-27 09:15:00 | 479.21 | 508.59 | 526.58 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-27 11:15:00 | 476.37 | 507.97 | 526.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:15:00 | 533.00 | 457.15 | 474.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:45:00 | 533.40 | 457.90 | 475.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 539.45 | 459.43 | 475.79 | SL hit (close>static) qty=1.00 sl=539.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 539.45 | 459.43 | 475.79 | SL hit (close>static) qty=1.00 sl=539.10 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 564.80 | 489.93 | 489.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 572.00 | 514.31 | 503.91 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-18 11:15:00 | 611.10 | 2025-12-29 09:15:00 | 601.25 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-12-31 15:00:00 | 614.70 | 2026-01-01 12:15:00 | 600.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2026-02-04 11:45:00 | 503.30 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.57% |
| SELL | retest1 | 2026-02-04 12:45:00 | 503.75 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.47% |
| SELL | retest1 | 2026-02-05 09:30:00 | 503.30 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.57% |
| SELL | retest1 | 2026-02-09 10:00:00 | 504.55 | 2026-02-12 09:15:00 | 536.35 | STOP_HIT | 1.00 | -6.30% |
| SELL | retest2 | 2026-02-13 09:15:00 | 529.30 | 2026-02-19 15:15:00 | 502.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 14:00:00 | 532.45 | 2026-02-19 15:15:00 | 505.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 529.30 | 2026-02-27 09:15:00 | 479.21 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2026-02-13 14:00:00 | 532.45 | 2026-02-27 11:15:00 | 476.37 | TARGET_HIT | 0.50 | 10.53% |
| SELL | retest2 | 2026-04-16 12:15:00 | 533.00 | 2026-04-16 14:15:00 | 539.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-16 12:45:00 | 533.40 | 2026-04-16 14:15:00 | 539.45 | STOP_HIT | 1.00 | -1.13% |
