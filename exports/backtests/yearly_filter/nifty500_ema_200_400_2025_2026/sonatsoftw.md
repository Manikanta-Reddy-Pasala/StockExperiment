# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 296.65
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 31
- **Target hits / Stop hits / Partials:** 0 / 34 / 2
- **Avg / median % per leg:** -2.14% / -2.43%
- **Sum % (uncompounded):** -77.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.90% | -7.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.90% | -7.6% |
| SELL (all) | 32 | 5 | 15.6% | 0 | 30 | 2 | -2.17% | -69.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 5 | 15.6% | 0 | 30 | 2 | -2.17% | -69.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 5 | 13.9% | 0 | 34 | 2 | -2.14% | -77.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 423.35 | 401.36 | 401.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 425.95 | 403.58 | 402.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 404.85 | 406.06 | 403.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 404.85 | 406.06 | 403.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 404.85 | 406.06 | 403.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 404.85 | 406.06 | 403.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 406.00 | 406.06 | 403.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 404.70 | 406.06 | 403.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 403.95 | 406.03 | 403.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 403.95 | 406.03 | 403.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 400.45 | 405.98 | 403.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 406.80 | 405.98 | 403.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:00:00 | 405.30 | 405.97 | 403.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:30:00 | 404.40 | 405.94 | 403.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 13:45:00 | 404.85 | 405.91 | 403.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 397.65 | 405.82 | 403.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 397.65 | 405.82 | 403.90 | SL hit (close<static) qty=1.00 sl=399.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 356.95 | 410.35 | 410.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 354.00 | 408.25 | 409.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 384.20 | 380.61 | 391.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:45:00 | 382.55 | 380.61 | 391.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 392.85 | 380.73 | 391.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 395.75 | 380.73 | 391.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 387.30 | 380.79 | 391.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 384.30 | 380.87 | 391.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 13:15:00 | 365.08 | 380.31 | 391.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 377.50 | 370.47 | 382.23 | SL hit (close>ema200) qty=0.50 sl=370.47 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:45:00 | 395.75 | 2025-05-13 09:15:00 | 403.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-05-12 12:15:00 | 395.40 | 2025-05-13 09:15:00 | 403.05 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-05-12 13:00:00 | 396.30 | 2025-05-13 09:15:00 | 403.05 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-05-13 12:30:00 | 400.40 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-05-13 13:15:00 | 400.30 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-05-15 09:45:00 | 400.75 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-05-16 09:45:00 | 399.60 | 2025-05-23 10:15:00 | 400.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-05-19 11:15:00 | 394.30 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-05-19 11:45:00 | 393.65 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-05-20 10:15:00 | 393.30 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-05-20 10:45:00 | 393.75 | 2025-05-26 10:15:00 | 405.25 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-05-21 12:00:00 | 390.30 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2025-05-22 09:15:00 | 391.10 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-05-22 11:30:00 | 391.20 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2025-05-22 12:00:00 | 391.15 | 2025-05-30 15:15:00 | 409.35 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2025-06-20 09:15:00 | 406.80 | 2025-06-20 14:15:00 | 397.65 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-20 10:00:00 | 405.30 | 2025-06-20 14:15:00 | 397.65 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-06-20 12:30:00 | 404.40 | 2025-06-20 14:15:00 | 397.65 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-06-20 13:45:00 | 404.85 | 2025-06-20 14:15:00 | 397.65 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-25 14:00:00 | 384.30 | 2025-08-26 13:15:00 | 365.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:00:00 | 384.30 | 2025-09-10 09:15:00 | 377.50 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2025-09-12 10:15:00 | 385.00 | 2025-09-18 09:15:00 | 413.20 | STOP_HIT | 1.00 | -7.32% |
| SELL | retest2 | 2025-09-17 10:15:00 | 384.45 | 2025-09-18 09:15:00 | 413.20 | STOP_HIT | 1.00 | -7.48% |
| SELL | retest2 | 2025-09-22 10:00:00 | 385.05 | 2025-09-22 15:15:00 | 395.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-10-09 09:30:00 | 367.20 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-10-09 10:00:00 | 367.55 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-10-10 13:30:00 | 368.85 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-10-10 14:30:00 | 369.15 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-10-16 10:00:00 | 366.55 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-10-16 11:00:00 | 366.60 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-10-17 09:30:00 | 366.55 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-10-20 09:15:00 | 364.10 | 2025-10-23 09:15:00 | 378.00 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-10-24 13:30:00 | 374.25 | 2025-11-12 09:15:00 | 383.35 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-11-10 13:00:00 | 376.75 | 2025-11-12 09:15:00 | 383.35 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-11-14 09:15:00 | 369.30 | 2025-12-02 13:15:00 | 350.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:15:00 | 369.30 | 2025-12-03 10:15:00 | 368.40 | STOP_HIT | 0.50 | 0.24% |
