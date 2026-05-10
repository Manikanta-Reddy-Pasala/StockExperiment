# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
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
| ALERT2_SKIP | 0 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 27
- **Target hits / Stop hits / Partials:** 0 / 30 / 2
- **Avg / median % per leg:** -2.17% / -2.58%
- **Sum % (uncompounded):** -69.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 5 | 15.6% | 0 | 30 | 2 | -2.17% | -69.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 5 | 15.6% | 0 | 30 | 2 | -2.17% | -69.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 5 | 15.6% | 0 | 30 | 2 | -2.17% | -69.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 423.70 | 401.58 | 401.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 10:15:00 | 434.45 | 407.61 | 405.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 421.75 | 422.54 | 415.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:45:00 | 422.70 | 422.54 | 415.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 414.80 | 422.38 | 415.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 414.80 | 422.38 | 415.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 411.45 | 422.28 | 415.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 411.45 | 422.28 | 415.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 414.20 | 419.54 | 414.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 400.65 | 419.54 | 414.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 385.00 | 371.12 | 381.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 384.45 | 372.94 | 381.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 413.20 | 374.31 | 382.06 | SL hit (close>static) qty=1.00 sl=394.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 413.20 | 374.31 | 382.06 | SL hit (close>static) qty=1.00 sl=394.70 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:00:00 | 385.05 | 377.31 | 383.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 389.25 | 377.43 | 383.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 389.25 | 377.43 | 383.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 395.00 | 378.08 | 383.32 | SL hit (close>static) qty=1.00 sl=394.70 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 375.25 | 367.52 | 375.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 379.60 | 367.52 | 375.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 373.85 | 367.59 | 375.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 374.30 | 367.59 | 375.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 373.95 | 367.65 | 375.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 373.85 | 367.65 | 375.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 376.50 | 367.74 | 375.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 376.50 | 367.74 | 375.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 370.50 | 367.76 | 375.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 367.20 | 367.79 | 375.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 367.55 | 367.79 | 375.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 368.85 | 368.29 | 375.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:30:00 | 369.15 | 368.29 | 375.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 369.50 | 367.33 | 374.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:00:00 | 366.55 | 367.33 | 374.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 366.60 | 367.32 | 374.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 366.55 | 367.46 | 374.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 364.10 | 367.39 | 373.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=376.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=376.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=376.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=376.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=375.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 378.00 | 367.00 | 373.26 | SL hit (close>static) qty=1.00 sl=375.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 381.65 | 367.00 | 373.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 384.55 | 367.18 | 373.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 384.55 | 367.18 | 373.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 371.80 | 367.82 | 373.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 378.15 | 367.82 | 373.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 379.00 | 367.93 | 373.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 384.00 | 367.93 | 373.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 377.75 | 368.03 | 373.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 374.25 | 368.30 | 373.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 376.75 | 369.14 | 372.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 383.35 | 369.73 | 372.71 | SL hit (close>static) qty=1.00 sl=379.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 383.35 | 369.73 | 372.71 | SL hit (close>static) qty=1.00 sl=379.75 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 369.30 | 372.52 | 373.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 13:15:00 | 350.83 | 365.98 | 369.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 368.40 | 365.61 | 369.40 | SL hit (close>ema200) qty=0.50 sl=365.61 alert=retest2 |


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
