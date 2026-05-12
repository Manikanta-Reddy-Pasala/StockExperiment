# PCBL Chemical Ltd. (PCBL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 306.00
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
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 5 / 16 / 5
- **Avg / median % per leg:** 0.89% / -1.15%
- **Sum % (uncompounded):** 23.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.88% | -23.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.88% | -23.0% |
| SELL (all) | 18 | 10 | 55.6% | 5 | 8 | 5 | 2.57% | 46.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 10 | 55.6% | 5 | 8 | 5 | 2.57% | 46.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 10 | 38.5% | 5 | 16 | 5 | 0.89% | 23.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 417.50 | 395.98 | 395.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 429.50 | 396.75 | 396.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 401.30 | 402.30 | 399.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 401.30 | 402.30 | 399.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 398.80 | 402.28 | 399.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 394.30 | 402.28 | 399.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 401.45 | 402.28 | 399.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 407.55 | 402.28 | 399.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 407.00 | 402.29 | 399.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 389.90 | 401.97 | 399.59 | SL hit (close<static) qty=1.00 sl=390.40 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 388.00 | 404.06 | 404.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 383.50 | 403.32 | 403.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 394.75 | 394.74 | 398.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:00:00 | 394.75 | 394.74 | 398.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 396.00 | 385.92 | 391.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:15:00 | 389.75 | 386.10 | 391.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 390.20 | 386.15 | 391.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 389.50 | 386.88 | 391.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 390.15 | 386.91 | 391.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 392.85 | 386.97 | 391.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 392.85 | 386.97 | 391.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 391.55 | 387.01 | 391.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 389.25 | 387.19 | 391.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 389.05 | 387.35 | 391.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 394.25 | 387.54 | 391.57 | SL hit (close>static) qty=1.00 sl=392.95 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 302.50 | 283.44 | 283.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 306.00 | 284.43 | 283.87 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:15:00 | 407.55 | 2025-06-17 12:15:00 | 389.90 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-06-13 12:15:00 | 407.00 | 2025-06-17 12:15:00 | 389.90 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2025-06-25 10:45:00 | 406.50 | 2025-07-08 10:15:00 | 399.70 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-07 11:30:00 | 404.50 | 2025-07-08 10:15:00 | 399.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-08 09:15:00 | 405.30 | 2025-07-08 10:15:00 | 399.70 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-08 09:45:00 | 404.00 | 2025-07-24 09:15:00 | 399.35 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-08 10:15:00 | 404.10 | 2025-07-25 11:15:00 | 385.80 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2025-07-08 14:30:00 | 404.35 | 2025-07-25 11:15:00 | 385.80 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-09-10 13:15:00 | 389.75 | 2025-09-17 11:15:00 | 394.25 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-10 14:15:00 | 390.20 | 2025-09-17 11:15:00 | 394.25 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-12 12:00:00 | 389.50 | 2025-09-19 13:15:00 | 394.55 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-15 11:30:00 | 390.15 | 2025-09-19 13:15:00 | 394.55 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-16 11:00:00 | 389.25 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -6.13% |
| SELL | retest2 | 2025-09-16 15:15:00 | 389.05 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2025-09-18 09:15:00 | 390.15 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2025-09-18 09:45:00 | 390.10 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-10-09 15:00:00 | 386.00 | 2025-10-17 13:15:00 | 366.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:00:00 | 386.75 | 2025-10-17 13:15:00 | 367.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:30:00 | 386.50 | 2025-10-17 13:15:00 | 367.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 12:00:00 | 386.80 | 2025-10-17 13:15:00 | 367.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 15:15:00 | 386.00 | 2025-10-17 13:15:00 | 366.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 15:00:00 | 386.00 | 2025-11-06 09:15:00 | 348.07 | TARGET_HIT | 0.50 | 9.83% |
| SELL | retest2 | 2025-10-10 11:00:00 | 386.75 | 2025-11-06 09:15:00 | 348.12 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2025-10-10 11:30:00 | 386.50 | 2025-11-06 10:15:00 | 347.40 | TARGET_HIT | 0.50 | 10.12% |
| SELL | retest2 | 2025-10-10 12:00:00 | 386.80 | 2025-11-06 10:15:00 | 347.85 | TARGET_HIT | 0.50 | 10.07% |
| SELL | retest2 | 2025-10-10 15:15:00 | 386.00 | 2025-11-06 10:15:00 | 347.40 | TARGET_HIT | 0.50 | 10.00% |
