# JK Tyre & Industries Ltd. (JKTYRE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 406.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 4
- **Avg / median % per leg:** 1.00% / 2.14%
- **Sum % (uncompounded):** 8.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.98% | -11.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.98% | -11.9% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 0 | 4 | 4.98% | 19.9% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 0 | 4 | 4.98% | 19.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 0 | 4 | 4.98% | 19.9% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.98% | -11.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 332.10 | 305.46 | 305.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 335.00 | 306.87 | 306.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 356.75 | 360.47 | 343.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 356.85 | 360.47 | 343.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 348.90 | 360.20 | 345.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 356.85 | 358.43 | 345.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 354.65 | 358.16 | 346.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:45:00 | 354.50 | 363.78 | 355.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 355.30 | 363.68 | 355.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 352.70 | 363.47 | 355.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 351.75 | 363.47 | 355.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 355.50 | 362.82 | 355.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 355.50 | 362.82 | 355.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 353.45 | 362.73 | 355.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 353.55 | 362.73 | 355.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 353.55 | 362.64 | 355.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 353.55 | 362.64 | 355.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 353.85 | 362.39 | 355.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 351.95 | 362.39 | 355.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 344.75 | 361.04 | 355.02 | SL hit (close<static) qty=1.00 sl=345.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 344.75 | 361.04 | 355.02 | SL hit (close<static) qty=1.00 sl=345.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 344.75 | 361.04 | 355.02 | SL hit (close<static) qty=1.00 sl=345.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 344.75 | 361.04 | 355.02 | SL hit (close<static) qty=1.00 sl=345.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 321.10 | 350.35 | 350.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 318.55 | 348.64 | 349.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 347.55 | 332.50 | 338.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:15:00 | 346.60 | 332.50 | 338.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 345.60 | 332.63 | 338.81 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 369.00 | 343.16 | 343.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 370.10 | 343.68 | 343.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 357.40 | 358.68 | 352.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 357.40 | 358.68 | 352.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 514.70 | 539.83 | 516.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 514.70 | 539.83 | 516.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 508.15 | 539.51 | 516.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 508.15 | 539.51 | 516.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 426.15 | 499.17 | 499.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 418.50 | 495.62 | 497.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 13:15:00 | 433.70 | 432.33 | 455.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 418.25 | 432.35 | 455.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 430.85 | 431.62 | 454.11 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:45:00 | 431.55 | 431.62 | 454.00 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 13:15:00 | 431.20 | 430.80 | 451.95 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 409.31 | 428.86 | 447.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 409.97 | 428.86 | 447.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 409.64 | 428.86 | 447.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 13:15:00 | 397.34 | 427.79 | 446.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-25 09:45:00 | 356.85 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-06-26 13:00:00 | 354.65 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-07-22 14:45:00 | 354.50 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-23 09:15:00 | 355.30 | 2025-07-28 10:15:00 | 344.75 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest1 | 2026-04-13 09:15:00 | 418.25 | 2026-04-24 09:15:00 | 409.31 | PARTIAL | 0.50 | 2.14% |
| SELL | retest1 | 2026-04-15 11:00:00 | 430.85 | 2026-04-24 09:15:00 | 409.97 | PARTIAL | 0.50 | 4.85% |
| SELL | retest1 | 2026-04-15 11:45:00 | 431.55 | 2026-04-24 09:15:00 | 409.64 | PARTIAL | 0.50 | 5.08% |
| SELL | retest1 | 2026-04-17 13:15:00 | 431.20 | 2026-04-24 13:15:00 | 397.34 | PARTIAL | 0.50 | 7.85% |
