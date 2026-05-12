# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 339.00
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
| ENTRY2 | 43 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 35
- **Target hits / Stop hits / Partials:** 0 / 43 / 7
- **Avg / median % per leg:** -0.46% / -1.23%
- **Sum % (uncompounded):** -23.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.43% | -29.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.43% | -29.1% |
| SELL (all) | 38 | 15 | 39.5% | 0 | 31 | 7 | 0.16% | 6.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 15 | 39.5% | 0 | 31 | 7 | 0.16% | 6.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 15 | 30.0% | 0 | 43 | 7 | -0.46% | -23.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 272.05 | 262.83 | 262.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 276.50 | 263.98 | 263.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 272.35 | 273.03 | 268.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 272.35 | 273.03 | 268.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 348.00 | 368.03 | 348.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 348.00 | 368.03 | 348.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 348.45 | 367.84 | 348.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:30:00 | 352.75 | 366.79 | 348.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 15:15:00 | 353.80 | 363.99 | 350.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 361.80 | 362.91 | 350.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 353.00 | 364.58 | 353.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 348.50 | 364.42 | 353.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 348.50 | 364.42 | 353.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-02 11:15:00 | 344.60 | 364.22 | 353.71 | SL hit (close<static) qty=1.00 sl=346.80 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 317.80 | 360.01 | 360.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 314.50 | 355.56 | 357.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 316.55 | 309.36 | 324.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 320.80 | 310.73 | 322.35 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-13 09:15:00 | 271.05 | 2025-06-26 14:15:00 | 269.30 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-06-16 14:15:00 | 272.50 | 2025-06-26 14:15:00 | 269.30 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-06-18 11:15:00 | 267.25 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-06-18 12:15:00 | 267.20 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-06-24 13:30:00 | 267.35 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-25 13:15:00 | 267.30 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-06-26 11:00:00 | 266.15 | 2025-07-03 09:15:00 | 268.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-26 12:30:00 | 266.05 | 2025-07-03 09:15:00 | 268.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-01 12:00:00 | 266.20 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-02 09:15:00 | 264.60 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-07-02 10:15:00 | 260.45 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-07-02 13:00:00 | 261.55 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-07-02 14:30:00 | 261.35 | 2025-07-07 09:15:00 | 268.55 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-07-02 15:15:00 | 261.20 | 2025-07-07 09:15:00 | 268.55 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-04 11:30:00 | 266.05 | 2025-07-10 13:15:00 | 268.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-04 15:15:00 | 266.35 | 2025-07-10 13:15:00 | 268.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-10 11:45:00 | 266.40 | 2025-07-18 10:15:00 | 258.88 | PARTIAL | 0.50 | 2.82% |
| SELL | retest2 | 2025-07-10 11:45:00 | 266.40 | 2025-07-22 15:15:00 | 266.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2025-07-10 12:45:00 | 266.45 | 2025-07-23 09:15:00 | 257.50 | PARTIAL | 0.50 | 3.36% |
| SELL | retest2 | 2025-07-11 10:30:00 | 265.00 | 2025-07-28 09:15:00 | 251.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 09:30:00 | 264.95 | 2025-07-28 09:15:00 | 251.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 10:30:00 | 265.15 | 2025-07-28 09:15:00 | 251.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 11:15:00 | 265.05 | 2025-07-28 09:15:00 | 251.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 261.00 | 2025-07-28 11:15:00 | 247.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 12:45:00 | 266.45 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2025-07-11 10:30:00 | 265.00 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2025-07-15 09:30:00 | 264.95 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2025-07-15 10:30:00 | 265.15 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2025-07-15 11:15:00 | 265.05 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-07-23 09:15:00 | 261.00 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2025-08-19 09:15:00 | 262.65 | 2025-08-20 14:15:00 | 268.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-08-20 14:15:00 | 264.55 | 2025-08-20 14:15:00 | 268.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-08-21 13:00:00 | 264.50 | 2025-08-25 13:15:00 | 267.75 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-28 09:15:00 | 258.20 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-08-28 14:45:00 | 259.90 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-09-02 14:15:00 | 260.00 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-03 09:30:00 | 260.00 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-03 14:30:00 | 259.50 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-01-12 13:30:00 | 352.75 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-01-20 15:15:00 | 353.80 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-01-22 09:15:00 | 361.80 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2026-02-02 09:45:00 | 353.00 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-24 13:15:00 | 364.55 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-24 13:45:00 | 365.00 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-02-24 14:30:00 | 365.00 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-03-02 09:30:00 | 367.50 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-03-04 14:30:00 | 363.45 | 2026-03-09 09:15:00 | 348.25 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-04 15:15:00 | 362.85 | 2026-03-09 09:15:00 | 348.25 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2026-03-05 12:15:00 | 363.30 | 2026-03-09 09:15:00 | 348.25 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2026-03-10 11:45:00 | 363.50 | 2026-03-10 12:15:00 | 359.25 | STOP_HIT | 1.00 | -1.17% |
