# Bharat Heavy Electricals Ltd. (BHEL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 403.20
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
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 9
- **Target hits / Stop hits / Partials:** 1 / 13 / 4
- **Avg / median % per leg:** 0.96% / 1.35%
- **Sum % (uncompounded):** 17.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 3 | 0 | 0.96% | 3.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 3 | 75.0% | 1 | 3 | 0 | 0.96% | 3.8% |
| SELL (all) | 14 | 6 | 42.9% | 0 | 10 | 4 | 0.96% | 13.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 0 | 10 | 4 | 0.96% | 13.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 9 | 50.0% | 1 | 13 | 4 | 0.96% | 17.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 228.18 | 246.38 | 246.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 224.65 | 246.16 | 246.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 225.13 | 222.74 | 230.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:45:00 | 226.21 | 222.74 | 230.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 227.59 | 223.06 | 230.35 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 243.26 | 233.55 | 233.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 246.13 | 235.44 | 234.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 273.50 | 274.40 | 262.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:30:00 | 273.65 | 274.40 | 262.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 272.05 | 284.76 | 275.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 272.05 | 284.76 | 275.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 276.80 | 284.68 | 275.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 283.20 | 284.65 | 275.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 261.00 | 283.88 | 275.12 | SL hit (close<static) qty=1.00 sl=271.85 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 253.65 | 269.47 | 269.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 250.70 | 266.84 | 268.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 267.55 | 266.24 | 267.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 267.55 | 266.24 | 267.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 267.55 | 266.24 | 267.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:30:00 | 268.20 | 266.24 | 267.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 267.10 | 266.24 | 267.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:30:00 | 267.70 | 266.24 | 267.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 267.60 | 266.26 | 267.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:00:00 | 267.60 | 266.26 | 267.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 267.90 | 266.27 | 267.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:30:00 | 267.50 | 266.27 | 267.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 267.40 | 266.29 | 267.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 268.15 | 266.29 | 267.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 271.30 | 266.34 | 267.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 271.30 | 266.34 | 267.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 270.20 | 266.37 | 267.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:15:00 | 269.15 | 266.37 | 267.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 13:15:00 | 273.00 | 266.53 | 267.83 | SL hit (close>static) qty=1.00 sl=271.40 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 294.43 | 262.68 | 262.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 310.49 | 265.81 | 264.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 227.54 | 2025-05-15 15:15:00 | 246.92 | TARGET_HIT | 1.00 | 8.52% |
| BUY | retest2 | 2025-08-07 10:45:00 | 225.15 | 2025-08-07 15:15:00 | 228.18 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-08-07 13:30:00 | 224.10 | 2025-08-07 15:15:00 | 228.18 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-09 09:45:00 | 283.20 | 2026-01-12 09:15:00 | 261.00 | STOP_HIT | 1.00 | -7.84% |
| SELL | retest2 | 2026-02-04 11:15:00 | 269.15 | 2026-02-04 13:15:00 | 273.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-05 09:30:00 | 268.60 | 2026-02-09 09:15:00 | 272.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-05 10:45:00 | 268.90 | 2026-02-09 09:15:00 | 272.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-05 12:15:00 | 269.10 | 2026-02-09 09:15:00 | 272.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-02-06 09:15:00 | 267.00 | 2026-02-09 09:15:00 | 272.30 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-02-11 09:15:00 | 260.10 | 2026-03-12 11:15:00 | 269.55 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2026-03-12 12:30:00 | 267.80 | 2026-03-16 09:15:00 | 254.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 13:00:00 | 268.15 | 2026-03-16 09:15:00 | 254.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 12:30:00 | 267.80 | 2026-03-18 13:15:00 | 260.20 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-03-12 13:00:00 | 268.15 | 2026-03-18 13:15:00 | 260.20 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2026-03-20 14:15:00 | 262.60 | 2026-03-23 12:15:00 | 249.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:15:00 | 262.60 | 2026-03-25 09:15:00 | 266.10 | STOP_HIT | 0.50 | -1.33% |
| SELL | retest2 | 2026-03-25 14:45:00 | 262.40 | 2026-03-30 09:15:00 | 249.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:45:00 | 262.40 | 2026-04-08 09:15:00 | 262.81 | STOP_HIT | 0.50 | -0.16% |
