# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 263.50
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
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 2 / 25
- **Target hits / Stop hits / Partials:** 0 / 25 / 2
- **Avg / median % per leg:** -1.46% / -1.67%
- **Sum % (uncompounded):** -39.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.26% | -38.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.26% | -38.4% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.10% | -1.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.10% | -1.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 2 | 7.4% | 0 | 25 | 2 | -1.46% | -39.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 234.50 | 240.31 | 240.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 233.89 | 239.71 | 240.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 239.75 | 238.66 | 239.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 239.50 | 238.66 | 239.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 238.35 | 238.65 | 239.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:15:00 | 238.34 | 238.65 | 239.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:45:00 | 238.17 | 238.65 | 239.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 240.09 | 238.65 | 239.39 | SL hit (close>static) qty=1.00 sl=240.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 240.09 | 238.65 | 239.39 | SL hit (close>static) qty=1.00 sl=240.05 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:45:00 | 238.32 | 238.66 | 239.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:45:00 | 238.33 | 238.65 | 239.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 238.10 | 238.55 | 239.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 239.09 | 238.55 | 239.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 239.15 | 238.53 | 239.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 239.03 | 238.53 | 239.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 239.18 | 238.54 | 239.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 240.34 | 238.58 | 239.27 | SL hit (close>static) qty=1.00 sl=240.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 240.34 | 238.58 | 239.27 | SL hit (close>static) qty=1.00 sl=240.05 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 252.50 | 239.90 | 239.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 257.25 | 242.24 | 241.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 284.40 | 284.57 | 274.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:45:00 | 284.00 | 284.57 | 274.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 293.25 | 299.16 | 292.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 297.05 | 293.05 | 290.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 282.05 | 302.05 | 297.22 | SL hit (close<static) qty=1.00 sl=288.25 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 273.60 | 293.85 | 293.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 272.15 | 293.64 | 293.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 09:15:00 | 277.86 | 276.51 | 283.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 277.86 | 276.51 | 283.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 279.80 | 276.26 | 282.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 278.06 | 276.27 | 282.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 278.92 | 276.30 | 282.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 278.60 | 276.40 | 282.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 278.95 | 276.52 | 282.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 280.20 | 276.88 | 282.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 284.05 | 276.95 | 282.45 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 284.05 | 276.95 | 282.45 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 284.05 | 276.95 | 282.45 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 284.05 | 276.95 | 282.45 | SL hit (close>static) qty=1.00 sl=283.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:45:00 | 274.65 | 278.06 | 282.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 277.50 | 278.04 | 282.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 263.62 | 276.21 | 280.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:15:00 | 260.92 | 275.92 | 280.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 13:30:00 | 240.25 | 2025-06-19 09:15:00 | 233.26 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-06-25 12:30:00 | 240.08 | 2025-07-11 09:15:00 | 237.71 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-27 09:15:00 | 243.00 | 2025-07-30 15:15:00 | 239.53 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-04 12:30:00 | 240.32 | 2025-07-30 15:15:00 | 239.53 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-07-09 09:15:00 | 240.82 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-14 09:30:00 | 240.95 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-14 12:00:00 | 240.36 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-23 09:15:00 | 240.85 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-23 13:00:00 | 241.47 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-07-23 13:45:00 | 241.57 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-24 13:15:00 | 242.50 | 2025-07-31 09:15:00 | 236.93 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-07-28 14:30:00 | 241.62 | 2025-08-26 09:15:00 | 238.71 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-29 12:45:00 | 243.07 | 2025-08-26 09:15:00 | 238.71 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-29 14:15:00 | 242.97 | 2025-08-28 09:15:00 | 233.08 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-08-11 09:45:00 | 243.10 | 2025-08-28 09:15:00 | 233.08 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-08-11 12:00:00 | 243.07 | 2025-08-28 09:15:00 | 233.08 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-09-10 13:15:00 | 238.34 | 2025-09-11 10:15:00 | 240.09 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-10 13:45:00 | 238.17 | 2025-09-11 10:15:00 | 240.09 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-11 14:45:00 | 238.32 | 2025-09-16 13:15:00 | 240.34 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-12 09:45:00 | 238.33 | 2025-09-16 13:15:00 | 240.34 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-02-17 10:30:00 | 297.05 | 2026-03-09 09:15:00 | 282.05 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-04-15 11:00:00 | 278.06 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-15 11:45:00 | 278.92 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-15 15:15:00 | 278.60 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-16 11:30:00 | 278.95 | 2026-04-20 10:15:00 | 284.05 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-04-23 12:45:00 | 274.65 | 2026-04-30 09:15:00 | 263.62 | PARTIAL | 0.50 | 4.01% |
| SELL | retest2 | 2026-04-23 13:30:00 | 277.50 | 2026-04-30 11:15:00 | 260.92 | PARTIAL | 0.50 | 5.98% |
