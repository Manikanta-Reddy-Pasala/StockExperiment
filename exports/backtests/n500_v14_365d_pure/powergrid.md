# Power Grid Corporation of India Ltd. (POWERGRID)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 313.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 29 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 31
- **Target hits / Stop hits / Partials:** 1 / 34 / 6
- **Avg / median % per leg:** 0.24% / -0.89%
- **Sum % (uncompounded):** 9.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 1 | 20 | 3 | 0.14% | 3.4% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 0 | 4 | 3 | 1.55% | 10.8% |
| BUY @ 3rd Alert (retest2) | 17 | 1 | 5.9% | 1 | 16 | 0 | -0.44% | -7.5% |
| SELL (all) | 17 | 6 | 35.3% | 0 | 14 | 3 | 0.37% | 6.4% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.96% | 11.8% |
| SELL @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.42% | -5.5% |
| retest1 (combined) | 11 | 7 | 63.6% | 0 | 6 | 5 | 2.06% | 22.7% |
| retest2 (combined) | 30 | 3 | 10.0% | 1 | 28 | 1 | -0.43% | -12.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.10 | 294.84 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 293.95 | 292.68 | 293.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 293.00 | 292.69 | 293.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 294.40 | 292.69 | 293.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 297.85 | 292.74 | 293.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 297.85 | 292.74 | 293.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 294.90 | 293.93 | 294.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 293.90 | 293.93 | 294.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 295.05 | 293.94 | 294.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 295.05 | 293.94 | 294.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 295.05 | 293.95 | 294.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 295.35 | 293.95 | 294.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 294.30 | 293.96 | 294.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 295.15 | 293.96 | 294.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 293.85 | 293.96 | 294.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 292.00 | 293.95 | 294.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 15:15:00 | 294.70 | 293.97 | 294.16 | SL hit (close>static) qty=1.00 sl=294.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 299.15 | 294.37 | 294.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 299.50 | 294.46 | 294.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 294.15 | 295.80 | 295.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 294.25 | 295.79 | 295.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 294.50 | 295.79 | 295.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 296.30 | 295.76 | 295.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:30:00 | 296.95 | 295.77 | 295.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 13:30:00 | 296.45 | 295.77 | 295.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:00:00 | 297.10 | 295.79 | 295.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:00:00 | 296.50 | 295.81 | 295.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 295.40 | 295.80 | 295.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 295.40 | 295.80 | 295.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 297.15 | 295.82 | 295.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 297.85 | 295.82 | 295.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 297.70 | 295.89 | 295.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 11:45:00 | 297.55 | 296.17 | 295.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 297.90 | 296.19 | 295.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=294.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=294.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=294.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=294.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=295.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=295.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=295.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 294.35 | 296.26 | 295.51 | SL hit (close<static) qty=1.00 sl=295.05 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 294.35 | 296.26 | 295.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 294.00 | 296.23 | 295.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 293.00 | 296.23 | 295.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 295.05 | 295.63 | 295.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 295.05 | 295.63 | 295.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 295.15 | 295.63 | 295.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 295.05 | 295.63 | 295.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 289.35 | 295.56 | 295.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 289.35 | 295.56 | 295.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 290.00 | 295.51 | 295.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 291.70 | 295.51 | 295.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 15:00:00 | 290.30 | 290.96 | 292.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 09:15:00 | 289.35 | 290.95 | 292.56 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 275.79 | 288.63 | 291.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 274.88 | 288.03 | 290.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 287.15 | 286.53 | 289.64 | SL hit (close>ema200) qty=0.50 sl=286.53 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 287.15 | 286.53 | 289.64 | SL hit (close>ema200) qty=0.50 sl=286.53 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 288.30 | 285.75 | 288.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 288.60 | 285.86 | 288.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 288.50 | 285.86 | 288.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 287.60 | 285.87 | 288.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 287.25 | 285.89 | 288.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 287.00 | 286.09 | 288.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 287.00 | 286.10 | 288.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 287.10 | 286.11 | 288.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 287.40 | 286.15 | 288.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-18 15:15:00 | 289.60 | 286.25 | 288.30 | SL hit (close>static) qty=1.00 sl=288.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 15:15:00 | 289.60 | 286.25 | 288.30 | SL hit (close>static) qty=1.00 sl=288.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 15:15:00 | 289.60 | 286.25 | 288.30 | SL hit (close>static) qty=1.00 sl=288.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 15:15:00 | 289.60 | 286.25 | 288.30 | SL hit (close>static) qty=1.00 sl=288.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 285.90 | 286.29 | 288.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 285.95 | 286.30 | 288.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 285.95 | 286.32 | 288.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 288.80 | 286.41 | 288.19 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 288.80 | 286.41 | 288.19 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 288.80 | 286.41 | 288.19 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 286.00 | 286.98 | 288.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 286.85 | 285.61 | 287.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 289.45 | 285.66 | 287.41 | SL hit (close>static) qty=1.00 sl=288.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 284.70 | 285.96 | 287.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 283.00 | 285.94 | 287.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 285.96 | 287.33 | SL hit (close>static) qty=1.00 sl=288.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 290.20 | 285.96 | 287.33 | SL hit (close>static) qty=1.00 sl=288.05 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.17 | 288.16 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.82 | 288.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.65 | 268.51 | 274.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 268.65 | 268.51 | 274.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 268.55 | 267.61 | 271.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 255.12 | 266.35 | 270.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 261.50 | 261.21 | 266.18 | SL hit (close>ema200) qty=0.50 sl=261.21 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 11:45:00 | 293.65 | 290.22 | 282.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 12:45:00 | 293.10 | 290.24 | 282.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 13:15:00 | 293.35 | 290.24 | 282.94 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 308.33 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 307.76 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 308.02 | 292.36 | 284.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 291.90 | 292.94 | 285.50 | SL hit (close<ema200) qty=0.50 sl=292.94 alert=retest1 |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 294.50 | 292.94 | 285.50 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | SL hit (close<ema400) qty=1.00 sl=289.21 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 287.25 | 295.14 | 289.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 284.50 | 295.04 | 289.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 285.60 | 295.04 | 289.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 290.95 | 294.71 | 289.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 292.60 | 294.71 | 289.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 287.95 | 294.61 | 289.17 | SL hit (close<static) qty=1.00 sl=289.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 292.10 | 294.50 | 289.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 10:15:00 | 321.31 | 299.38 | 293.20 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 15:00:00 | 296.05 | 2025-05-15 09:15:00 | 291.55 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-05-15 13:00:00 | 295.25 | 2025-05-22 09:15:00 | 289.35 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-05-23 10:00:00 | 295.30 | 2025-05-27 09:15:00 | 293.25 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-05-23 10:30:00 | 296.70 | 2025-05-27 09:15:00 | 293.25 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-09 09:45:00 | 297.75 | 2025-06-11 13:15:00 | 295.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-06-09 11:30:00 | 297.70 | 2025-06-11 13:15:00 | 295.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-09 12:00:00 | 297.85 | 2025-06-11 13:15:00 | 295.10 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-04 09:30:00 | 292.00 | 2025-07-04 15:15:00 | 294.70 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-21 12:30:00 | 296.95 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-21 13:30:00 | 296.45 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-21 15:00:00 | 297.10 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-22 10:00:00 | 296.50 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-22 12:15:00 | 297.85 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-23 09:15:00 | 297.70 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-07-24 11:45:00 | 297.55 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-24 12:45:00 | 297.90 | 2025-07-25 09:15:00 | 294.35 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest1 | 2025-08-18 15:00:00 | 290.30 | 2025-08-28 09:15:00 | 275.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-19 09:15:00 | 289.35 | 2025-08-28 14:15:00 | 274.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-18 15:00:00 | 290.30 | 2025-09-02 10:15:00 | 287.15 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-08-19 09:15:00 | 289.35 | 2025-09-02 10:15:00 | 287.15 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2025-09-15 10:45:00 | 287.25 | 2025-09-18 15:15:00 | 289.60 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-17 09:45:00 | 287.00 | 2025-09-18 15:15:00 | 289.60 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-17 11:00:00 | 287.00 | 2025-09-18 15:15:00 | 289.60 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-17 12:00:00 | 287.10 | 2025-09-18 15:15:00 | 289.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-19 14:30:00 | 285.90 | 2025-09-23 14:15:00 | 288.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-22 12:15:00 | 285.95 | 2025-09-23 14:15:00 | 288.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-23 09:15:00 | 285.95 | 2025-09-23 14:15:00 | 288.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-25 13:45:00 | 286.00 | 2025-10-03 14:15:00 | 289.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-08 14:00:00 | 284.70 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-10-09 09:15:00 | 283.00 | 2025-10-10 09:15:00 | 290.20 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-01-06 11:15:00 | 268.55 | 2026-01-12 09:15:00 | 255.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 268.55 | 2026-01-29 15:15:00 | 261.50 | STOP_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2026-03-09 11:45:00 | 293.65 | 2026-03-13 09:15:00 | 308.33 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 12:45:00 | 293.10 | 2026-03-13 09:15:00 | 307.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 13:15:00 | 293.35 | 2026-03-13 09:15:00 | 308.02 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-03-09 11:45:00 | 293.65 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.60% |
| BUY | retest1 | 2026-03-09 12:45:00 | 293.10 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.41% |
| BUY | retest1 | 2026-03-09 13:15:00 | 293.35 | 2026-03-16 10:15:00 | 291.90 | STOP_HIT | 0.50 | -0.49% |
| BUY | retest1 | 2026-03-16 11:15:00 | 294.50 | 2026-04-02 09:15:00 | 286.60 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-04-06 09:15:00 | 292.60 | 2026-04-06 10:15:00 | 287.95 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-04-06 13:30:00 | 292.10 | 2026-04-20 10:15:00 | 321.31 | TARGET_HIT | 1.00 | 10.00% |
