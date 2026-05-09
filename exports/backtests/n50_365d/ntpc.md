# NTPC (NTPC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 402.10
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
| ALERT2_SKIP | 1 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 32
- **Target hits / Stop hits / Partials:** 2 / 32 / 0
- **Avg / median % per leg:** -0.33% / -0.97%
- **Sum % (uncompounded):** -11.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.94% | 9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 2 | 20.0% | 2 | 8 | 0 | 0.94% | 9.4% |
| SELL (all) | 24 | 0 | 0.0% | 0 | 24 | 0 | -0.86% | -20.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -0.86% | -20.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 2 | 5.9% | 2 | 32 | 0 | -0.33% | -11.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 331.15 | 343.92 | 343.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 328.85 | 343.21 | 343.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 12:15:00 | 339.90 | 339.69 | 341.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:00:00 | 339.90 | 339.69 | 341.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 341.25 | 339.69 | 341.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 341.25 | 339.69 | 341.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 341.95 | 339.71 | 341.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 341.75 | 339.71 | 341.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 342.20 | 339.74 | 341.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 342.20 | 339.74 | 341.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 341.90 | 339.76 | 341.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 342.00 | 339.76 | 341.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 340.30 | 339.73 | 341.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 340.15 | 339.73 | 341.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 339.45 | 336.02 | 338.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 337.95 | 336.16 | 338.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 337.95 | 335.56 | 337.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:00:00 | 337.90 | 335.59 | 337.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 338.10 | 335.61 | 337.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 338.15 | 335.66 | 337.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 340.00 | 335.66 | 337.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 341.55 | 335.72 | 337.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.55 | 335.72 | 337.96 | SL hit (close>static) qty=1.00 sl=341.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.55 | 335.72 | 337.96 | SL hit (close>static) qty=1.00 sl=341.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.55 | 335.72 | 337.96 | SL hit (close>static) qty=1.00 sl=341.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 341.55 | 335.72 | 337.96 | SL hit (close>static) qty=1.00 sl=341.30 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 341.55 | 335.72 | 337.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 340.85 | 335.77 | 337.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:15:00 | 340.00 | 335.77 | 337.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 13:00:00 | 340.50 | 335.86 | 338.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.20 | 335.94 | 338.03 | SL hit (close>static) qty=1.00 sl=342.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 13:15:00 | 343.20 | 335.94 | 338.03 | SL hit (close>static) qty=1.00 sl=342.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 340.50 | 337.62 | 338.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 343.35 | 337.80 | 338.73 | SL hit (close>static) qty=1.00 sl=342.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:30:00 | 340.45 | 338.89 | 339.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 342.75 | 339.23 | 339.35 | SL hit (close>static) qty=1.00 sl=342.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 340.95 | 339.47 | 339.47 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 337.15 | 339.46 | 339.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 335.65 | 339.40 | 339.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 339.00 | 338.40 | 338.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 339.00 | 338.40 | 338.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 339.55 | 338.41 | 338.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 339.50 | 338.41 | 338.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 340.55 | 338.43 | 338.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 341.40 | 338.43 | 338.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 338.30 | 338.47 | 338.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 337.95 | 338.47 | 338.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 337.80 | 338.47 | 338.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 10:15:00 | 339.40 | 338.48 | 338.92 | SL hit (close>static) qty=1.00 sl=339.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 10:15:00 | 339.40 | 338.48 | 338.92 | SL hit (close>static) qty=1.00 sl=339.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 338.05 | 338.47 | 338.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 339.60 | 336.36 | 337.58 | SL hit (close>static) qty=1.00 sl=339.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 337.90 | 336.81 | 337.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 337.10 | 336.81 | 337.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 339.35 | 336.89 | 337.76 | SL hit (close>static) qty=1.00 sl=339.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 335.80 | 336.99 | 337.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:30:00 | 336.25 | 336.97 | 337.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 339.50 | 336.89 | 337.70 | SL hit (close>static) qty=1.00 sl=337.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 339.50 | 336.89 | 337.70 | SL hit (close>static) qty=1.00 sl=337.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 335.85 | 337.27 | 337.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 336.70 | 337.26 | 337.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 337.90 | 337.26 | 337.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 337.90 | 337.26 | 337.82 | SL hit (close>static) qty=1.00 sl=337.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 337.90 | 337.26 | 337.82 | SL hit (close>static) qty=1.00 sl=337.80 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 337.90 | 337.26 | 337.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 337.45 | 337.26 | 337.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 336.95 | 337.26 | 337.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 338.75 | 337.28 | 337.82 | SL hit (close>static) qty=1.00 sl=337.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 334.65 | 337.34 | 337.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:45:00 | 336.95 | 335.80 | 336.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:45:00 | 335.80 | 335.80 | 336.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 336.65 | 335.81 | 336.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 336.65 | 335.81 | 336.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 336.20 | 335.81 | 336.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:30:00 | 334.40 | 335.81 | 336.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 335.35 | 333.17 | 335.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:30:00 | 335.50 | 333.25 | 335.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:30:00 | 335.15 | 333.42 | 335.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 335.05 | 333.44 | 335.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 335.05 | 333.44 | 335.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 337.00 | 333.47 | 335.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 337.00 | 333.47 | 335.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 336.95 | 333.51 | 335.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 337.85 | 333.51 | 335.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=337.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=337.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=337.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=338.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=338.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=338.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 339.10 | 333.60 | 335.10 | SL hit (close>static) qty=1.00 sl=338.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 340.10 | 336.35 | 336.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 10:15:00 | 341.25 | 336.40 | 336.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 337.25 | 337.46 | 336.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 337.25 | 337.46 | 336.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 337.45 | 337.46 | 336.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 338.90 | 337.49 | 336.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 334.35 | 337.46 | 336.97 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 338.65 | 337.11 | 336.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:00:00 | 338.70 | 337.34 | 336.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 335.60 | 337.52 | 337.05 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 335.60 | 337.52 | 337.05 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 340.90 | 337.50 | 337.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 339.90 | 337.52 | 337.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 10:45:00 | 341.85 | 337.73 | 337.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 341.50 | 337.85 | 337.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:15:00 | 342.05 | 337.85 | 337.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 15:15:00 | 341.50 | 337.89 | 337.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 338.30 | 339.19 | 338.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 338.40 | 339.19 | 338.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 338.50 | 339.18 | 338.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 338.50 | 339.18 | 338.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 339.15 | 339.18 | 338.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 338.20 | 339.18 | 338.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 338.10 | 340.10 | 338.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 336.45 | 339.96 | 338.65 | SL hit (close<static) qty=1.00 sl=336.75 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 326.00 | 337.52 | 337.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 325.10 | 336.73 | 337.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 325.45 | 324.54 | 328.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:30:00 | 325.80 | 324.54 | 328.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 328.90 | 324.57 | 327.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 327.95 | 324.57 | 327.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 329.50 | 324.62 | 327.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:15:00 | 330.05 | 324.62 | 327.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 348.15 | 330.27 | 330.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 351.50 | 334.39 | 332.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 336.30 | 336.95 | 334.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 336.30 | 336.95 | 334.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 367.15 | 374.18 | 365.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:45:00 | 367.00 | 374.18 | 365.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 365.40 | 374.03 | 365.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:45:00 | 362.65 | 374.03 | 365.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 364.35 | 373.93 | 365.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 363.70 | 373.93 | 365.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 365.95 | 373.85 | 365.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 353.25 | 373.85 | 365.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 355.60 | 373.67 | 365.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 356.95 | 373.67 | 365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 369.40 | 371.93 | 365.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 373.80 | 371.69 | 365.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:45:00 | 370.00 | 371.67 | 365.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 14:15:00 | 407.00 | 381.27 | 372.51 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-27 10:15:00 | 411.18 | 384.68 | 375.00 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-27 15:00:00 | 337.95 | 2025-07-08 09:15:00 | 341.55 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-07 09:45:00 | 337.95 | 2025-07-08 09:15:00 | 341.55 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-07 11:00:00 | 337.90 | 2025-07-08 09:15:00 | 341.55 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-07 11:45:00 | 338.10 | 2025-07-08 09:15:00 | 341.55 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-08 11:15:00 | 340.00 | 2025-07-08 13:15:00 | 343.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-08 13:00:00 | 340.50 | 2025-07-08 13:15:00 | 343.20 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-14 14:15:00 | 340.50 | 2025-07-15 10:15:00 | 343.35 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-07-21 09:30:00 | 340.45 | 2025-07-23 09:15:00 | 342.75 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-31 09:15:00 | 337.95 | 2025-07-31 10:15:00 | 339.40 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-07-31 10:15:00 | 337.80 | 2025-07-31 10:15:00 | 339.40 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-31 13:45:00 | 338.05 | 2025-08-12 11:15:00 | 339.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-08-14 09:15:00 | 337.90 | 2025-08-14 14:15:00 | 339.35 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-18 15:00:00 | 335.80 | 2025-08-20 09:15:00 | 339.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-19 09:30:00 | 336.25 | 2025-08-20 09:15:00 | 339.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-22 09:30:00 | 335.85 | 2025-08-22 13:15:00 | 337.90 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-22 11:15:00 | 336.70 | 2025-08-22 13:15:00 | 337.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-08-22 15:15:00 | 336.95 | 2025-08-25 09:15:00 | 338.75 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-26 09:15:00 | 334.65 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-02 12:45:00 | 336.95 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-02 13:45:00 | 335.80 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-03 12:30:00 | 334.40 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-17 10:30:00 | 335.35 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-09-17 13:30:00 | 335.50 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-18 12:30:00 | 335.15 | 2025-09-19 10:15:00 | 339.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-07 14:00:00 | 338.90 | 2025-10-08 10:15:00 | 334.35 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-10 09:15:00 | 338.65 | 2025-10-14 12:15:00 | 335.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-13 11:00:00 | 338.70 | 2025-10-14 12:15:00 | 335.60 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-15 09:15:00 | 340.90 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-16 10:45:00 | 341.85 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-10-16 13:45:00 | 341.50 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-16 14:15:00 | 342.05 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-10-16 15:15:00 | 341.50 | 2025-10-31 14:15:00 | 336.45 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-04-08 09:15:00 | 373.80 | 2026-04-22 14:15:00 | 407.00 | TARGET_HIT | 1.00 | 8.88% |
| BUY | retest2 | 2026-04-08 10:45:00 | 370.00 | 2026-04-27 10:15:00 | 411.18 | TARGET_HIT | 1.00 | 11.13% |
