# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 120.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 76 |
| ALERT1 | 54 |
| ALERT2 | 50 |
| ALERT2_SKIP | 31 |
| ALERT3 | 124 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 61 |
| PARTIAL | 15 |
| TARGET_HIT | 1 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 43
- **Target hits / Stop hits / Partials:** 1 / 61 / 15
- **Avg / median % per leg:** 1.24% / -0.34%
- **Sum % (uncompounded):** 95.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 5 | 14.7% | 1 | 32 | 1 | -0.13% | -4.3% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.55% | 9.1% |
| BUY @ 3rd Alert (retest2) | 32 | 3 | 9.4% | 1 | 31 | 0 | -0.42% | -13.4% |
| SELL (all) | 43 | 29 | 67.4% | 0 | 29 | 14 | 2.32% | 100.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 29 | 67.4% | 0 | 29 | 14 | 2.32% | 100.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.55% | 9.1% |
| retest2 (combined) | 75 | 32 | 42.7% | 1 | 60 | 14 | 1.15% | 86.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 137.06 | 134.02 | 134.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 138.15 | 136.22 | 135.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 139.08 | 139.24 | 138.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 139.08 | 139.24 | 138.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 138.32 | 139.06 | 138.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 138.28 | 139.06 | 138.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 138.87 | 139.02 | 138.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:30:00 | 138.28 | 139.02 | 138.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 138.70 | 138.95 | 138.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:30:00 | 138.60 | 138.95 | 138.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 140.25 | 140.06 | 139.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 139.20 | 140.06 | 139.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 139.26 | 140.04 | 139.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 139.26 | 140.04 | 139.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 139.45 | 139.92 | 139.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:15:00 | 139.75 | 139.92 | 139.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 139.75 | 139.89 | 139.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 138.85 | 139.89 | 139.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 139.50 | 139.81 | 139.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 139.28 | 139.81 | 139.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 139.55 | 139.76 | 139.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:15:00 | 138.82 | 139.76 | 139.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 138.49 | 139.51 | 139.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 138.00 | 139.20 | 139.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 138.44 | 136.61 | 137.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 138.44 | 136.61 | 137.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 138.44 | 136.61 | 137.00 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 138.89 | 137.46 | 137.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 140.17 | 138.55 | 137.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 137.83 | 139.46 | 138.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 137.83 | 139.46 | 138.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 137.83 | 139.46 | 138.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 137.83 | 139.46 | 138.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 138.92 | 139.35 | 138.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 139.30 | 139.37 | 138.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 139.44 | 139.37 | 138.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 13:15:00 | 139.28 | 139.31 | 138.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 139.37 | 139.37 | 139.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 138.76 | 139.35 | 139.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 138.71 | 139.35 | 139.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 139.22 | 139.32 | 139.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 139.90 | 139.35 | 139.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:45:00 | 140.15 | 139.84 | 139.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 139.26 | 140.75 | 140.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 10:15:00 | 138.62 | 139.94 | 140.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 139.32 | 139.25 | 139.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 139.32 | 139.25 | 139.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 139.32 | 139.25 | 139.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:00:00 | 138.76 | 139.29 | 139.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 141.18 | 139.89 | 139.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 141.18 | 139.89 | 139.77 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 139.89 | 141.15 | 141.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 138.18 | 139.83 | 140.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 138.58 | 138.41 | 139.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 138.41 | 138.41 | 139.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 136.96 | 138.18 | 138.94 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 139.60 | 138.78 | 138.67 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 138.00 | 138.58 | 138.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 137.67 | 138.27 | 138.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 12:15:00 | 134.73 | 134.51 | 135.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 13:00:00 | 134.73 | 134.51 | 135.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 133.58 | 134.43 | 135.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 133.58 | 134.43 | 135.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 134.85 | 134.57 | 135.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:30:00 | 134.43 | 134.53 | 135.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 134.44 | 134.53 | 135.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 134.43 | 134.53 | 135.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 136.05 | 134.92 | 135.07 | SL hit (close>static) qty=1.00 sl=135.48 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 136.05 | 134.92 | 135.07 | SL hit (close>static) qty=1.00 sl=135.48 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 136.05 | 134.92 | 135.07 | SL hit (close>static) qty=1.00 sl=135.48 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 136.30 | 135.20 | 135.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 137.01 | 135.78 | 135.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 137.53 | 137.83 | 136.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 137.53 | 137.83 | 136.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 139.81 | 139.75 | 139.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 139.81 | 139.75 | 139.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 139.11 | 139.58 | 139.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 139.11 | 139.58 | 139.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 139.20 | 139.50 | 139.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 139.32 | 139.50 | 139.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 138.85 | 139.37 | 139.19 | SL hit (close<static) qty=1.00 sl=138.98 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 139.38 | 139.27 | 139.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 138.62 | 139.14 | 139.12 | SL hit (close<static) qty=1.00 sl=138.98 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 138.62 | 139.04 | 139.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 137.91 | 138.37 | 138.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 138.81 | 138.40 | 138.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 138.81 | 138.40 | 138.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 138.81 | 138.40 | 138.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 138.68 | 138.40 | 138.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 138.50 | 138.42 | 138.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:30:00 | 137.70 | 138.36 | 138.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 13:15:00 | 138.01 | 138.36 | 138.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:15:00 | 138.04 | 138.34 | 138.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 139.28 | 138.66 | 138.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 139.28 | 138.66 | 138.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 139.28 | 138.66 | 138.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 139.28 | 138.66 | 138.63 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 138.47 | 138.61 | 138.62 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 143.35 | 139.55 | 139.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 10:15:00 | 143.80 | 140.40 | 139.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 145.60 | 145.64 | 143.91 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:15:00 | 147.20 | 145.64 | 143.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 14:15:00 | 154.56 | 150.94 | 147.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 153.24 | 153.52 | 151.12 | SL hit (close<ema200) qty=0.50 sl=153.52 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 152.49 | 153.31 | 151.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 155.42 | 153.37 | 152.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 151.83 | 153.54 | 153.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 151.83 | 153.54 | 153.63 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 154.60 | 153.64 | 153.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 09:15:00 | 154.85 | 153.88 | 153.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 154.98 | 155.61 | 154.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 154.98 | 155.61 | 154.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 154.98 | 155.61 | 154.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 155.20 | 155.61 | 154.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 154.75 | 155.44 | 154.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 154.75 | 155.44 | 154.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 155.85 | 155.52 | 154.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 156.04 | 155.55 | 155.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 11:30:00 | 156.00 | 156.47 | 156.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 156.25 | 156.24 | 156.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:45:00 | 156.39 | 156.24 | 156.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 156.24 | 156.24 | 156.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 156.63 | 156.24 | 156.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 154.01 | 155.80 | 155.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 154.01 | 155.80 | 155.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 154.01 | 155.80 | 155.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 154.01 | 155.80 | 155.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 154.01 | 155.80 | 155.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 154.01 | 155.80 | 155.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 153.30 | 155.30 | 155.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 154.28 | 154.18 | 154.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 154.19 | 154.18 | 154.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 150.34 | 150.49 | 151.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 150.08 | 150.49 | 151.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:45:00 | 150.00 | 150.40 | 151.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 150.00 | 150.09 | 150.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 150.05 | 149.85 | 150.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 150.54 | 149.99 | 150.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 149.58 | 149.90 | 150.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 10:30:00 | 150.27 | 150.03 | 150.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:00:00 | 149.51 | 150.03 | 150.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 142.76 | 143.95 | 144.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 144.55 | 144.03 | 144.59 | SL hit (close>ema200) qty=0.50 sl=144.03 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 142.58 | 143.81 | 144.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 142.50 | 143.81 | 144.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 142.50 | 143.81 | 144.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 142.55 | 143.81 | 144.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:15:00 | 142.10 | 143.29 | 144.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:15:00 | 142.03 | 143.29 | 144.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 143.57 | 143.26 | 143.98 | SL hit (close>ema200) qty=0.50 sl=143.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 143.57 | 143.26 | 143.98 | SL hit (close>ema200) qty=0.50 sl=143.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 143.57 | 143.26 | 143.98 | SL hit (close>ema200) qty=0.50 sl=143.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 143.57 | 143.26 | 143.98 | SL hit (close>ema200) qty=0.50 sl=143.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 143.57 | 143.26 | 143.98 | SL hit (close>ema200) qty=0.50 sl=143.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 143.57 | 143.26 | 143.98 | SL hit (close>ema200) qty=0.50 sl=143.26 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 146.35 | 144.14 | 143.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 146.75 | 144.66 | 144.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 145.68 | 146.12 | 145.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 145.68 | 146.12 | 145.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 145.68 | 146.12 | 145.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 145.68 | 146.12 | 145.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 145.02 | 145.83 | 145.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 145.02 | 145.83 | 145.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 145.50 | 145.76 | 145.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 151.80 | 145.76 | 145.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-21 15:15:00 | 166.98 | 160.87 | 156.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 165.48 | 166.46 | 166.57 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 167.15 | 166.72 | 166.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 170.50 | 167.67 | 167.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 167.53 | 168.41 | 167.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 167.53 | 168.41 | 167.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 167.53 | 168.41 | 167.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 167.73 | 168.41 | 167.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 167.92 | 168.32 | 167.70 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 165.47 | 167.07 | 167.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 165.00 | 166.66 | 167.05 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 170.10 | 167.35 | 167.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 13:15:00 | 171.53 | 169.98 | 169.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 175.00 | 176.76 | 174.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 175.00 | 176.76 | 174.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 175.00 | 176.76 | 174.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 175.00 | 176.76 | 174.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 175.81 | 176.57 | 174.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 175.39 | 176.57 | 174.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 175.82 | 176.48 | 175.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 175.10 | 176.48 | 175.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 174.51 | 176.09 | 175.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 174.02 | 176.09 | 175.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 173.39 | 175.55 | 175.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 172.02 | 175.55 | 175.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 173.21 | 175.08 | 175.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 172.33 | 174.53 | 174.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 178.11 | 174.56 | 174.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 178.11 | 174.56 | 174.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 178.11 | 174.56 | 174.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 178.11 | 174.56 | 174.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 178.55 | 175.36 | 175.02 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 175.95 | 176.14 | 176.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 174.10 | 175.73 | 175.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 10:15:00 | 172.61 | 172.38 | 173.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 10:45:00 | 172.52 | 172.38 | 173.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 176.42 | 173.18 | 173.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 176.42 | 173.18 | 173.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 175.96 | 173.74 | 173.84 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 175.92 | 174.18 | 174.03 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 172.54 | 173.85 | 173.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 171.50 | 173.22 | 173.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 14:15:00 | 171.14 | 170.59 | 171.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 14:15:00 | 171.14 | 170.59 | 171.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 171.14 | 170.59 | 171.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 171.14 | 170.59 | 171.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 171.00 | 170.67 | 171.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 171.84 | 170.67 | 171.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 172.30 | 171.00 | 171.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 172.85 | 171.00 | 171.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 171.21 | 171.04 | 171.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:30:00 | 170.60 | 171.32 | 171.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 172.21 | 171.61 | 171.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 172.21 | 171.61 | 171.59 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 171.33 | 171.55 | 171.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 170.41 | 171.32 | 171.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 165.29 | 165.15 | 167.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 165.29 | 165.15 | 167.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 165.89 | 165.10 | 166.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 165.89 | 165.10 | 166.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 166.02 | 165.28 | 166.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 166.01 | 165.28 | 166.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 166.25 | 165.48 | 166.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 166.25 | 165.48 | 166.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 166.26 | 165.63 | 166.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 166.26 | 165.63 | 166.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 166.48 | 165.80 | 166.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 166.10 | 165.80 | 166.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 166.38 | 165.92 | 166.23 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 171.65 | 167.22 | 166.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 172.73 | 169.57 | 168.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 171.40 | 171.44 | 169.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 171.76 | 171.44 | 169.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 170.60 | 171.07 | 170.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 170.17 | 171.07 | 170.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 168.85 | 170.63 | 170.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 168.85 | 170.63 | 170.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 168.26 | 170.15 | 169.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 168.26 | 170.15 | 169.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 167.96 | 169.49 | 169.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 167.35 | 168.80 | 169.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 165.24 | 164.85 | 166.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 165.24 | 164.85 | 166.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 165.24 | 164.85 | 166.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 166.01 | 164.85 | 166.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 166.10 | 164.34 | 164.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:15:00 | 166.50 | 164.34 | 164.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 165.56 | 164.58 | 164.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 164.07 | 164.85 | 165.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 165.25 | 163.62 | 163.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 165.25 | 163.62 | 163.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 168.30 | 166.18 | 165.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 165.85 | 166.65 | 165.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 14:15:00 | 165.85 | 166.65 | 165.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 165.85 | 166.65 | 165.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 165.85 | 166.65 | 165.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 166.60 | 166.64 | 165.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 168.89 | 167.10 | 166.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 14:15:00 | 167.48 | 166.99 | 166.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 167.00 | 166.91 | 166.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 165.28 | 166.92 | 166.91 | SL hit (close<static) qty=1.00 sl=165.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 165.28 | 166.92 | 166.91 | SL hit (close<static) qty=1.00 sl=165.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 165.28 | 166.92 | 166.91 | SL hit (close<static) qty=1.00 sl=165.40 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 10:15:00 | 165.42 | 166.62 | 166.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 164.53 | 165.58 | 166.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 165.55 | 164.94 | 165.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 165.55 | 164.94 | 165.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 165.55 | 164.94 | 165.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 165.60 | 164.94 | 165.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 165.41 | 165.04 | 165.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 165.56 | 165.04 | 165.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 165.19 | 165.07 | 165.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 165.53 | 165.07 | 165.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 165.00 | 165.06 | 165.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 166.12 | 165.06 | 165.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 165.15 | 165.08 | 165.32 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 167.99 | 165.66 | 165.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 168.36 | 166.98 | 166.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 166.67 | 168.09 | 167.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 166.67 | 168.09 | 167.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 166.67 | 168.09 | 167.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 166.67 | 168.09 | 167.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 165.01 | 167.47 | 167.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 166.88 | 167.47 | 167.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 165.90 | 166.92 | 166.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 165.90 | 166.92 | 166.94 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 167.33 | 166.98 | 166.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 168.18 | 167.22 | 167.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 167.45 | 167.47 | 167.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 167.45 | 167.47 | 167.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 167.45 | 167.47 | 167.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 167.39 | 167.47 | 167.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 167.85 | 167.55 | 167.28 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 165.72 | 167.06 | 167.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 163.29 | 165.67 | 166.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 159.82 | 159.35 | 161.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 15:00:00 | 159.82 | 159.35 | 161.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 160.91 | 159.68 | 160.90 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 163.40 | 161.69 | 161.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 163.60 | 162.07 | 161.72 | Break + close above crossover candle high |

### Cycle 38 — SELL (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 09:15:00 | 156.11 | 160.88 | 161.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 152.61 | 155.36 | 156.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 152.80 | 152.74 | 154.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 153.60 | 152.91 | 154.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 153.60 | 152.91 | 154.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 153.60 | 152.91 | 154.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 153.70 | 153.07 | 154.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:15:00 | 154.06 | 153.07 | 154.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 154.77 | 153.41 | 154.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 154.77 | 153.41 | 154.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 153.51 | 153.43 | 154.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 153.23 | 153.43 | 154.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 150.30 | 153.60 | 153.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 151.85 | 151.10 | 151.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 151.85 | 151.10 | 151.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 151.85 | 151.10 | 151.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 152.77 | 151.43 | 151.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 12:15:00 | 162.22 | 162.23 | 160.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 162.22 | 162.23 | 160.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 162.86 | 162.35 | 161.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:30:00 | 162.35 | 162.35 | 161.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 160.87 | 161.96 | 161.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 160.87 | 161.96 | 161.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 162.01 | 161.97 | 161.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 163.71 | 162.02 | 161.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 160.20 | 163.30 | 163.11 | SL hit (close<static) qty=1.00 sl=160.88 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 160.36 | 162.71 | 162.86 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 164.37 | 162.94 | 162.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 165.15 | 163.38 | 163.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 12:15:00 | 162.44 | 163.67 | 163.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 12:15:00 | 162.44 | 163.67 | 163.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 162.44 | 163.67 | 163.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 162.44 | 163.67 | 163.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 161.50 | 163.24 | 163.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:00:00 | 161.50 | 163.24 | 163.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 161.41 | 162.87 | 163.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 160.29 | 162.13 | 162.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 12:15:00 | 162.50 | 161.70 | 162.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 162.50 | 161.70 | 162.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 162.50 | 161.70 | 162.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 162.50 | 161.70 | 162.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 161.81 | 161.73 | 162.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 161.15 | 161.61 | 162.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 164.17 | 162.29 | 162.32 | SL hit (close>static) qty=1.00 sl=163.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 163.76 | 162.58 | 162.45 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 161.40 | 162.29 | 162.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 160.39 | 161.33 | 161.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 159.40 | 159.35 | 160.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 159.45 | 159.35 | 160.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 159.66 | 159.26 | 159.81 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 162.00 | 160.28 | 160.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 163.68 | 162.18 | 161.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 162.70 | 163.12 | 162.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 162.70 | 163.12 | 162.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 162.40 | 162.94 | 162.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 164.13 | 162.94 | 162.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 163.12 | 163.06 | 162.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 162.00 | 162.73 | 162.54 | SL hit (close<static) qty=1.00 sl=162.09 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 162.00 | 162.73 | 162.54 | SL hit (close<static) qty=1.00 sl=162.09 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 161.69 | 162.38 | 162.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 161.05 | 161.76 | 162.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 161.92 | 160.76 | 161.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 161.92 | 160.76 | 161.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 161.92 | 160.76 | 161.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 162.51 | 160.76 | 161.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 161.50 | 160.91 | 161.26 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 161.76 | 161.49 | 161.46 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 10:15:00 | 160.60 | 161.35 | 161.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 11:15:00 | 160.03 | 161.09 | 161.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 160.32 | 160.19 | 160.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 160.32 | 160.19 | 160.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 160.32 | 160.19 | 160.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 161.73 | 160.19 | 160.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 149.41 | 148.79 | 150.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 149.01 | 148.79 | 150.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 149.05 | 148.71 | 150.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 149.05 | 148.71 | 150.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 154.70 | 149.91 | 150.71 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 151.85 | 151.19 | 151.14 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 149.68 | 151.10 | 151.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 149.58 | 150.79 | 150.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 151.86 | 150.77 | 150.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 151.86 | 150.77 | 150.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 151.86 | 150.77 | 150.91 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 153.98 | 151.41 | 151.19 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 149.90 | 151.69 | 151.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 12:15:00 | 149.48 | 150.96 | 151.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 132.50 | 132.17 | 135.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 133.61 | 132.17 | 135.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 126.72 | 125.94 | 127.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 126.90 | 125.94 | 127.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 126.91 | 126.59 | 127.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 126.91 | 126.59 | 127.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 125.36 | 126.41 | 127.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:45:00 | 124.30 | 125.82 | 126.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 130.30 | 125.96 | 126.26 | SL hit (close>static) qty=1.00 sl=127.36 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 129.08 | 126.58 | 126.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 132.76 | 129.20 | 127.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 129.85 | 130.12 | 128.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 129.85 | 130.12 | 128.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 129.85 | 130.12 | 128.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 129.23 | 130.12 | 128.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 128.66 | 129.83 | 128.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 128.66 | 129.83 | 128.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 127.74 | 129.41 | 128.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 127.74 | 129.41 | 128.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 128.00 | 129.13 | 128.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 125.98 | 129.13 | 128.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 125.06 | 127.85 | 128.08 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 130.50 | 128.38 | 128.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 131.34 | 128.97 | 128.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 13:15:00 | 129.12 | 130.74 | 130.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 13:15:00 | 129.12 | 130.74 | 130.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 129.12 | 130.74 | 130.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 128.87 | 130.74 | 130.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 129.12 | 130.42 | 129.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 129.12 | 130.42 | 129.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 129.20 | 130.17 | 129.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 128.81 | 130.17 | 129.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 128.50 | 129.64 | 129.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 126.83 | 128.77 | 129.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 128.39 | 127.52 | 128.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 128.39 | 127.52 | 128.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 128.39 | 127.52 | 128.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 128.18 | 127.52 | 128.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 129.76 | 127.97 | 128.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 129.12 | 127.97 | 128.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 129.10 | 128.20 | 128.41 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 129.32 | 128.55 | 128.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 132.55 | 129.64 | 129.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 133.00 | 133.23 | 131.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:45:00 | 133.00 | 133.23 | 131.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 133.64 | 133.61 | 132.48 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 131.19 | 132.09 | 132.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 127.32 | 131.14 | 131.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 117.72 | 117.64 | 119.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:30:00 | 118.63 | 117.64 | 119.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 114.87 | 114.44 | 115.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:15:00 | 114.22 | 114.59 | 115.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:00:00 | 114.19 | 114.55 | 115.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 116.25 | 114.86 | 115.43 | SL hit (close>static) qty=1.00 sl=115.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 116.25 | 114.86 | 115.43 | SL hit (close>static) qty=1.00 sl=115.99 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 117.60 | 115.93 | 115.85 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 114.25 | 115.63 | 115.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 113.50 | 115.20 | 115.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 113.94 | 113.53 | 114.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 113.94 | 113.53 | 114.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 111.12 | 111.06 | 112.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 111.00 | 111.06 | 112.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:30:00 | 111.06 | 111.27 | 112.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 111.06 | 111.63 | 111.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:15:00 | 111.06 | 111.38 | 111.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 107.70 | 108.14 | 109.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:00:00 | 107.05 | 107.85 | 109.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:15:00 | 107.50 | 107.73 | 108.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:30:00 | 107.40 | 107.70 | 108.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 105.45 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 105.51 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 105.51 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 105.51 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 101.70 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 102.12 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 102.03 | 106.29 | 107.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 106.08 | 105.96 | 107.04 | SL hit (close>ema200) qty=0.50 sl=105.96 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 106.53 | 104.02 | 103.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 107.46 | 105.14 | 104.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 107.10 | 107.11 | 106.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 106.24 | 107.11 | 106.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 105.85 | 106.86 | 106.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 106.90 | 106.87 | 106.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:30:00 | 107.08 | 106.62 | 106.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 106.80 | 106.62 | 106.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 105.80 | 106.29 | 106.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 105.80 | 106.29 | 106.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 105.80 | 106.29 | 106.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 105.80 | 106.29 | 106.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 101.70 | 105.37 | 105.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 105.95 | 103.45 | 104.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 105.95 | 103.45 | 104.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 105.95 | 103.45 | 104.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 105.98 | 103.45 | 104.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 104.95 | 103.75 | 104.36 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 106.57 | 104.93 | 104.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 108.00 | 105.88 | 105.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 105.26 | 106.72 | 106.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 105.26 | 106.72 | 106.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 105.26 | 106.72 | 106.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 105.26 | 106.72 | 106.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 105.18 | 106.42 | 106.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 105.00 | 106.42 | 106.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 104.71 | 105.79 | 105.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 102.09 | 104.79 | 105.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 105.20 | 102.65 | 103.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 105.20 | 102.65 | 103.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 105.20 | 102.65 | 103.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 105.20 | 102.65 | 103.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 106.00 | 103.32 | 103.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 106.00 | 103.32 | 103.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 107.80 | 104.71 | 104.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 109.22 | 107.77 | 106.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 108.57 | 109.05 | 107.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 108.57 | 109.05 | 107.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 108.57 | 109.05 | 107.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 109.79 | 109.05 | 107.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 109.76 | 109.28 | 108.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 112.85 | 109.33 | 108.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 112.03 | 112.79 | 112.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 112.03 | 112.79 | 112.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 112.03 | 112.79 | 112.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 112.03 | 112.79 | 112.89 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 114.35 | 112.94 | 112.92 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 113.41 | 114.00 | 114.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 113.10 | 113.82 | 113.95 | Break + close below crossover candle low |

### Cycle 69 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 124.36 | 115.83 | 114.84 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 117.45 | 120.39 | 120.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 116.50 | 119.61 | 120.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 120.02 | 118.51 | 119.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 120.02 | 118.51 | 119.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 120.02 | 118.51 | 119.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 120.47 | 118.51 | 119.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 120.30 | 118.87 | 119.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 120.52 | 118.87 | 119.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 119.70 | 119.60 | 119.60 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 119.13 | 119.53 | 119.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 118.42 | 119.19 | 119.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 119.20 | 118.81 | 119.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 15:15:00 | 119.20 | 118.81 | 119.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 119.20 | 118.81 | 119.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 120.44 | 118.81 | 119.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 119.80 | 119.01 | 119.17 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 119.80 | 119.28 | 119.28 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 118.34 | 119.10 | 119.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 116.71 | 118.52 | 118.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 117.95 | 117.85 | 118.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 118.03 | 117.85 | 118.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 120.01 | 118.28 | 118.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 120.20 | 118.28 | 118.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 120.02 | 118.63 | 118.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 121.00 | 119.10 | 118.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 119.52 | 120.29 | 119.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 119.52 | 120.29 | 119.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 119.52 | 120.29 | 119.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 119.41 | 120.29 | 119.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 119.55 | 120.14 | 119.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 119.55 | 120.14 | 119.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 120.54 | 120.22 | 119.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 121.89 | 120.42 | 119.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:15:00 | 121.11 | 120.88 | 120.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 120.90 | 121.60 | 121.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:30:00 | 120.94 | 121.42 | 121.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 120.39 | 121.06 | 121.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 120.39 | 121.06 | 121.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 120.39 | 121.06 | 121.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 120.39 | 121.06 | 121.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 120.39 | 121.06 | 121.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 120.30 | 120.90 | 120.99 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 11:30:00 | 139.30 | 2025-05-30 14:15:00 | 139.26 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-05-27 12:00:00 | 139.44 | 2025-05-30 14:15:00 | 139.26 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-05-27 13:15:00 | 139.28 | 2025-05-30 14:15:00 | 139.26 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-05-27 14:30:00 | 139.37 | 2025-05-30 14:15:00 | 139.26 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-05-28 11:30:00 | 139.90 | 2025-05-30 14:15:00 | 139.26 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-05-28 12:45:00 | 140.15 | 2025-05-30 14:15:00 | 139.26 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-06-03 15:00:00 | 138.76 | 2025-06-05 09:15:00 | 141.18 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-23 12:30:00 | 134.43 | 2025-06-24 09:15:00 | 136.05 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-06-23 13:30:00 | 134.44 | 2025-06-24 09:15:00 | 136.05 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-06-23 14:15:00 | 134.43 | 2025-06-24 09:15:00 | 136.05 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-01 14:15:00 | 139.32 | 2025-07-02 09:15:00 | 138.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-02 12:00:00 | 139.38 | 2025-07-02 12:15:00 | 138.62 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-04 12:30:00 | 137.70 | 2025-07-07 09:15:00 | 139.28 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-04 13:15:00 | 138.01 | 2025-07-07 09:15:00 | 139.28 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-07-04 14:15:00 | 138.04 | 2025-07-07 09:15:00 | 139.28 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2025-07-10 09:15:00 | 147.20 | 2025-07-10 14:15:00 | 154.56 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-10 09:15:00 | 147.20 | 2025-07-11 15:15:00 | 153.24 | STOP_HIT | 0.50 | 4.10% |
| BUY | retest2 | 2025-07-17 09:15:00 | 155.42 | 2025-07-18 11:15:00 | 151.83 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-07-22 13:15:00 | 156.04 | 2025-07-25 11:15:00 | 154.01 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-24 11:30:00 | 156.00 | 2025-07-25 11:15:00 | 154.01 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-24 13:45:00 | 156.25 | 2025-07-25 11:15:00 | 154.01 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-25 09:45:00 | 156.39 | 2025-07-25 11:15:00 | 154.01 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-25 11:15:00 | 156.63 | 2025-07-25 11:15:00 | 154.01 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-30 10:15:00 | 150.08 | 2025-08-07 13:15:00 | 142.76 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2025-07-30 10:15:00 | 150.08 | 2025-08-07 15:15:00 | 144.55 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2025-07-30 12:45:00 | 150.00 | 2025-08-08 09:15:00 | 142.58 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-07-31 09:45:00 | 150.00 | 2025-08-08 09:15:00 | 142.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 12:30:00 | 150.05 | 2025-08-08 09:15:00 | 142.50 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-07-31 15:00:00 | 149.58 | 2025-08-08 09:15:00 | 142.55 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-08-01 10:30:00 | 150.27 | 2025-08-08 10:15:00 | 142.10 | PARTIAL | 0.50 | 5.44% |
| SELL | retest2 | 2025-08-01 11:00:00 | 149.51 | 2025-08-08 10:15:00 | 142.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 12:45:00 | 150.00 | 2025-08-08 12:15:00 | 143.57 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-07-31 09:45:00 | 150.00 | 2025-08-08 12:15:00 | 143.57 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-07-31 12:30:00 | 150.05 | 2025-08-08 12:15:00 | 143.57 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2025-07-31 15:00:00 | 149.58 | 2025-08-08 12:15:00 | 143.57 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-08-01 10:30:00 | 150.27 | 2025-08-08 12:15:00 | 143.57 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-08-01 11:00:00 | 149.51 | 2025-08-08 12:15:00 | 143.57 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2025-08-18 09:15:00 | 151.80 | 2025-08-21 15:15:00 | 166.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-24 12:30:00 | 170.60 | 2025-09-25 09:15:00 | 172.21 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-14 09:15:00 | 164.07 | 2025-10-16 10:15:00 | 165.25 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-21 13:45:00 | 168.89 | 2025-10-27 09:15:00 | 165.28 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-23 14:15:00 | 167.48 | 2025-10-27 09:15:00 | 165.28 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-23 15:15:00 | 167.00 | 2025-10-27 09:15:00 | 165.28 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-11-03 09:15:00 | 166.88 | 2025-11-03 10:15:00 | 165.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-20 14:15:00 | 153.23 | 2025-11-26 15:15:00 | 151.85 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-11-24 09:15:00 | 150.30 | 2025-11-26 15:15:00 | 151.85 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-05 09:15:00 | 163.71 | 2025-12-08 13:15:00 | 160.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-12-11 15:00:00 | 161.15 | 2025-12-12 10:15:00 | 164.17 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-24 09:15:00 | 164.13 | 2025-12-26 09:15:00 | 162.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-24 11:45:00 | 163.12 | 2025-12-26 09:15:00 | 162.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-29 11:45:00 | 124.30 | 2026-01-30 09:15:00 | 130.30 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2026-02-25 13:15:00 | 114.22 | 2026-02-26 09:15:00 | 116.25 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-02-25 15:00:00 | 114.19 | 2026-02-26 09:15:00 | 116.25 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-03-05 10:15:00 | 111.00 | 2026-03-12 09:15:00 | 105.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:30:00 | 111.06 | 2026-03-12 09:15:00 | 105.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 111.06 | 2026-03-12 09:15:00 | 105.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:15:00 | 111.06 | 2026-03-12 09:15:00 | 105.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:00:00 | 107.05 | 2026-03-12 09:15:00 | 101.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 14:15:00 | 107.50 | 2026-03-12 09:15:00 | 102.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:30:00 | 107.40 | 2026-03-12 09:15:00 | 102.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 111.00 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2026-03-05 11:30:00 | 111.06 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2026-03-06 10:45:00 | 111.06 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2026-03-06 14:15:00 | 111.06 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2026-03-10 12:00:00 | 107.05 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 0.91% |
| SELL | retest2 | 2026-03-10 14:15:00 | 107.50 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2026-03-11 11:30:00 | 107.40 | 2026-03-12 11:15:00 | 106.08 | STOP_HIT | 0.50 | 1.23% |
| BUY | retest2 | 2026-03-19 11:00:00 | 106.90 | 2026-03-20 15:15:00 | 105.80 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-03-19 14:30:00 | 107.08 | 2026-03-20 15:15:00 | 105.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-03-19 15:15:00 | 106.80 | 2026-03-20 15:15:00 | 105.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-07 10:15:00 | 109.79 | 2026-04-13 13:15:00 | 112.03 | STOP_HIT | 1.00 | 2.04% |
| BUY | retest2 | 2026-04-07 13:30:00 | 109.76 | 2026-04-13 13:15:00 | 112.03 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2026-04-08 09:15:00 | 112.85 | 2026-04-13 13:15:00 | 112.03 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-05-06 09:15:00 | 121.89 | 2026-05-08 12:15:00 | 120.39 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-05-06 13:15:00 | 121.11 | 2026-05-08 12:15:00 | 120.39 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-05-08 09:45:00 | 120.90 | 2026-05-08 12:15:00 | 120.39 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-05-08 10:30:00 | 120.94 | 2026-05-08 12:15:00 | 120.39 | STOP_HIT | 1.00 | -0.45% |
