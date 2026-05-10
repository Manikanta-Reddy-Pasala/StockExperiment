# Motherson Sumi Wiring India Ltd. (MSUMI)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 42.56
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 15 |
| ALERT3 | 115 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 50 |
| PARTIAL | 10 |
| TARGET_HIT | 2 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 35
- **Target hits / Stop hits / Partials:** 2 / 52 / 10
- **Avg / median % per leg:** 0.76% / -0.23%
- **Sum % (uncompounded):** 48.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 10 | 41.7% | 2 | 19 | 3 | 0.72% | 17.4% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 2 | 1 | 3 | 6.52% | 39.1% |
| BUY @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 0 | 18 | 0 | -1.21% | -21.8% |
| SELL (all) | 40 | 19 | 47.5% | 0 | 33 | 7 | 0.78% | 31.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.28% | 10.6% |
| SELL @ 3rd Alert (retest2) | 38 | 17 | 44.7% | 0 | 32 | 6 | 0.54% | 20.6% |
| retest1 (combined) | 8 | 8 | 100.0% | 2 | 2 | 4 | 6.21% | 49.7% |
| retest2 (combined) | 56 | 21 | 37.5% | 0 | 50 | 6 | -0.02% | -1.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 37.84 | 37.89 | 37.89 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 38.27 | 37.96 | 37.93 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 37.91 | 37.92 | 37.92 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 37.96 | 37.93 | 37.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 38.35 | 38.07 | 38.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 38.56 | 38.56 | 38.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 11:00:00 | 38.56 | 38.56 | 38.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 38.17 | 38.47 | 38.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 38.17 | 38.47 | 38.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 38.24 | 38.43 | 38.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:15:00 | 38.20 | 38.43 | 38.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 38.20 | 38.38 | 38.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 38.03 | 38.38 | 38.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 37.97 | 38.30 | 38.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 37.62 | 38.07 | 38.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 37.67 | 37.59 | 37.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 37.67 | 37.59 | 37.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 37.81 | 37.63 | 37.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 37.62 | 37.63 | 37.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 38.00 | 37.71 | 37.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 38.00 | 37.71 | 37.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 38.33 | 37.83 | 37.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 38.33 | 37.83 | 37.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 38.31 | 37.93 | 37.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 38.95 | 38.34 | 38.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 39.10 | 39.18 | 38.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 39.10 | 39.18 | 38.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 39.27 | 39.48 | 39.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:45:00 | 39.29 | 39.48 | 39.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 39.25 | 39.44 | 39.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 39.24 | 39.44 | 39.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 39.19 | 39.39 | 39.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 39.12 | 39.39 | 39.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 39.28 | 39.37 | 39.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 39.11 | 39.37 | 39.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 39.07 | 39.31 | 39.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 39.07 | 39.31 | 39.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 39.09 | 39.26 | 39.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 39.09 | 39.26 | 39.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 39.03 | 39.22 | 39.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 38.90 | 39.14 | 39.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 38.86 | 38.81 | 38.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:00:00 | 38.86 | 38.81 | 38.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 38.87 | 38.83 | 38.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 38.90 | 38.83 | 38.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 38.75 | 38.81 | 38.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 38.90 | 38.81 | 38.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 38.88 | 38.81 | 38.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 38.88 | 38.81 | 38.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 38.67 | 38.79 | 38.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 12:45:00 | 38.57 | 38.71 | 38.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 39.09 | 38.75 | 38.78 | SL hit (close>static) qty=1.00 sl=39.06 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 39.19 | 38.84 | 38.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 39.47 | 38.96 | 38.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 41.35 | 41.42 | 41.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:45:00 | 41.30 | 41.42 | 41.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 41.23 | 41.38 | 41.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 41.23 | 41.38 | 41.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 40.98 | 41.30 | 41.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 40.98 | 41.30 | 41.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 41.05 | 41.25 | 41.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 40.69 | 41.25 | 41.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 40.77 | 41.15 | 41.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 39.60 | 40.12 | 40.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 39.91 | 39.81 | 40.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 39.91 | 39.81 | 40.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 40.03 | 39.92 | 40.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:45:00 | 39.97 | 39.95 | 40.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 39.90 | 39.95 | 40.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 39.83 | 39.68 | 39.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 39.83 | 39.68 | 39.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 39.83 | 39.68 | 39.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 39.93 | 39.78 | 39.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 40.03 | 40.13 | 40.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 40.03 | 40.13 | 40.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 40.03 | 40.13 | 40.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:45:00 | 40.03 | 40.13 | 40.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 40.10 | 40.12 | 40.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 40.08 | 40.12 | 40.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 39.94 | 40.09 | 40.01 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 39.78 | 39.94 | 39.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 13:15:00 | 39.75 | 39.90 | 39.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 39.77 | 39.77 | 39.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 13:00:00 | 39.77 | 39.77 | 39.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 39.67 | 39.75 | 39.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 39.87 | 39.75 | 39.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 39.80 | 39.75 | 39.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 39.66 | 39.68 | 39.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 39.63 | 39.64 | 39.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 40.18 | 39.75 | 39.75 | SL hit (close>static) qty=1.00 sl=39.87 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 40.18 | 39.75 | 39.75 | SL hit (close>static) qty=1.00 sl=39.87 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 40.76 | 39.95 | 39.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 41.00 | 40.16 | 39.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 40.31 | 40.37 | 40.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 15:00:00 | 40.31 | 40.37 | 40.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 40.33 | 40.36 | 40.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 40.34 | 40.36 | 40.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 40.16 | 40.32 | 40.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 40.69 | 40.31 | 40.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 42.27 | 42.72 | 42.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 14:15:00 | 42.27 | 42.72 | 42.77 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 43.45 | 42.76 | 42.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 43.71 | 43.15 | 43.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 12:15:00 | 43.26 | 43.33 | 43.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 12:30:00 | 43.37 | 43.33 | 43.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 43.51 | 43.35 | 43.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 43.71 | 43.35 | 43.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 43.67 | 43.46 | 43.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:00:00 | 43.75 | 43.50 | 43.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 42.90 | 43.41 | 43.37 | SL hit (close<static) qty=1.00 sl=43.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 42.90 | 43.41 | 43.37 | SL hit (close<static) qty=1.00 sl=43.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 42.90 | 43.41 | 43.37 | SL hit (close<static) qty=1.00 sl=43.10 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 42.98 | 43.32 | 43.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 42.63 | 42.96 | 43.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 40.90 | 40.85 | 41.37 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 40.35 | 40.85 | 41.37 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 11:15:00 | 38.33 | 38.91 | 39.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 38.11 | 37.64 | 38.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 38.11 | 37.64 | 38.21 | SL hit (close>ema200) qty=0.50 sl=37.64 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 38.24 | 37.64 | 38.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 38.19 | 37.66 | 37.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 38.08 | 37.66 | 37.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 38.14 | 37.76 | 37.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:45:00 | 37.93 | 37.81 | 37.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 37.96 | 37.81 | 37.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:30:00 | 37.97 | 37.88 | 37.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 38.34 | 38.03 | 38.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 38.34 | 38.03 | 38.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 38.34 | 38.03 | 38.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 38.34 | 38.03 | 38.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 39.05 | 38.23 | 38.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 12:15:00 | 39.13 | 39.30 | 38.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 13:00:00 | 39.13 | 39.30 | 38.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 38.89 | 39.18 | 38.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 38.89 | 39.18 | 38.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 38.52 | 39.05 | 38.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 38.15 | 39.05 | 38.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 38.28 | 38.90 | 38.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 37.86 | 38.90 | 38.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 38.40 | 38.75 | 38.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 38.03 | 38.39 | 38.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 12:15:00 | 38.37 | 38.31 | 38.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 13:00:00 | 38.37 | 38.31 | 38.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 38.45 | 38.34 | 38.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:30:00 | 38.72 | 38.34 | 38.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 39.09 | 38.49 | 38.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 39.09 | 38.49 | 38.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 15:15:00 | 39.08 | 38.61 | 38.58 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 38.49 | 38.56 | 38.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 13:15:00 | 38.33 | 38.50 | 38.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 13:15:00 | 37.60 | 37.59 | 37.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:30:00 | 37.63 | 37.59 | 37.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 37.81 | 37.64 | 37.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 37.81 | 37.64 | 37.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 37.84 | 37.68 | 37.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 38.40 | 37.68 | 37.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 39.08 | 37.96 | 37.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 39.55 | 38.28 | 38.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 42.55 | 42.63 | 42.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 42.01 | 42.63 | 42.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 42.39 | 42.58 | 42.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 41.96 | 42.58 | 42.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 42.00 | 42.31 | 42.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 42.00 | 42.31 | 42.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 41.50 | 42.15 | 42.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 42.04 | 42.10 | 42.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 41.85 | 42.01 | 42.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 41.85 | 42.01 | 42.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 41.46 | 41.90 | 41.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 12:15:00 | 41.88 | 41.83 | 41.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 12:15:00 | 41.88 | 41.83 | 41.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 41.88 | 41.83 | 41.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:45:00 | 41.88 | 41.83 | 41.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 41.73 | 41.81 | 41.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 41.59 | 41.77 | 41.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 41.73 | 41.54 | 41.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 41.73 | 41.54 | 41.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 42.00 | 41.71 | 41.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 42.58 | 42.59 | 42.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:45:00 | 42.52 | 42.59 | 42.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 48.04 | 48.53 | 47.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 48.04 | 48.53 | 47.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 49.00 | 49.18 | 48.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 50.03 | 49.18 | 48.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:00:00 | 50.18 | 49.62 | 48.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:00:00 | 50.09 | 49.79 | 49.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 48.50 | 49.45 | 49.31 | SL hit (close<static) qty=1.00 sl=48.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 48.50 | 49.45 | 49.31 | SL hit (close<static) qty=1.00 sl=48.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 48.50 | 49.45 | 49.31 | SL hit (close<static) qty=1.00 sl=48.52 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 49.00 | 49.23 | 49.25 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 49.48 | 49.30 | 49.28 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 49.06 | 49.25 | 49.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 12:15:00 | 48.69 | 49.11 | 49.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 15:15:00 | 49.15 | 48.17 | 48.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 15:15:00 | 49.15 | 48.17 | 48.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 49.15 | 48.17 | 48.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:45:00 | 47.68 | 48.00 | 48.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:45:00 | 47.68 | 47.82 | 48.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 49.39 | 48.28 | 48.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 49.39 | 48.28 | 48.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 49.39 | 48.28 | 48.18 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 47.49 | 48.12 | 48.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 47.16 | 47.93 | 48.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 10:15:00 | 47.11 | 47.06 | 47.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:45:00 | 47.15 | 47.06 | 47.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 46.60 | 46.77 | 47.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:00:00 | 46.22 | 46.66 | 47.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 46.17 | 46.46 | 46.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 46.20 | 45.75 | 45.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 46.20 | 45.75 | 45.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 11:15:00 | 46.20 | 45.75 | 45.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 46.55 | 46.00 | 45.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 46.03 | 46.14 | 45.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 46.03 | 46.14 | 45.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 45.84 | 46.08 | 45.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 45.84 | 46.08 | 45.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 45.77 | 46.02 | 45.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 45.93 | 46.02 | 45.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 45.63 | 45.88 | 45.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 45.63 | 45.88 | 45.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 45.08 | 45.72 | 45.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 44.77 | 44.72 | 45.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 44.77 | 44.72 | 45.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 44.77 | 44.72 | 45.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 44.55 | 44.84 | 44.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 46.53 | 44.95 | 44.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 46.53 | 44.95 | 44.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 46.77 | 46.40 | 46.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 46.40 | 46.40 | 46.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:30:00 | 46.40 | 46.40 | 46.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 47.13 | 47.35 | 47.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 47.08 | 47.35 | 47.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 47.04 | 47.26 | 47.06 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 46.35 | 46.92 | 46.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 46.28 | 46.63 | 46.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 46.72 | 46.61 | 46.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 46.72 | 46.61 | 46.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 46.72 | 46.61 | 46.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 46.36 | 46.61 | 46.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 47.25 | 46.81 | 46.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 47.25 | 46.81 | 46.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 47.77 | 47.00 | 46.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 47.29 | 47.40 | 47.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 47.58 | 47.40 | 47.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 47.40 | 47.40 | 47.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 47.72 | 47.46 | 47.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 46.79 | 47.30 | 47.28 | SL hit (close<static) qty=1.00 sl=47.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 47.00 | 47.24 | 47.26 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 47.62 | 47.26 | 47.25 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 46.82 | 47.19 | 47.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 46.64 | 47.00 | 47.13 | Break + close below crossover candle low |

### Cycle 36 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 48.40 | 47.23 | 47.20 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 46.12 | 47.18 | 47.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 45.70 | 46.12 | 46.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 11:15:00 | 46.04 | 45.94 | 46.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 11:45:00 | 46.04 | 45.94 | 46.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 46.53 | 46.12 | 46.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 46.53 | 46.12 | 46.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 46.35 | 46.17 | 46.30 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 46.75 | 46.38 | 46.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 46.92 | 46.49 | 46.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 13:15:00 | 48.57 | 48.70 | 48.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 14:00:00 | 48.57 | 48.70 | 48.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 48.47 | 48.56 | 48.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 49.31 | 48.64 | 48.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 49.00 | 48.82 | 48.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 48.08 | 48.62 | 48.61 | SL hit (close<static) qty=1.00 sl=48.17 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 48.08 | 48.62 | 48.61 | SL hit (close<static) qty=1.00 sl=48.17 alert=retest2 |

### Cycle 39 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 48.49 | 48.60 | 48.61 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 48.72 | 48.62 | 48.62 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 47.91 | 48.48 | 48.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 47.72 | 48.24 | 48.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 46.28 | 46.26 | 46.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:45:00 | 46.27 | 46.26 | 46.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 45.99 | 46.22 | 46.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 45.77 | 46.12 | 46.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 45.74 | 45.99 | 46.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 14:45:00 | 45.72 | 45.88 | 46.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 46.70 | 46.23 | 46.24 | SL hit (close>static) qty=1.00 sl=46.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 46.70 | 46.23 | 46.24 | SL hit (close>static) qty=1.00 sl=46.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 46.70 | 46.23 | 46.24 | SL hit (close>static) qty=1.00 sl=46.55 alert=retest2 |

### Cycle 42 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 46.52 | 46.29 | 46.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 09:15:00 | 47.02 | 46.48 | 46.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 10:15:00 | 45.96 | 46.38 | 46.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 45.96 | 46.38 | 46.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 45.96 | 46.38 | 46.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 46.01 | 46.38 | 46.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 46.12 | 46.33 | 46.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 46.29 | 46.33 | 46.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 46.08 | 46.28 | 46.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 46.08 | 46.28 | 46.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 45.81 | 46.12 | 46.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 45.50 | 45.31 | 45.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 14:15:00 | 45.61 | 45.31 | 45.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 45.57 | 45.36 | 45.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 45.57 | 45.36 | 45.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 45.76 | 45.44 | 45.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 45.18 | 45.44 | 45.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 45.25 | 44.87 | 44.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 45.25 | 44.87 | 44.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 45.79 | 45.05 | 44.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 45.67 | 45.74 | 45.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 45.67 | 45.74 | 45.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 45.93 | 46.09 | 45.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 46.03 | 46.09 | 45.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 45.93 | 46.06 | 45.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 46.52 | 46.06 | 45.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 45.44 | 45.95 | 45.84 | SL hit (close<static) qty=1.00 sl=45.81 alert=retest2 |

### Cycle 45 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 45.54 | 45.73 | 45.75 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 46.00 | 45.81 | 45.78 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 45.46 | 45.71 | 45.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 45.21 | 45.61 | 45.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 45.65 | 45.15 | 45.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 45.65 | 45.15 | 45.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 45.65 | 45.15 | 45.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 45.65 | 45.15 | 45.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 45.90 | 45.30 | 45.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 45.90 | 45.30 | 45.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 46.00 | 45.44 | 45.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 46.35 | 45.62 | 45.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 46.80 | 46.83 | 46.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 46.80 | 46.83 | 46.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 46.28 | 46.61 | 46.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 46.28 | 46.61 | 46.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 46.29 | 46.55 | 46.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 46.11 | 46.55 | 46.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 46.15 | 46.47 | 46.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 46.12 | 46.47 | 46.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 46.05 | 46.38 | 46.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 46.05 | 46.38 | 46.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 45.38 | 46.04 | 46.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 45.14 | 45.51 | 45.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 44.96 | 44.89 | 45.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 44.96 | 44.89 | 45.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 46.75 | 45.26 | 45.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 46.75 | 45.26 | 45.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 46.25 | 45.46 | 45.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 47.02 | 45.89 | 45.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 11:15:00 | 47.86 | 47.90 | 47.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 13:30:00 | 48.42 | 48.10 | 47.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:30:00 | 48.11 | 48.15 | 47.59 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 14:15:00 | 48.99 | 48.07 | 47.69 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:15:00 | 50.84 | 49.81 | 49.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:15:00 | 50.52 | 49.81 | 49.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 13:15:00 | 51.44 | 50.34 | 49.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-06 09:15:00 | 53.26 | 51.19 | 50.09 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-06 09:15:00 | 52.92 | 51.19 | 50.09 | Target hit (10%) qty=0.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 51.00 | 51.38 | 50.47 | SL hit (close<ema200) qty=0.50 sl=51.38 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 50.80 | 51.18 | 50.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 50.64 | 51.18 | 50.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 50.35 | 51.01 | 50.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 50.33 | 51.01 | 50.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 50.43 | 50.90 | 50.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 50.38 | 50.90 | 50.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 49.88 | 50.35 | 50.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 48.67 | 49.59 | 49.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 46.30 | 46.03 | 46.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 46.30 | 46.03 | 46.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 46.30 | 46.03 | 46.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 46.41 | 46.03 | 46.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 45.57 | 45.91 | 46.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 45.24 | 45.80 | 46.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 45.31 | 45.71 | 46.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:45:00 | 45.26 | 45.55 | 46.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 45.32 | 45.50 | 45.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 42.98 | 44.04 | 44.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 43.04 | 44.04 | 44.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 43.00 | 44.04 | 44.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 43.05 | 44.04 | 44.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 44.04 | 43.87 | 44.40 | SL hit (close>ema200) qty=0.50 sl=43.87 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 44.04 | 43.87 | 44.40 | SL hit (close>ema200) qty=0.50 sl=43.87 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 44.04 | 43.87 | 44.40 | SL hit (close>ema200) qty=0.50 sl=43.87 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 44.04 | 43.87 | 44.40 | SL hit (close>ema200) qty=0.50 sl=43.87 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 44.56 | 44.08 | 44.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 44.62 | 44.08 | 44.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 44.45 | 44.15 | 44.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:45:00 | 44.10 | 44.15 | 44.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:15:00 | 41.89 | 42.39 | 42.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 12:15:00 | 42.38 | 42.31 | 42.75 | SL hit (close>ema200) qty=0.50 sl=42.31 alert=retest2 |

### Cycle 52 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 42.89 | 42.54 | 42.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 43.29 | 42.83 | 42.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 42.82 | 42.83 | 42.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 42.82 | 42.83 | 42.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 42.65 | 42.79 | 42.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 42.79 | 42.79 | 42.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 42.90 | 42.81 | 42.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 42.89 | 42.81 | 42.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 42.53 | 42.76 | 42.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 42.53 | 42.76 | 42.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 42.20 | 42.65 | 42.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 41.00 | 42.65 | 42.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 41.13 | 42.34 | 42.51 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 44.00 | 42.72 | 42.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 44.50 | 43.26 | 42.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 44.50 | 44.96 | 44.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 44.50 | 44.96 | 44.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 44.50 | 44.96 | 44.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 44.32 | 44.96 | 44.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 44.11 | 44.79 | 44.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 44.02 | 44.79 | 44.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 44.19 | 44.67 | 44.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:00:00 | 44.75 | 44.68 | 44.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 43.17 | 44.32 | 44.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 43.17 | 44.32 | 44.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 42.95 | 44.05 | 44.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 43.34 | 43.27 | 43.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 43.34 | 43.27 | 43.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 43.35 | 43.23 | 43.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 43.35 | 43.23 | 43.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 43.30 | 43.25 | 43.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 43.42 | 43.25 | 43.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 43.20 | 43.24 | 43.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:15:00 | 42.93 | 43.23 | 43.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 42.99 | 43.14 | 43.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 42.82 | 43.25 | 43.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 42.99 | 43.03 | 43.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 43.30 | 43.10 | 43.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 43.32 | 43.10 | 43.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 43.25 | 43.13 | 43.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 43.00 | 43.13 | 43.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 12:45:00 | 43.19 | 43.10 | 43.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 13:15:00 | 43.18 | 43.10 | 43.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 43.10 | 42.91 | 42.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 43.17 | 43.00 | 42.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 43.06 | 43.08 | 43.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 43.06 | 43.08 | 43.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 43.02 | 43.07 | 43.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:30:00 | 43.04 | 43.07 | 43.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 42.54 | 42.96 | 42.96 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 42.73 | 42.91 | 42.94 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 43.25 | 42.98 | 42.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 43.98 | 43.17 | 43.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 43.81 | 43.97 | 43.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 43.81 | 43.97 | 43.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 43.81 | 43.97 | 43.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 43.64 | 43.97 | 43.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 43.55 | 43.85 | 43.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 43.55 | 43.85 | 43.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 43.46 | 43.77 | 43.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:45:00 | 43.35 | 43.77 | 43.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 43.54 | 43.72 | 43.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:30:00 | 43.41 | 43.72 | 43.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 43.65 | 43.70 | 43.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 43.60 | 43.70 | 43.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 43.77 | 43.72 | 43.62 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 43.20 | 43.52 | 43.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 10:15:00 | 43.03 | 43.40 | 43.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 11:15:00 | 42.97 | 42.87 | 43.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:30:00 | 42.97 | 42.87 | 43.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 43.21 | 42.96 | 43.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 43.21 | 42.96 | 43.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 43.17 | 43.00 | 43.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:30:00 | 43.39 | 43.00 | 43.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 43.00 | 43.00 | 43.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 42.18 | 43.00 | 43.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 40.07 | 41.11 | 41.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 40.48 | 40.29 | 40.62 | SL hit (close>ema200) qty=0.50 sl=40.29 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 38.21 | 37.25 | 37.14 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 36.75 | 37.33 | 37.35 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 13:15:00 | 37.75 | 37.40 | 37.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 15:15:00 | 38.10 | 37.59 | 37.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 36.98 | 37.47 | 37.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 36.98 | 37.47 | 37.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 36.98 | 37.47 | 37.43 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 36.93 | 37.36 | 37.38 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 37.91 | 37.36 | 37.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 38.13 | 37.52 | 37.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 37.04 | 37.68 | 37.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 37.04 | 37.68 | 37.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 37.04 | 37.68 | 37.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 37.04 | 37.68 | 37.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 36.75 | 37.49 | 37.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 36.75 | 37.49 | 37.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 37.06 | 37.41 | 37.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 10:15:00 | 36.44 | 36.81 | 37.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 12:15:00 | 36.81 | 36.80 | 36.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 12:30:00 | 36.79 | 36.80 | 36.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 36.96 | 36.83 | 36.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:30:00 | 37.13 | 36.83 | 36.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 37.16 | 36.90 | 36.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 37.14 | 36.90 | 36.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 37.13 | 36.95 | 37.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:15:00 | 38.29 | 36.95 | 37.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 38.98 | 37.35 | 37.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 39.09 | 37.70 | 37.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 38.03 | 38.39 | 38.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 12:15:00 | 38.03 | 38.39 | 38.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 38.03 | 38.39 | 38.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 38.03 | 38.39 | 38.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 37.98 | 38.30 | 38.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 37.87 | 38.30 | 38.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 38.66 | 38.38 | 38.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 38.23 | 38.38 | 38.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 38.69 | 39.11 | 38.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 38.86 | 39.07 | 38.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 38.98 | 38.79 | 38.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 38.95 | 39.41 | 39.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 39.35 | 40.25 | 40.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 39.35 | 40.25 | 40.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 39.35 | 40.25 | 40.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 39.35 | 40.25 | 40.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 39.30 | 40.06 | 40.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 40.02 | 39.86 | 40.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 15:15:00 | 40.02 | 39.86 | 40.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 40.02 | 39.86 | 40.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 40.78 | 39.86 | 40.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 40.96 | 40.08 | 40.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 40.96 | 40.08 | 40.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 40.99 | 40.26 | 40.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 41.30 | 40.72 | 40.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 39.59 | 40.95 | 40.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 39.59 | 40.95 | 40.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 39.59 | 40.95 | 40.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 39.59 | 40.95 | 40.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 39.34 | 40.63 | 40.65 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 40.70 | 40.08 | 40.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 41.09 | 40.54 | 40.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 41.08 | 41.11 | 40.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:00:00 | 41.08 | 41.11 | 40.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 42.54 | 42.65 | 42.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 42.60 | 42.65 | 42.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-04 12:45:00 | 38.57 | 2025-06-04 14:15:00 | 39.09 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-06-20 14:45:00 | 39.97 | 2025-06-25 12:15:00 | 39.83 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-06-20 15:15:00 | 39.90 | 2025-06-25 12:15:00 | 39.83 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-07-02 09:30:00 | 39.66 | 2025-07-03 09:15:00 | 40.18 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-02 13:15:00 | 39.63 | 2025-07-03 09:15:00 | 40.18 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-07 11:45:00 | 40.69 | 2025-07-11 14:15:00 | 42.27 | STOP_HIT | 1.00 | 3.88% |
| BUY | retest2 | 2025-07-18 15:15:00 | 43.71 | 2025-07-22 10:15:00 | 42.90 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-07-21 10:30:00 | 43.67 | 2025-07-22 10:15:00 | 42.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-21 15:00:00 | 43.75 | 2025-07-22 10:15:00 | 42.90 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest1 | 2025-07-28 09:15:00 | 40.35 | 2025-07-30 11:15:00 | 38.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-07-28 09:15:00 | 40.35 | 2025-08-01 09:15:00 | 38.11 | STOP_HIT | 0.50 | 5.55% |
| SELL | retest2 | 2025-08-04 11:45:00 | 37.93 | 2025-08-04 15:15:00 | 38.34 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-04 12:15:00 | 37.96 | 2025-08-04 15:15:00 | 38.34 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-08-04 13:30:00 | 37.97 | 2025-08-04 15:15:00 | 38.34 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-08-25 09:30:00 | 42.04 | 2025-08-25 11:15:00 | 41.85 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-26 14:30:00 | 41.59 | 2025-09-01 10:15:00 | 41.73 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-09-12 10:15:00 | 50.03 | 2025-09-15 14:15:00 | 48.50 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-09-12 13:00:00 | 50.18 | 2025-09-15 14:15:00 | 48.50 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-09-12 15:00:00 | 50.09 | 2025-09-15 14:15:00 | 48.50 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-09-22 14:45:00 | 47.68 | 2025-09-24 09:15:00 | 49.39 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-23 09:45:00 | 47.68 | 2025-09-24 09:15:00 | 49.39 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-29 11:00:00 | 46.22 | 2025-10-07 11:15:00 | 46.20 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-09-30 10:15:00 | 46.17 | 2025-10-07 11:15:00 | 46.20 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-08 12:15:00 | 45.93 | 2025-10-08 15:15:00 | 45.63 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-14 10:00:00 | 44.55 | 2025-10-15 14:15:00 | 46.53 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-10-29 09:45:00 | 46.36 | 2025-10-29 13:15:00 | 47.25 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-31 12:15:00 | 47.72 | 2025-11-03 09:15:00 | 46.79 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-11-20 09:30:00 | 49.31 | 2025-11-21 12:15:00 | 48.08 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-11-20 12:45:00 | 49.00 | 2025-11-21 12:15:00 | 48.08 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-11-28 10:45:00 | 45.77 | 2025-12-01 14:15:00 | 46.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-11-28 12:30:00 | 45.74 | 2025-12-01 14:15:00 | 46.70 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-11-28 14:45:00 | 45.72 | 2025-12-01 14:15:00 | 46.70 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-03 12:15:00 | 46.29 | 2025-12-03 12:15:00 | 46.08 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-12-08 09:15:00 | 45.18 | 2025-12-11 13:15:00 | 45.25 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-12-16 09:15:00 | 46.52 | 2025-12-16 11:15:00 | 45.44 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest1 | 2025-12-31 13:30:00 | 48.42 | 2026-01-05 11:15:00 | 50.84 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-01-01 10:30:00 | 48.11 | 2026-01-05 11:15:00 | 50.52 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-01-01 14:15:00 | 48.99 | 2026-01-05 13:15:00 | 51.44 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-12-31 13:30:00 | 48.42 | 2026-01-06 09:15:00 | 53.26 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-01-01 10:30:00 | 48.11 | 2026-01-06 09:15:00 | 52.92 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-01-01 14:15:00 | 48.99 | 2026-01-06 12:15:00 | 51.00 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2026-01-16 11:15:00 | 45.24 | 2026-01-20 15:15:00 | 42.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 45.31 | 2026-01-20 15:15:00 | 43.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:45:00 | 45.26 | 2026-01-20 15:15:00 | 43.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 15:00:00 | 45.32 | 2026-01-20 15:15:00 | 43.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:15:00 | 45.24 | 2026-01-21 11:15:00 | 44.04 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2026-01-16 12:15:00 | 45.31 | 2026-01-21 11:15:00 | 44.04 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-01-16 13:45:00 | 45.26 | 2026-01-21 11:15:00 | 44.04 | STOP_HIT | 0.50 | 2.70% |
| SELL | retest2 | 2026-01-16 15:00:00 | 45.32 | 2026-01-21 11:15:00 | 44.04 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-01-22 09:45:00 | 44.10 | 2026-01-28 09:15:00 | 41.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 09:45:00 | 44.10 | 2026-01-28 12:15:00 | 42.38 | STOP_HIT | 0.50 | 3.90% |
| BUY | retest2 | 2026-02-05 13:00:00 | 44.75 | 2026-02-06 09:15:00 | 43.17 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-02-10 13:15:00 | 42.93 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-10 14:45:00 | 42.99 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-02-12 09:15:00 | 42.82 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-12 12:45:00 | 42.99 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-02-13 09:15:00 | 43.00 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-02-13 12:45:00 | 43.19 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2026-02-13 13:15:00 | 43.18 | 2026-02-18 15:15:00 | 43.10 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2026-03-02 09:15:00 | 42.18 | 2026-03-09 09:15:00 | 40.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 42.18 | 2026-03-10 12:15:00 | 40.48 | STOP_HIT | 0.50 | 4.03% |
| BUY | retest2 | 2026-04-13 10:30:00 | 38.86 | 2026-04-24 10:15:00 | 39.35 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2026-04-15 09:15:00 | 38.98 | 2026-04-24 10:15:00 | 39.35 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2026-04-20 10:00:00 | 38.95 | 2026-04-24 10:15:00 | 39.35 | STOP_HIT | 1.00 | 1.03% |
