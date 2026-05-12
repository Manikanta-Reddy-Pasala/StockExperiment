# Indian Oil Corporation Ltd. (IOC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 144.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 27 |
| ALERT3 | 135 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 62 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 45
- **Target hits / Stop hits / Partials:** 0 / 66 / 5
- **Avg / median % per leg:** 0.31% / -0.54%
- **Sum % (uncompounded):** 22.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 12 | 46.2% | 0 | 26 | 0 | -0.29% | -7.4% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.21% | -8.8% |
| BUY @ 3rd Alert (retest2) | 22 | 12 | 54.5% | 0 | 22 | 0 | 0.06% | 1.4% |
| SELL (all) | 45 | 14 | 31.1% | 0 | 40 | 5 | 0.66% | 29.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 14 | 31.1% | 0 | 40 | 5 | 0.66% | 29.5% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.21% | -8.8% |
| retest2 (combined) | 67 | 26 | 38.8% | 0 | 62 | 5 | 0.46% | 30.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 142.16 | 142.06 | 142.05 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 10:15:00 | 141.64 | 141.98 | 142.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 13:15:00 | 141.35 | 141.80 | 141.92 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 142.96 | 142.00 | 141.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 144.70 | 143.29 | 142.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 144.58 | 145.21 | 144.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 144.58 | 145.21 | 144.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 144.58 | 145.21 | 144.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 144.58 | 145.21 | 144.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 144.78 | 145.12 | 144.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 145.25 | 145.12 | 144.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 145.49 | 145.20 | 144.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:45:00 | 145.01 | 145.10 | 144.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 144.42 | 144.96 | 144.68 | SL hit (close<static) qty=1.00 sl=144.43 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 143.32 | 144.38 | 144.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 142.90 | 143.95 | 144.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 143.09 | 142.79 | 143.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:30:00 | 142.81 | 142.79 | 143.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 143.36 | 142.90 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 143.36 | 142.90 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 143.40 | 143.00 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 143.40 | 143.00 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 143.24 | 143.05 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 143.73 | 143.05 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 143.40 | 143.12 | 143.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 143.40 | 143.12 | 143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 143.00 | 143.10 | 143.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:15:00 | 143.59 | 143.10 | 143.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 143.31 | 143.14 | 143.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 143.55 | 143.14 | 143.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 142.35 | 142.98 | 143.21 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 144.49 | 143.33 | 143.32 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 143.40 | 143.66 | 143.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 142.82 | 143.49 | 143.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 143.62 | 143.51 | 143.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 143.62 | 143.51 | 143.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 143.62 | 143.51 | 143.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 143.62 | 143.51 | 143.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 143.63 | 143.53 | 143.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:45:00 | 143.74 | 143.53 | 143.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 143.05 | 143.44 | 143.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 142.88 | 143.43 | 143.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 142.94 | 143.37 | 143.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 143.92 | 143.48 | 143.52 | SL hit (close>static) qty=1.00 sl=143.66 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 143.70 | 143.54 | 143.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 144.21 | 143.68 | 143.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 143.59 | 143.76 | 143.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 143.59 | 143.76 | 143.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 143.59 | 143.76 | 143.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 143.59 | 143.76 | 143.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 142.96 | 143.60 | 143.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 142.96 | 143.60 | 143.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 143.25 | 143.53 | 143.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 142.63 | 143.37 | 143.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 143.00 | 142.51 | 142.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 12:15:00 | 143.00 | 142.51 | 142.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 143.00 | 142.51 | 142.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:45:00 | 142.89 | 142.51 | 142.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 143.05 | 142.62 | 142.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 143.23 | 142.62 | 142.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 143.25 | 142.87 | 142.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 143.26 | 142.87 | 142.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 142.23 | 142.71 | 142.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:30:00 | 142.58 | 142.71 | 142.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 140.40 | 140.36 | 141.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 140.40 | 140.36 | 141.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 141.18 | 140.53 | 141.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:45:00 | 140.55 | 140.49 | 141.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 140.59 | 140.52 | 141.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 140.45 | 140.08 | 140.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 140.59 | 140.24 | 140.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 140.62 | 140.32 | 140.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:15:00 | 140.87 | 140.32 | 140.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 140.72 | 140.40 | 140.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:30:00 | 140.56 | 140.40 | 140.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 142.21 | 140.82 | 140.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 142.21 | 140.82 | 140.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 12:15:00 | 142.83 | 141.71 | 141.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 142.61 | 142.66 | 142.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 142.26 | 142.59 | 142.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 142.26 | 142.59 | 142.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:45:00 | 142.04 | 142.59 | 142.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 142.10 | 142.49 | 142.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 145.20 | 142.49 | 142.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 142.02 | 144.27 | 143.61 | SL hit (close<static) qty=1.00 sl=142.10 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 140.80 | 142.96 | 143.22 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 142.36 | 141.92 | 141.89 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 141.63 | 141.83 | 141.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 09:15:00 | 140.60 | 141.55 | 141.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 140.83 | 140.83 | 141.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:30:00 | 140.75 | 140.83 | 141.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 139.89 | 140.63 | 141.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 139.35 | 140.46 | 140.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 139.44 | 140.29 | 140.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 139.40 | 140.29 | 140.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:45:00 | 139.47 | 138.51 | 138.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 139.94 | 138.79 | 138.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 143.95 | 139.99 | 139.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 143.95 | 139.99 | 139.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 146.49 | 144.25 | 143.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 147.00 | 147.17 | 145.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 147.00 | 147.17 | 145.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 146.54 | 146.92 | 146.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 146.34 | 146.92 | 146.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 147.61 | 147.51 | 147.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 147.21 | 147.51 | 147.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 147.60 | 147.85 | 147.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 148.75 | 147.85 | 147.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 150.35 | 151.73 | 151.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 150.35 | 151.73 | 151.82 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 152.10 | 151.63 | 151.62 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 151.15 | 151.53 | 151.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 150.89 | 151.40 | 151.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 151.45 | 151.41 | 151.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 12:00:00 | 151.45 | 151.41 | 151.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 151.60 | 151.45 | 151.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 151.57 | 151.45 | 151.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 150.60 | 151.28 | 151.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 151.10 | 151.28 | 151.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 151.17 | 151.07 | 151.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 151.15 | 151.07 | 151.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 150.63 | 150.98 | 151.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 150.12 | 150.98 | 151.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 151.47 | 150.91 | 150.92 | SL hit (close>static) qty=1.00 sl=151.40 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 15:15:00 | 151.45 | 151.02 | 150.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 151.73 | 151.16 | 151.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 151.40 | 151.61 | 151.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 151.40 | 151.61 | 151.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 151.40 | 151.61 | 151.37 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 150.80 | 151.29 | 151.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 150.64 | 151.16 | 151.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 150.44 | 150.29 | 150.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:30:00 | 150.23 | 150.29 | 150.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 150.55 | 150.34 | 150.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 150.55 | 150.34 | 150.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 150.75 | 150.42 | 150.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 150.76 | 150.42 | 150.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 150.82 | 150.50 | 150.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 150.82 | 150.50 | 150.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 151.04 | 150.61 | 150.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 151.04 | 150.61 | 150.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 151.47 | 150.87 | 150.80 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 151.10 | 151.41 | 151.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 149.16 | 150.96 | 151.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 147.76 | 147.50 | 148.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 147.76 | 147.50 | 148.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 148.15 | 147.74 | 148.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 147.37 | 147.74 | 148.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 146.90 | 147.57 | 148.07 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 149.17 | 148.34 | 148.28 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 144.60 | 147.68 | 148.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 144.25 | 146.01 | 146.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 142.16 | 141.87 | 143.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 142.16 | 141.87 | 143.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 139.81 | 141.65 | 142.96 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 143.00 | 141.38 | 141.29 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 140.82 | 141.96 | 142.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 140.11 | 141.23 | 141.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 141.09 | 140.56 | 140.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 141.09 | 140.56 | 140.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 141.09 | 140.56 | 140.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 141.09 | 140.56 | 140.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 141.19 | 140.69 | 140.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 141.19 | 140.69 | 140.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 141.18 | 140.79 | 140.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:15:00 | 141.35 | 140.79 | 140.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 141.96 | 141.16 | 141.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 142.12 | 141.68 | 141.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 141.37 | 141.78 | 141.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 141.37 | 141.78 | 141.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 141.37 | 141.78 | 141.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 141.37 | 141.78 | 141.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 141.91 | 141.80 | 141.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 142.37 | 141.80 | 141.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 141.97 | 142.07 | 141.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 140.54 | 141.62 | 141.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 140.54 | 141.62 | 141.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 140.16 | 140.92 | 141.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 140.09 | 139.85 | 140.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 140.09 | 139.85 | 140.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 139.25 | 139.76 | 140.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 138.79 | 139.61 | 140.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 138.94 | 139.51 | 139.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 138.33 | 139.45 | 139.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:45:00 | 138.95 | 139.21 | 139.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 138.27 | 138.82 | 139.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 137.98 | 138.72 | 139.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 137.93 | 137.79 | 138.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 139.61 | 138.37 | 138.47 | SL hit (close>static) qty=1.00 sl=139.49 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 139.56 | 138.61 | 138.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 140.22 | 138.93 | 138.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 13:15:00 | 140.10 | 140.20 | 139.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 14:00:00 | 140.10 | 140.20 | 139.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 140.41 | 140.30 | 139.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 139.97 | 140.30 | 139.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 139.90 | 140.22 | 139.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 139.90 | 140.22 | 139.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 139.77 | 140.13 | 139.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 139.77 | 140.13 | 139.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 139.63 | 140.03 | 139.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 139.84 | 140.03 | 139.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 139.32 | 139.86 | 139.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 139.32 | 139.86 | 139.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 139.62 | 139.81 | 139.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 138.84 | 139.62 | 139.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 140.04 | 139.61 | 139.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 140.04 | 139.61 | 139.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 140.04 | 139.61 | 139.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 140.04 | 139.61 | 139.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 139.65 | 139.62 | 139.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 140.00 | 139.62 | 139.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 140.00 | 139.69 | 139.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 141.03 | 139.69 | 139.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 141.54 | 140.06 | 139.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 142.94 | 141.50 | 141.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 142.96 | 143.54 | 142.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 142.96 | 143.54 | 142.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 142.95 | 143.42 | 142.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 142.85 | 143.42 | 142.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 142.63 | 143.27 | 142.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 142.60 | 143.27 | 142.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 142.81 | 143.17 | 142.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 142.84 | 143.17 | 142.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 143.08 | 143.16 | 142.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 144.10 | 143.23 | 142.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 147.28 | 147.84 | 147.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 147.28 | 147.84 | 147.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 146.80 | 147.64 | 147.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 147.70 | 147.51 | 147.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 147.70 | 147.51 | 147.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 147.70 | 147.51 | 147.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 147.70 | 147.51 | 147.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 147.81 | 147.57 | 147.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 147.62 | 147.57 | 147.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 147.45 | 147.54 | 147.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 147.71 | 147.54 | 147.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 147.35 | 147.50 | 147.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:00:00 | 147.35 | 147.50 | 147.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 147.15 | 147.43 | 147.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 147.52 | 147.43 | 147.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 148.76 | 145.97 | 146.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 148.22 | 145.97 | 146.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 148.91 | 146.56 | 146.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 149.74 | 146.56 | 146.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 148.51 | 146.95 | 146.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 149.45 | 147.77 | 147.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 11:15:00 | 149.75 | 149.81 | 149.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 11:45:00 | 149.82 | 149.81 | 149.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 149.20 | 149.68 | 149.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:30:00 | 149.15 | 149.68 | 149.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 149.79 | 149.70 | 149.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 150.43 | 149.69 | 149.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 148.99 | 149.40 | 149.18 | SL hit (close<static) qty=1.00 sl=149.01 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 152.46 | 154.26 | 154.43 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 155.40 | 154.42 | 154.35 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 153.30 | 154.20 | 154.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 152.66 | 153.89 | 154.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 153.94 | 153.48 | 153.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 153.94 | 153.48 | 153.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 153.94 | 153.48 | 153.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 153.94 | 153.48 | 153.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 153.97 | 153.58 | 153.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 153.97 | 153.58 | 153.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 154.35 | 153.73 | 153.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 154.65 | 153.73 | 153.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 153.31 | 153.65 | 153.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 153.05 | 153.41 | 153.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:00:00 | 153.04 | 153.44 | 153.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 153.03 | 153.53 | 153.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 154.03 | 153.53 | 153.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 154.03 | 153.53 | 153.47 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 151.36 | 153.28 | 153.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 149.94 | 151.52 | 152.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 153.71 | 151.28 | 151.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 153.71 | 151.28 | 151.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 153.71 | 151.28 | 151.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 153.71 | 151.28 | 151.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 153.79 | 151.78 | 151.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 153.79 | 151.78 | 151.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 153.06 | 152.04 | 151.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 154.29 | 152.49 | 152.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 167.00 | 167.35 | 165.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:15:00 | 168.62 | 167.35 | 165.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 168.64 | 168.59 | 167.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 167.26 | 168.16 | 167.76 | SL hit (close<ema400) qty=1.00 sl=167.76 alert=retest1 |

### Cycle 38 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 171.04 | 172.01 | 172.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 169.50 | 170.53 | 171.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 165.15 | 164.59 | 165.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 165.15 | 164.59 | 165.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 165.15 | 164.59 | 165.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 165.48 | 164.59 | 165.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 165.90 | 164.85 | 165.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 165.90 | 164.85 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 165.68 | 165.02 | 165.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 165.71 | 165.02 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 164.85 | 164.98 | 165.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 164.68 | 164.98 | 165.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 164.19 | 164.92 | 165.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 13:15:00 | 164.33 | 162.98 | 162.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 164.33 | 162.98 | 162.92 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 162.70 | 163.01 | 163.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 162.15 | 162.84 | 162.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 162.79 | 162.70 | 162.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 162.79 | 162.70 | 162.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 162.79 | 162.70 | 162.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 162.76 | 162.70 | 162.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 162.72 | 162.70 | 162.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 162.82 | 162.70 | 162.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 163.20 | 162.80 | 162.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 163.20 | 162.80 | 162.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 163.26 | 162.89 | 162.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 163.77 | 163.07 | 162.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 162.96 | 163.15 | 163.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 10:15:00 | 162.96 | 163.15 | 163.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 162.96 | 163.15 | 163.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 162.96 | 163.15 | 163.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 162.86 | 163.09 | 163.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:15:00 | 162.71 | 163.09 | 163.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 162.39 | 162.95 | 162.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 161.65 | 162.69 | 162.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 162.65 | 162.20 | 162.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 162.65 | 162.20 | 162.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 162.65 | 162.20 | 162.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 162.65 | 162.20 | 162.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 161.99 | 162.16 | 162.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 162.70 | 162.16 | 162.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 162.78 | 162.28 | 162.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 162.62 | 162.28 | 162.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 163.08 | 162.44 | 162.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 163.08 | 162.44 | 162.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 162.95 | 162.54 | 162.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 163.80 | 162.54 | 162.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 164.97 | 163.03 | 162.79 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 161.25 | 162.91 | 162.98 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 164.27 | 162.78 | 162.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 164.36 | 163.10 | 162.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 166.96 | 167.03 | 165.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 166.96 | 167.03 | 165.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 161.98 | 166.76 | 166.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 161.98 | 166.76 | 166.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 162.18 | 165.85 | 166.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 161.71 | 163.99 | 165.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 162.61 | 162.05 | 163.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 162.61 | 162.05 | 163.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 164.34 | 162.58 | 163.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 164.34 | 162.58 | 163.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 163.50 | 162.76 | 163.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 163.46 | 162.76 | 163.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:30:00 | 163.45 | 163.34 | 163.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 164.33 | 163.57 | 163.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 164.33 | 163.57 | 163.55 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 163.15 | 163.49 | 163.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 162.60 | 163.32 | 163.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 161.70 | 160.75 | 161.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 161.70 | 160.75 | 161.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 161.70 | 160.75 | 161.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 161.94 | 160.75 | 161.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 161.80 | 160.96 | 161.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 161.85 | 160.96 | 161.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 161.28 | 161.21 | 161.50 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 162.16 | 161.74 | 161.69 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 161.36 | 161.63 | 161.65 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 164.47 | 162.16 | 161.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 165.54 | 162.84 | 162.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 165.01 | 165.58 | 164.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:00:00 | 165.01 | 165.58 | 164.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 165.24 | 165.55 | 164.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 164.99 | 165.55 | 164.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 165.52 | 166.09 | 165.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 165.23 | 166.09 | 165.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 163.89 | 165.65 | 165.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 163.89 | 165.65 | 165.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 165.24 | 165.57 | 165.47 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 164.70 | 165.39 | 165.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 162.67 | 164.85 | 165.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 164.23 | 163.67 | 164.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 14:15:00 | 164.23 | 163.67 | 164.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 164.23 | 163.67 | 164.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 164.23 | 163.67 | 164.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 163.90 | 163.71 | 164.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 162.92 | 163.71 | 164.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 163.53 | 163.68 | 164.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:00:00 | 162.49 | 163.39 | 163.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:15:00 | 162.33 | 163.25 | 163.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 162.50 | 163.11 | 163.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 159.27 | 158.06 | 157.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 159.27 | 158.06 | 157.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 159.55 | 158.35 | 158.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 160.05 | 160.12 | 159.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 160.05 | 160.12 | 159.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 160.04 | 160.68 | 159.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:45:00 | 160.04 | 160.68 | 159.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 160.28 | 160.60 | 159.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 159.91 | 160.60 | 159.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 159.90 | 160.46 | 159.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 159.74 | 160.46 | 159.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 160.33 | 160.44 | 159.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:15:00 | 159.79 | 160.44 | 159.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 160.95 | 160.54 | 160.06 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 159.13 | 159.76 | 159.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 158.19 | 159.32 | 159.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 159.35 | 159.05 | 159.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 159.35 | 159.05 | 159.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 159.35 | 159.05 | 159.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 157.68 | 158.70 | 159.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:00:00 | 157.77 | 158.66 | 159.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 161.04 | 159.18 | 159.20 | SL hit (close>static) qty=1.00 sl=159.90 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 160.00 | 159.35 | 159.27 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 13:15:00 | 158.58 | 159.14 | 159.20 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 160.16 | 159.26 | 159.23 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 158.62 | 159.13 | 159.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 157.76 | 158.86 | 159.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 157.85 | 157.52 | 158.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:45:00 | 157.75 | 157.52 | 158.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 157.76 | 157.57 | 158.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 157.68 | 157.57 | 158.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 158.25 | 157.71 | 158.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:00:00 | 158.25 | 157.71 | 158.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 157.94 | 157.75 | 158.15 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 161.76 | 159.00 | 158.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 163.90 | 159.98 | 159.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 162.84 | 162.90 | 161.88 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 160.18 | 161.82 | 161.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 159.90 | 161.44 | 161.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 161.33 | 160.79 | 161.25 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 164.42 | 161.92 | 161.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 164.88 | 162.51 | 161.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 11:15:00 | 174.58 | 174.63 | 172.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 12:00:00 | 174.58 | 174.63 | 172.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 175.61 | 178.81 | 178.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 175.61 | 178.81 | 178.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 177.40 | 178.53 | 178.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 177.50 | 178.53 | 178.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 177.57 | 178.34 | 178.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 176.69 | 177.65 | 177.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 176.69 | 177.65 | 177.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 173.91 | 176.48 | 177.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 175.18 | 175.15 | 176.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 175.18 | 175.15 | 176.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 175.55 | 174.47 | 175.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 175.55 | 174.47 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 175.75 | 174.73 | 175.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 176.74 | 174.73 | 175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 177.43 | 175.64 | 175.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 177.80 | 176.61 | 176.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 175.72 | 177.02 | 176.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 175.72 | 177.02 | 176.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 176.79 | 176.97 | 176.47 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 174.03 | 176.09 | 176.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 173.84 | 175.64 | 175.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 174.24 | 174.14 | 174.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 174.24 | 174.14 | 174.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 175.60 | 174.27 | 174.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 176.00 | 174.27 | 174.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 176.54 | 174.72 | 174.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 176.54 | 174.72 | 174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 175.79 | 175.19 | 175.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 176.40 | 175.58 | 175.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 179.81 | 185.41 | 184.32 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 179.36 | 183.35 | 183.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 178.24 | 182.33 | 183.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 174.65 | 173.36 | 176.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 174.65 | 173.36 | 176.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 161.73 | 162.36 | 165.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:00:00 | 160.05 | 161.43 | 164.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 160.45 | 160.81 | 163.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:00:00 | 160.33 | 160.88 | 162.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 160.38 | 160.55 | 161.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.05 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.43 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.31 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 152.36 | 155.91 | 158.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 147.39 | 147.37 | 149.86 | SL hit (close>ema200) qty=0.50 sl=147.37 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 142.12 | 135.26 | 134.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 144.18 | 138.18 | 135.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 141.15 | 141.21 | 138.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 143.04 | 141.84 | 140.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 13:30:00 | 143.07 | 142.45 | 141.06 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 142.96 | 142.56 | 141.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 139.19 | 141.95 | 141.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 139.19 | 141.95 | 141.19 | SL hit (close<ema400) qty=1.00 sl=141.19 alert=retest1 |

### Cycle 68 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 140.14 | 140.80 | 140.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 140.96 | 140.83 | 140.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 145.08 | 141.74 | 141.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 144.35 | 144.50 | 143.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 13:00:00 | 144.35 | 144.50 | 143.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 144.28 | 144.43 | 143.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 144.64 | 144.41 | 143.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 144.56 | 144.57 | 143.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 145.56 | 146.56 | 146.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 145.56 | 146.56 | 146.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 145.41 | 146.33 | 146.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 145.39 | 144.30 | 145.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 146.04 | 144.30 | 145.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 145.73 | 144.59 | 145.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 145.69 | 144.59 | 145.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 146.22 | 145.40 | 145.35 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 144.78 | 145.31 | 145.37 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 146.37 | 145.58 | 145.48 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 144.45 | 145.42 | 145.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 142.35 | 144.81 | 145.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 143.12 | 142.90 | 143.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 142.17 | 143.00 | 143.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 142.35 | 142.76 | 143.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 141.20 | 142.76 | 143.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:30:00 | 142.37 | 142.13 | 142.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 144.08 | 142.56 | 142.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 144.14 | 143.08 | 143.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 144.14 | 143.08 | 143.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 146.75 | 143.93 | 143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 145.23 | 146.26 | 145.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 145.05 | 146.26 | 145.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 145.48 | 146.10 | 145.43 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 14:15:00 | 142.08 | 2025-05-13 09:15:00 | 142.16 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-05-13 09:15:00 | 141.85 | 2025-05-13 09:15:00 | 142.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-05-19 09:15:00 | 145.25 | 2025-05-19 13:15:00 | 144.42 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-05-19 10:00:00 | 145.49 | 2025-05-19 13:15:00 | 144.42 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-05-19 12:45:00 | 145.01 | 2025-05-19 13:15:00 | 144.42 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-05-28 09:15:00 | 142.88 | 2025-05-28 10:15:00 | 143.92 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-28 10:15:00 | 142.94 | 2025-05-28 10:15:00 | 143.92 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-05 10:45:00 | 140.55 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-06-05 12:15:00 | 140.59 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-06 10:30:00 | 140.45 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-06 13:00:00 | 140.59 | 2025-06-09 09:15:00 | 142.21 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-11 09:15:00 | 145.20 | 2025-06-12 09:15:00 | 142.02 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-12 10:15:00 | 142.40 | 2025-06-13 09:15:00 | 140.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-19 10:30:00 | 139.35 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-06-19 11:30:00 | 139.44 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-06-19 12:15:00 | 139.40 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-06-23 13:45:00 | 139.47 | 2025-06-24 09:15:00 | 143.95 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-07-04 09:15:00 | 148.75 | 2025-07-09 14:15:00 | 150.35 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-07-14 11:15:00 | 150.12 | 2025-07-15 14:15:00 | 151.47 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-21 09:15:00 | 142.37 | 2025-08-22 09:15:00 | 140.54 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-08-21 13:15:00 | 141.97 | 2025-08-22 09:15:00 | 140.54 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-26 10:30:00 | 138.79 | 2025-09-01 14:15:00 | 139.61 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-08-26 15:00:00 | 138.94 | 2025-09-01 14:15:00 | 139.61 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-08-28 09:15:00 | 138.33 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-28 12:45:00 | 138.95 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-08-29 12:15:00 | 137.98 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-01 11:45:00 | 137.93 | 2025-09-01 15:15:00 | 139.56 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-09-15 15:00:00 | 144.10 | 2025-09-24 13:15:00 | 147.28 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-10-03 09:15:00 | 150.43 | 2025-10-03 11:15:00 | 148.99 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-03 13:15:00 | 149.87 | 2025-10-13 09:15:00 | 152.46 | STOP_HIT | 1.00 | 1.73% |
| SELL | retest2 | 2025-10-16 09:30:00 | 153.05 | 2025-10-20 13:15:00 | 154.03 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-16 13:00:00 | 153.04 | 2025-10-20 13:15:00 | 154.03 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-17 11:30:00 | 153.03 | 2025-10-20 13:15:00 | 154.03 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2025-11-04 09:15:00 | 168.62 | 2025-11-07 09:15:00 | 167.26 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-07 13:15:00 | 169.35 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-11-10 09:15:00 | 170.79 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-11-10 11:15:00 | 169.40 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-11-10 13:00:00 | 169.28 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-11-11 11:30:00 | 170.12 | 2025-11-18 10:15:00 | 171.04 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-11-26 13:15:00 | 164.68 | 2025-12-03 13:15:00 | 164.33 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-11-27 09:30:00 | 164.19 | 2025-12-03 13:15:00 | 164.33 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-12-22 11:15:00 | 163.46 | 2025-12-23 09:15:00 | 164.33 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-22 14:30:00 | 163.45 | 2025-12-23 09:15:00 | 164.33 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-07 12:00:00 | 162.49 | 2026-01-14 11:15:00 | 159.27 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2026-01-07 13:15:00 | 162.33 | 2026-01-14 11:15:00 | 159.27 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2026-01-07 15:15:00 | 162.50 | 2026-01-14 11:15:00 | 159.27 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2026-01-21 10:30:00 | 157.68 | 2026-01-22 09:15:00 | 161.04 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-01-21 14:00:00 | 157.77 | 2026-01-22 09:15:00 | 161.04 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-02-12 13:15:00 | 177.50 | 2026-02-13 10:15:00 | 176.69 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-12 14:45:00 | 177.57 | 2026-02-13 10:15:00 | 176.69 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-03-10 14:00:00 | 160.05 | 2026-03-16 09:15:00 | 152.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:45:00 | 160.45 | 2026-03-16 09:15:00 | 152.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 160.33 | 2026-03-16 09:15:00 | 152.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 160.38 | 2026-03-16 09:15:00 | 152.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 14:00:00 | 160.05 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 7.91% |
| SELL | retest2 | 2026-03-11 10:45:00 | 160.45 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.14% |
| SELL | retest2 | 2026-03-11 15:00:00 | 160.33 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.07% |
| SELL | retest2 | 2026-03-12 14:30:00 | 160.38 | 2026-03-18 10:15:00 | 147.39 | STOP_HIT | 0.50 | 8.10% |
| SELL | retest2 | 2026-03-23 09:15:00 | 140.83 | 2026-04-02 09:15:00 | 133.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 140.83 | 2026-04-02 14:15:00 | 134.38 | STOP_HIT | 0.50 | 4.58% |
| BUY | retest1 | 2026-04-10 09:30:00 | 143.04 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest1 | 2026-04-10 13:30:00 | 143.07 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest1 | 2026-04-10 15:00:00 | 142.96 | 2026-04-13 09:15:00 | 139.19 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-04-13 10:15:00 | 139.72 | 2026-04-13 13:15:00 | 140.14 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2026-04-13 10:45:00 | 139.70 | 2026-04-13 13:15:00 | 140.14 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-17 09:15:00 | 144.64 | 2026-04-23 13:15:00 | 145.56 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2026-04-17 09:45:00 | 144.56 | 2026-04-23 13:15:00 | 145.56 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-05-04 13:15:00 | 142.17 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-05-04 14:30:00 | 142.35 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-05 09:15:00 | 141.20 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-05-05 14:30:00 | 142.37 | 2026-05-06 11:15:00 | 144.14 | STOP_HIT | 1.00 | -1.24% |
