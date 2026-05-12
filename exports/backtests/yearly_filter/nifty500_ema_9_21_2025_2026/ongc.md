# Oil & Natural Gas Corporation Ltd. (ONGC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 279.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 76 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 23 |
| ALERT3 | 154 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 41
- **Target hits / Stop hits / Partials:** 0 / 54 / 3
- **Avg / median % per leg:** -0.09% / -0.68%
- **Sum % (uncompounded):** -5.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 9 | 30.0% | 0 | 30 | 0 | -0.31% | -9.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 9 | 30.0% | 0 | 30 | 0 | -0.31% | -9.3% |
| SELL (all) | 27 | 7 | 25.9% | 0 | 24 | 3 | 0.15% | 4.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 7 | 25.9% | 0 | 24 | 3 | 0.15% | 4.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 16 | 28.1% | 0 | 54 | 3 | -0.09% | -5.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 241.58 | 237.02 | 236.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 243.10 | 239.04 | 237.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 241.30 | 242.03 | 240.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 241.33 | 242.03 | 240.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 240.89 | 241.59 | 240.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 243.60 | 241.47 | 240.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 245.75 | 247.74 | 248.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 245.75 | 247.74 | 248.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 243.68 | 246.54 | 247.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 12:15:00 | 243.96 | 243.68 | 245.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 243.96 | 243.68 | 245.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 246.49 | 244.40 | 244.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 246.35 | 244.40 | 244.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 246.39 | 245.39 | 245.32 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 243.76 | 245.23 | 245.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 242.40 | 243.94 | 244.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 243.39 | 243.17 | 243.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 243.39 | 243.17 | 243.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 243.39 | 243.17 | 243.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 242.07 | 242.90 | 243.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:00:00 | 242.22 | 242.76 | 243.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 242.18 | 242.82 | 243.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:45:00 | 242.20 | 242.70 | 243.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 243.18 | 242.79 | 243.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 243.18 | 242.79 | 243.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 244.45 | 243.13 | 243.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 244.45 | 243.13 | 243.45 | SL hit (close>static) qty=1.00 sl=244.29 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 238.73 | 238.19 | 238.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 239.47 | 238.59 | 238.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 247.65 | 248.41 | 246.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 247.65 | 248.41 | 246.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 252.77 | 254.29 | 252.25 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 250.54 | 251.88 | 252.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 250.00 | 251.27 | 251.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 11:15:00 | 251.21 | 250.58 | 251.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 11:15:00 | 251.21 | 250.58 | 251.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 251.21 | 250.58 | 251.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 251.21 | 250.58 | 251.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 250.17 | 250.49 | 251.13 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 252.63 | 251.40 | 251.39 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 251.08 | 251.43 | 251.44 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 251.82 | 251.51 | 251.47 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 251.01 | 251.41 | 251.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 09:15:00 | 246.67 | 250.41 | 250.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 12:15:00 | 243.07 | 242.96 | 244.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 243.07 | 242.96 | 244.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 244.67 | 243.41 | 244.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 244.67 | 243.41 | 244.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 245.02 | 243.73 | 244.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 244.50 | 243.73 | 244.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 244.29 | 243.84 | 244.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 10:45:00 | 243.16 | 243.91 | 244.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:00:00 | 243.58 | 243.97 | 244.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 242.92 | 243.72 | 244.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 11:00:00 | 243.56 | 243.90 | 244.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 244.31 | 243.77 | 244.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 244.31 | 243.77 | 244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 244.00 | 243.82 | 244.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 244.40 | 243.82 | 244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 243.27 | 243.71 | 243.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 242.50 | 243.71 | 243.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 242.72 | 243.23 | 243.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 15:15:00 | 243.19 | 243.23 | 243.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 242.00 | 242.85 | 243.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 244.60 | 242.35 | 242.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 244.60 | 242.35 | 242.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 245.94 | 243.07 | 243.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 245.94 | 243.07 | 243.04 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 242.00 | 243.53 | 243.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 241.40 | 242.83 | 243.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 242.29 | 242.17 | 242.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 242.29 | 242.17 | 242.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 243.45 | 242.43 | 242.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 243.45 | 242.43 | 242.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 243.30 | 242.60 | 242.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 243.56 | 242.60 | 242.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 243.44 | 242.98 | 243.01 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 243.98 | 243.18 | 243.10 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 243.00 | 243.22 | 243.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 241.55 | 242.89 | 243.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 242.44 | 242.16 | 242.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 242.44 | 242.16 | 242.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 242.44 | 242.16 | 242.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 242.18 | 242.16 | 242.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 242.59 | 242.25 | 242.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 242.59 | 242.25 | 242.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 243.49 | 242.50 | 242.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 243.49 | 242.50 | 242.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 243.10 | 242.62 | 242.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:15:00 | 244.15 | 242.62 | 242.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 244.23 | 242.94 | 242.80 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 242.74 | 243.18 | 243.24 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 244.25 | 243.40 | 243.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 244.75 | 243.98 | 243.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 245.10 | 245.46 | 244.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 245.10 | 245.46 | 244.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 245.10 | 245.46 | 244.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 244.94 | 245.46 | 244.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 245.00 | 245.28 | 244.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 244.64 | 245.28 | 244.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 245.19 | 245.26 | 244.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 245.29 | 245.26 | 244.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 245.00 | 245.17 | 244.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 244.85 | 245.17 | 244.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 244.94 | 245.12 | 244.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 244.87 | 245.12 | 244.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 244.80 | 245.06 | 244.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 245.18 | 245.06 | 244.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 244.73 | 244.99 | 244.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 244.73 | 244.99 | 244.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 245.51 | 245.10 | 244.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 245.93 | 245.10 | 244.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 245.73 | 245.44 | 245.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:45:00 | 245.80 | 245.44 | 245.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 244.29 | 245.21 | 245.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 244.29 | 245.21 | 245.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 242.61 | 244.54 | 244.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 241.90 | 240.68 | 241.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 241.90 | 240.68 | 241.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 241.90 | 240.68 | 241.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 241.90 | 240.68 | 241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 241.50 | 240.85 | 241.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 241.73 | 240.85 | 241.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 241.32 | 240.94 | 241.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 241.07 | 240.94 | 241.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 241.26 | 241.01 | 241.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 241.58 | 241.01 | 241.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 241.66 | 241.14 | 241.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 241.66 | 241.14 | 241.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 241.10 | 241.13 | 241.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 242.08 | 241.13 | 241.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 240.29 | 240.96 | 241.42 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 242.90 | 241.76 | 241.70 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 239.80 | 241.43 | 241.59 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 242.41 | 241.66 | 241.65 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 241.00 | 241.53 | 241.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 240.25 | 241.27 | 241.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 13:15:00 | 237.84 | 237.83 | 239.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 14:00:00 | 237.84 | 237.83 | 239.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 234.00 | 233.26 | 234.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 234.00 | 233.26 | 234.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 233.55 | 233.32 | 233.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 234.49 | 233.32 | 233.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 233.58 | 233.37 | 233.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 232.52 | 233.21 | 233.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 232.80 | 233.07 | 233.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 234.36 | 233.60 | 233.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 234.36 | 233.60 | 233.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 235.28 | 233.95 | 233.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 236.01 | 237.63 | 236.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 236.01 | 237.63 | 236.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 236.01 | 237.63 | 236.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 235.03 | 237.63 | 236.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 236.25 | 237.36 | 236.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 235.76 | 237.36 | 236.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 237.10 | 237.30 | 236.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 237.22 | 237.26 | 236.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 237.36 | 237.30 | 236.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:00:00 | 237.46 | 237.30 | 236.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 237.32 | 237.10 | 236.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 237.35 | 237.54 | 237.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 236.94 | 237.54 | 237.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 238.20 | 237.88 | 237.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 238.32 | 237.89 | 237.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 238.50 | 237.96 | 237.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 236.75 | 238.19 | 238.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 236.75 | 238.19 | 238.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 236.62 | 237.88 | 238.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 236.58 | 236.55 | 237.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 236.58 | 236.55 | 237.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 236.58 | 236.55 | 237.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 235.18 | 236.63 | 237.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 236.35 | 234.54 | 234.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 236.35 | 234.54 | 234.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 237.20 | 235.27 | 234.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 239.30 | 239.52 | 237.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 239.30 | 239.52 | 237.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 239.07 | 239.78 | 238.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 239.07 | 239.78 | 238.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 239.02 | 239.63 | 238.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 238.96 | 239.63 | 238.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 238.94 | 239.49 | 238.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 238.60 | 239.49 | 238.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 236.71 | 238.88 | 238.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 236.71 | 238.88 | 238.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 236.88 | 238.48 | 238.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 236.66 | 238.48 | 238.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 237.25 | 238.23 | 238.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 236.17 | 237.57 | 237.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 234.01 | 233.44 | 234.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 234.01 | 233.44 | 234.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 234.01 | 233.44 | 234.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 234.01 | 233.44 | 234.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 233.75 | 232.25 | 232.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 234.03 | 232.25 | 232.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 234.38 | 232.68 | 232.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 234.51 | 232.68 | 232.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 233.78 | 233.05 | 232.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 234.65 | 233.68 | 233.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 233.45 | 233.64 | 233.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 233.45 | 233.64 | 233.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 232.86 | 233.48 | 233.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 232.86 | 233.48 | 233.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 233.45 | 233.48 | 233.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 233.82 | 233.48 | 233.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 234.04 | 233.39 | 233.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 232.45 | 233.20 | 233.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 232.45 | 233.20 | 233.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 232.20 | 232.72 | 232.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 234.43 | 232.95 | 233.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 234.43 | 232.95 | 233.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 234.43 | 232.95 | 233.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 234.43 | 232.95 | 233.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 234.69 | 233.30 | 233.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 234.83 | 233.83 | 233.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 235.55 | 236.04 | 235.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 235.55 | 236.04 | 235.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 235.03 | 235.75 | 235.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 235.03 | 235.75 | 235.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 234.60 | 235.52 | 235.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 234.60 | 235.52 | 235.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 235.00 | 235.42 | 235.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 235.62 | 235.46 | 235.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 235.58 | 235.44 | 235.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 12:15:00 | 242.45 | 244.36 | 244.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 242.45 | 244.36 | 244.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 241.68 | 243.55 | 243.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 243.45 | 243.07 | 243.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 243.45 | 243.07 | 243.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 244.00 | 243.26 | 243.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 244.00 | 243.26 | 243.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 243.62 | 243.33 | 243.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 243.10 | 243.33 | 243.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 246.35 | 243.88 | 243.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 246.35 | 243.88 | 243.78 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 243.45 | 244.21 | 244.29 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 248.33 | 244.86 | 244.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 248.85 | 247.74 | 247.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 247.85 | 247.95 | 247.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 247.85 | 247.95 | 247.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 247.85 | 247.95 | 247.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 247.85 | 247.95 | 247.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 253.05 | 254.14 | 252.44 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 250.85 | 252.26 | 252.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 250.37 | 251.65 | 252.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 254.00 | 251.94 | 252.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 254.00 | 251.94 | 252.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 254.00 | 251.94 | 252.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 254.00 | 251.94 | 252.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 253.90 | 252.33 | 252.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 256.51 | 254.23 | 253.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 254.05 | 254.43 | 253.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 254.05 | 254.43 | 253.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 254.27 | 254.40 | 253.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 253.75 | 254.40 | 253.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 254.54 | 254.43 | 253.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 253.95 | 254.43 | 253.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 253.87 | 254.36 | 253.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 253.56 | 254.36 | 253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 255.34 | 254.55 | 254.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 254.47 | 254.55 | 254.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 257.00 | 255.73 | 254.93 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 253.40 | 255.27 | 255.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 252.30 | 254.67 | 255.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 254.20 | 254.14 | 254.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 254.20 | 254.14 | 254.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 252.25 | 252.34 | 253.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 252.90 | 252.34 | 253.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 252.40 | 252.45 | 253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 252.95 | 252.45 | 253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 254.45 | 252.83 | 253.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 255.20 | 252.83 | 253.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 251.70 | 252.60 | 253.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:15:00 | 251.15 | 252.38 | 252.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 251.00 | 251.88 | 252.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 250.30 | 251.27 | 252.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 253.50 | 251.38 | 251.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 253.50 | 251.38 | 251.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 254.20 | 252.93 | 252.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 249.60 | 252.26 | 251.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 249.60 | 252.26 | 251.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 249.60 | 252.26 | 251.89 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 251.00 | 251.67 | 251.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 247.75 | 250.59 | 251.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 248.90 | 248.55 | 249.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 248.90 | 248.55 | 249.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 248.90 | 248.55 | 249.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 248.70 | 248.55 | 249.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 247.85 | 248.20 | 248.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:30:00 | 249.10 | 248.20 | 248.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 248.20 | 247.59 | 248.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 248.20 | 247.59 | 248.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 249.25 | 247.93 | 248.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 249.25 | 247.93 | 248.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 249.05 | 248.45 | 248.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 250.10 | 248.95 | 248.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 248.70 | 248.97 | 248.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 12:15:00 | 248.70 | 248.97 | 248.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 248.70 | 248.97 | 248.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 248.70 | 248.97 | 248.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 249.60 | 249.10 | 248.81 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 247.85 | 248.49 | 248.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 246.35 | 247.43 | 247.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 247.00 | 246.19 | 246.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 10:15:00 | 247.00 | 246.19 | 246.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 247.00 | 246.19 | 246.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:45:00 | 247.00 | 246.19 | 246.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 247.30 | 246.41 | 246.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 247.20 | 246.41 | 246.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 247.10 | 246.11 | 246.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 247.10 | 246.11 | 246.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 247.15 | 246.32 | 246.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 247.00 | 246.32 | 246.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 247.40 | 246.75 | 246.73 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 245.15 | 246.73 | 246.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 244.90 | 246.37 | 246.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 244.50 | 243.81 | 244.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 244.50 | 243.81 | 244.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 244.50 | 243.81 | 244.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 245.13 | 243.81 | 244.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 244.89 | 244.03 | 244.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 244.97 | 244.03 | 244.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 244.22 | 244.06 | 244.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 244.34 | 244.06 | 244.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 244.30 | 244.11 | 244.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:15:00 | 244.30 | 244.11 | 244.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 244.97 | 244.28 | 244.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 244.97 | 244.28 | 244.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 244.98 | 244.42 | 244.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:30:00 | 244.80 | 244.42 | 244.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 244.35 | 244.41 | 244.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 245.97 | 244.41 | 244.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 244.14 | 244.35 | 244.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 243.12 | 244.02 | 244.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 242.25 | 243.93 | 244.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:15:00 | 230.96 | 236.84 | 237.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:15:00 | 230.14 | 236.84 | 237.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 235.33 | 235.06 | 236.47 | SL hit (close>ema200) qty=0.50 sl=235.06 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 234.99 | 233.08 | 232.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 236.74 | 234.60 | 233.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 235.48 | 235.62 | 234.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 235.48 | 235.62 | 234.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 235.25 | 235.54 | 234.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 234.96 | 235.54 | 234.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 234.83 | 235.40 | 234.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 234.75 | 235.40 | 234.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 234.09 | 235.14 | 234.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 234.09 | 235.14 | 234.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 233.52 | 234.82 | 234.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 233.60 | 234.82 | 234.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 233.80 | 234.61 | 234.64 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 235.26 | 234.61 | 234.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 10:15:00 | 238.55 | 235.40 | 234.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 235.45 | 235.70 | 235.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:00:00 | 235.45 | 235.70 | 235.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 234.85 | 235.53 | 235.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 234.85 | 235.53 | 235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 234.69 | 235.36 | 235.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 234.85 | 235.36 | 235.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 234.99 | 235.23 | 235.08 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 233.94 | 234.80 | 234.90 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 236.70 | 235.02 | 234.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 238.59 | 236.09 | 235.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 238.65 | 238.70 | 237.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 238.90 | 238.70 | 237.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 237.72 | 238.28 | 237.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 237.88 | 238.28 | 237.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 236.83 | 239.84 | 239.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 236.83 | 239.84 | 239.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 238.20 | 239.51 | 238.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 235.97 | 239.51 | 238.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 237.62 | 238.61 | 238.65 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 241.05 | 238.93 | 238.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 241.88 | 240.33 | 239.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 240.18 | 240.70 | 240.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 240.18 | 240.70 | 240.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 239.93 | 240.55 | 240.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 239.24 | 240.55 | 240.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 238.60 | 240.16 | 239.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 238.40 | 240.16 | 239.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 239.14 | 239.96 | 239.83 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 236.20 | 239.08 | 239.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 234.48 | 238.16 | 239.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 235.79 | 234.14 | 236.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 235.79 | 234.14 | 236.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 235.79 | 234.14 | 236.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:30:00 | 231.96 | 233.83 | 235.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 240.05 | 235.79 | 235.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 240.05 | 235.79 | 235.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 11:15:00 | 241.58 | 237.67 | 236.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 246.15 | 247.05 | 243.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 246.15 | 247.05 | 243.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 246.15 | 247.05 | 243.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 247.48 | 247.05 | 243.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 247.00 | 246.97 | 244.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:45:00 | 247.44 | 246.34 | 244.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 243.10 | 244.36 | 244.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 243.10 | 244.36 | 244.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 241.24 | 243.29 | 243.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 242.00 | 241.67 | 242.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 242.00 | 241.67 | 242.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 242.00 | 241.67 | 242.64 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 245.51 | 242.84 | 242.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 247.02 | 245.32 | 244.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 270.44 | 271.62 | 265.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:15:00 | 270.00 | 271.62 | 265.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 265.50 | 269.57 | 267.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 265.50 | 269.57 | 267.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 260.55 | 267.77 | 267.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 263.70 | 267.77 | 267.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 259.75 | 266.16 | 266.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 254.55 | 263.84 | 265.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 255.50 | 253.82 | 257.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:45:00 | 255.80 | 253.82 | 257.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 256.80 | 254.91 | 256.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 257.10 | 254.91 | 256.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 256.75 | 255.28 | 256.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 265.85 | 255.28 | 256.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 268.00 | 257.82 | 257.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 268.70 | 263.83 | 261.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 265.80 | 267.39 | 265.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 265.80 | 267.39 | 265.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 265.80 | 267.39 | 265.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 265.15 | 267.39 | 265.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 267.90 | 267.49 | 265.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 265.35 | 267.49 | 265.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 266.00 | 267.43 | 266.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 266.00 | 267.43 | 266.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 267.60 | 267.46 | 266.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 268.80 | 267.72 | 266.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 269.70 | 267.13 | 266.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 268.60 | 269.37 | 268.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 268.65 | 269.17 | 268.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 269.00 | 269.14 | 268.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:15:00 | 271.55 | 269.14 | 268.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 266.90 | 273.41 | 272.28 | SL hit (close<static) qty=1.00 sl=268.20 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 269.20 | 271.16 | 271.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 267.15 | 269.82 | 270.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 271.15 | 269.71 | 270.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 271.15 | 269.71 | 270.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 271.15 | 269.71 | 270.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 271.15 | 269.71 | 270.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 270.90 | 269.95 | 270.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 271.50 | 269.95 | 270.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 270.55 | 270.07 | 270.53 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 271.50 | 270.87 | 270.81 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 270.00 | 270.69 | 270.73 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 272.10 | 270.77 | 270.70 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 264.20 | 269.63 | 270.20 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 272.70 | 268.69 | 268.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 13:15:00 | 275.35 | 270.02 | 269.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 09:15:00 | 273.80 | 275.85 | 273.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 273.80 | 275.85 | 273.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 273.80 | 275.85 | 273.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 273.65 | 275.85 | 273.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 273.80 | 275.44 | 273.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 273.15 | 275.44 | 273.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 272.70 | 274.89 | 273.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 272.70 | 274.89 | 273.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 272.65 | 274.44 | 273.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 273.40 | 274.44 | 273.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 273.80 | 274.31 | 273.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 274.45 | 274.31 | 273.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:45:00 | 274.80 | 274.61 | 273.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:30:00 | 274.10 | 274.81 | 274.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:30:00 | 274.90 | 275.05 | 274.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 276.70 | 275.88 | 274.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 280.05 | 277.86 | 276.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 279.20 | 278.90 | 277.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:00:00 | 280.50 | 281.28 | 280.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 279.00 | 280.82 | 280.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 277.15 | 280.09 | 280.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 15:00:00 | 277.15 | 280.09 | 280.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 275.70 | 279.21 | 279.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 275.70 | 279.21 | 279.67 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 284.80 | 280.33 | 280.13 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 276.75 | 280.19 | 280.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 09:15:00 | 273.80 | 278.91 | 279.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 11:15:00 | 277.80 | 277.78 | 279.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 12:00:00 | 277.80 | 277.78 | 279.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 279.60 | 278.14 | 279.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 279.75 | 278.14 | 279.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 279.45 | 278.40 | 279.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 279.85 | 278.40 | 279.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 278.85 | 278.49 | 279.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:30:00 | 277.00 | 278.48 | 279.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 263.15 | 265.86 | 267.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 261.80 | 261.59 | 264.03 | SL hit (close>ema200) qty=0.50 sl=261.59 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 265.00 | 264.13 | 264.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-19 10:15:00 | 269.50 | 265.44 | 264.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 12:15:00 | 268.60 | 268.73 | 267.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-20 12:30:00 | 269.10 | 268.73 | 267.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 266.35 | 268.25 | 267.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:45:00 | 266.20 | 268.25 | 267.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 265.45 | 267.69 | 267.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:45:00 | 264.95 | 267.69 | 267.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 266.80 | 267.27 | 266.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 266.80 | 267.27 | 266.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 267.05 | 267.23 | 266.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:15:00 | 265.80 | 267.23 | 266.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 266.15 | 267.01 | 266.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:30:00 | 266.90 | 267.01 | 266.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 266.05 | 266.82 | 266.83 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 270.20 | 267.16 | 266.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 276.90 | 271.29 | 269.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 286.85 | 287.01 | 283.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-01 13:30:00 | 286.40 | 287.01 | 283.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 285.40 | 286.98 | 284.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 284.65 | 286.98 | 284.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 285.40 | 286.67 | 284.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 285.00 | 286.67 | 284.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 285.55 | 286.24 | 284.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:30:00 | 285.20 | 286.24 | 284.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 285.40 | 286.42 | 285.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:15:00 | 284.70 | 286.42 | 285.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 281.15 | 285.36 | 284.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 281.15 | 285.36 | 284.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 281.85 | 284.66 | 284.59 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 283.10 | 284.35 | 284.46 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 285.55 | 284.25 | 284.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 286.35 | 284.67 | 284.38 | Break + close above crossover candle high |

### Cycle 70 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 277.85 | 283.83 | 284.12 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 14:15:00 | 285.55 | 284.23 | 284.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 288.20 | 285.28 | 284.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 286.00 | 287.82 | 286.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 286.00 | 287.82 | 286.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 286.00 | 287.82 | 286.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 286.00 | 287.82 | 286.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 286.85 | 287.63 | 286.94 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 284.45 | 286.16 | 286.38 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 287.80 | 286.41 | 286.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 288.60 | 287.12 | 286.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 287.10 | 287.42 | 286.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 287.10 | 287.42 | 286.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 287.30 | 287.39 | 287.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 287.05 | 287.39 | 287.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 287.50 | 287.41 | 287.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:15:00 | 286.60 | 287.41 | 287.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 285.95 | 287.12 | 286.95 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 283.90 | 286.48 | 286.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 283.05 | 285.79 | 286.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 11:15:00 | 284.45 | 283.86 | 284.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 11:45:00 | 284.15 | 283.86 | 284.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 284.75 | 284.14 | 284.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:30:00 | 285.25 | 284.14 | 284.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 283.95 | 284.10 | 284.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:45:00 | 285.35 | 284.10 | 284.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 284.00 | 284.08 | 284.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 284.25 | 284.08 | 284.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 283.10 | 283.89 | 284.51 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 285.50 | 283.93 | 283.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 293.95 | 287.25 | 286.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 300.65 | 301.36 | 296.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 300.65 | 301.36 | 296.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 297.45 | 300.09 | 298.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 297.45 | 300.09 | 298.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 300.10 | 300.10 | 298.35 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 294.15 | 296.88 | 297.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 290.20 | 293.75 | 295.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 10:15:00 | 284.50 | 284.24 | 287.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:30:00 | 285.10 | 284.24 | 287.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 243.60 | 2025-05-22 10:15:00 | 245.75 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-05-29 10:30:00 | 242.07 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-05-29 12:00:00 | 242.22 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-29 12:30:00 | 242.18 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-29 13:45:00 | 242.20 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-30 10:15:00 | 241.40 | 2025-06-06 11:15:00 | 238.73 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-06-27 10:45:00 | 243.16 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-27 14:00:00 | 243.58 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-27 14:30:00 | 242.92 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-30 11:00:00 | 243.56 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-01 10:15:00 | 242.50 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-07-01 14:30:00 | 242.72 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-01 15:15:00 | 243.19 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-02 11:30:00 | 242.00 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-22 12:15:00 | 245.93 | 2025-07-24 11:15:00 | 244.29 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-23 15:00:00 | 245.73 | 2025-07-24 11:15:00 | 244.29 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-24 10:45:00 | 245.80 | 2025-07-24 11:15:00 | 244.29 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-11 09:30:00 | 232.52 | 2025-08-12 09:15:00 | 234.36 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-11 12:15:00 | 232.80 | 2025-08-12 09:15:00 | 234.36 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-14 12:45:00 | 237.22 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-08-14 13:30:00 | 237.36 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-14 14:00:00 | 237.46 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-08-18 12:30:00 | 237.32 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-08-20 14:30:00 | 238.32 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-21 09:15:00 | 238.50 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-08-26 09:15:00 | 235.18 | 2025-09-01 11:15:00 | 236.35 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-12 14:15:00 | 233.82 | 2025-09-15 09:15:00 | 232.45 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-15 09:15:00 | 234.04 | 2025-09-15 09:15:00 | 232.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-18 15:00:00 | 235.62 | 2025-10-08 12:15:00 | 242.45 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2025-09-19 10:15:00 | 235.58 | 2025-10-08 12:15:00 | 242.45 | STOP_HIT | 1.00 | 2.92% |
| SELL | retest2 | 2025-10-09 13:15:00 | 243.10 | 2025-10-10 09:15:00 | 246.35 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-11-10 12:15:00 | 251.15 | 2025-11-12 11:15:00 | 253.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-10 14:45:00 | 251.00 | 2025-11-12 11:15:00 | 253.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-11 09:30:00 | 250.30 | 2025-11-12 11:15:00 | 253.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-02 14:15:00 | 243.12 | 2025-12-15 09:15:00 | 230.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 242.25 | 2025-12-15 09:15:00 | 230.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 14:15:00 | 243.12 | 2025-12-15 14:15:00 | 235.33 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-12-03 09:15:00 | 242.25 | 2025-12-15 14:15:00 | 235.33 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-01-12 09:30:00 | 231.96 | 2026-01-13 09:15:00 | 240.05 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-01-16 10:15:00 | 247.48 | 2026-01-19 14:15:00 | 243.10 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-01-16 11:15:00 | 247.00 | 2026-01-19 14:15:00 | 243.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-16 14:45:00 | 247.44 | 2026-01-19 14:15:00 | 243.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-09 13:00:00 | 268.80 | 2026-02-13 09:15:00 | 266.90 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-02-10 09:15:00 | 269.70 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-02-11 10:15:00 | 268.60 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-02-11 13:00:00 | 268.65 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-02-11 14:15:00 | 271.55 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-23 14:15:00 | 274.45 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2026-02-23 14:45:00 | 274.80 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2026-02-24 10:30:00 | 274.10 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2026-02-24 13:30:00 | 274.90 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-02-26 15:00:00 | 280.05 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-02-27 10:30:00 | 279.20 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-03-04 13:00:00 | 280.50 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-03-04 14:00:00 | 279.00 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-09 09:30:00 | 277.00 | 2026-03-16 09:15:00 | 263.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:30:00 | 277.00 | 2026-03-17 10:15:00 | 261.80 | STOP_HIT | 0.50 | 5.49% |
