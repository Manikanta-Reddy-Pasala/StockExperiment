# Aegis Vopak Terminals Ltd. (AEGISVOPAK)

## Backtest Summary

- **Window:** 2025-06-02 09:15:00 → 2026-05-11 15:15:00 (1626 bars)
- **Last close:** 211.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 52 |
| ALERT1 | 38 |
| ALERT2 | 36 |
| ALERT2_SKIP | 18 |
| ALERT3 | 93 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 43 |
| PARTIAL | 14 |
| TARGET_HIT | 3 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 36 / 21
- **Target hits / Stop hits / Partials:** 3 / 40 / 14
- **Avg / median % per leg:** 1.70% / 0.90%
- **Sum % (uncompounded):** 96.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 8 | 42.1% | 2 | 17 | 0 | 0.49% | 9.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 8 | 44.4% | 2 | 16 | 0 | 0.52% | 9.3% |
| SELL (all) | 38 | 28 | 73.7% | 1 | 23 | 14 | 2.30% | 87.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 28 | 73.7% | 1 | 23 | 14 | 2.30% | 87.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 36 | 64.3% | 3 | 39 | 14 | 1.73% | 96.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 254.99 | 255.64 | 255.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 248.25 | 253.98 | 254.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 251.16 | 249.22 | 251.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 251.16 | 249.22 | 251.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 251.16 | 249.22 | 251.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 251.16 | 249.22 | 251.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 249.01 | 249.17 | 251.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 246.42 | 248.87 | 251.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 234.10 | 241.93 | 246.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 235.23 | 234.85 | 239.17 | SL hit (close>ema200) qty=0.50 sl=234.85 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 245.60 | 239.34 | 238.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 246.70 | 244.14 | 241.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 245.08 | 245.15 | 243.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 245.08 | 245.15 | 243.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 245.08 | 245.15 | 243.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:00:00 | 250.55 | 246.22 | 245.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 245.07 | 251.62 | 251.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 245.07 | 251.62 | 251.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 243.71 | 247.35 | 248.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 246.36 | 245.65 | 247.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 246.36 | 245.65 | 247.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 246.36 | 245.65 | 247.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 247.83 | 245.65 | 247.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 248.40 | 245.39 | 246.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 248.40 | 245.39 | 246.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 248.79 | 246.07 | 246.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 247.54 | 246.07 | 246.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 247.88 | 246.58 | 246.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:45:00 | 248.23 | 246.58 | 246.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 246.73 | 246.61 | 246.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:45:00 | 245.92 | 246.14 | 246.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 251.80 | 244.28 | 243.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 251.80 | 244.28 | 243.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 255.72 | 249.39 | 247.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 258.55 | 259.73 | 256.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:30:00 | 260.15 | 259.73 | 256.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 255.71 | 258.93 | 256.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 257.48 | 258.93 | 256.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 257.54 | 258.65 | 256.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 259.90 | 258.78 | 256.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 285.89 | 277.20 | 270.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 282.15 | 284.99 | 285.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 277.48 | 282.14 | 283.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 279.03 | 278.72 | 280.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:45:00 | 279.12 | 278.72 | 280.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 247.45 | 244.74 | 247.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 247.45 | 244.74 | 247.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 245.01 | 244.80 | 247.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 244.09 | 244.80 | 247.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 13:15:00 | 231.89 | 237.36 | 241.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 237.31 | 234.62 | 238.34 | SL hit (close>ema200) qty=0.50 sl=234.62 alert=retest2 |

### Cycle 6 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 251.81 | 240.50 | 240.29 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 243.00 | 243.89 | 243.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 239.21 | 242.71 | 243.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 241.00 | 240.24 | 241.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 15:15:00 | 241.00 | 240.24 | 241.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 241.00 | 240.24 | 241.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 238.22 | 240.24 | 241.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 246.91 | 239.91 | 240.90 | SL hit (close>static) qty=1.00 sl=242.60 alert=retest2 |

### Cycle 8 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 246.70 | 242.44 | 241.95 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 239.44 | 241.42 | 241.66 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 244.98 | 241.92 | 241.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 247.00 | 242.94 | 242.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 243.54 | 244.03 | 243.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 14:00:00 | 243.54 | 244.03 | 243.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 246.26 | 244.48 | 243.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:30:00 | 245.00 | 244.48 | 243.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 242.94 | 244.10 | 243.42 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 238.54 | 242.62 | 242.96 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 244.30 | 243.20 | 243.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 245.12 | 243.58 | 243.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 247.20 | 248.67 | 246.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 10:15:00 | 247.20 | 248.67 | 246.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 247.20 | 248.67 | 246.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 247.08 | 248.67 | 246.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 249.45 | 252.24 | 250.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 249.45 | 252.24 | 250.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 253.75 | 252.54 | 250.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 254.92 | 252.23 | 251.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:45:00 | 257.34 | 252.58 | 251.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 248.77 | 252.86 | 252.21 | SL hit (close<static) qty=1.00 sl=249.45 alert=retest2 |

### Cycle 13 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 247.35 | 251.76 | 251.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 244.92 | 250.39 | 251.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 254.90 | 250.40 | 250.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 254.90 | 250.40 | 250.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 254.90 | 250.40 | 250.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 254.90 | 250.40 | 250.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 254.00 | 251.12 | 251.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 248.45 | 251.12 | 251.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 14:15:00 | 236.03 | 239.68 | 241.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 243.60 | 240.11 | 241.48 | SL hit (close>ema200) qty=0.50 sl=240.11 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 248.13 | 243.41 | 242.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 251.00 | 246.82 | 245.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 248.71 | 249.53 | 247.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 248.71 | 249.53 | 247.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 248.71 | 249.53 | 247.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 248.71 | 249.53 | 247.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 246.26 | 248.99 | 248.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 246.26 | 248.99 | 248.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 246.50 | 248.50 | 248.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 245.77 | 248.50 | 248.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 248.50 | 249.01 | 248.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 248.50 | 249.01 | 248.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 247.76 | 248.76 | 248.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 248.00 | 248.76 | 248.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 246.19 | 248.25 | 248.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:45:00 | 246.04 | 248.25 | 248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 252.65 | 249.13 | 248.59 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 245.85 | 248.23 | 248.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 244.15 | 246.98 | 247.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 240.38 | 239.06 | 242.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 240.38 | 239.06 | 242.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 244.43 | 240.24 | 242.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 244.41 | 240.24 | 242.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 243.00 | 240.79 | 242.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 243.83 | 240.79 | 242.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 242.52 | 241.66 | 242.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 242.49 | 241.66 | 242.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 242.53 | 241.84 | 242.52 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 245.98 | 243.40 | 243.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 249.28 | 245.13 | 244.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 245.00 | 246.05 | 244.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 10:15:00 | 245.00 | 246.05 | 244.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 245.00 | 246.05 | 244.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 245.00 | 246.05 | 244.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 244.00 | 245.64 | 244.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 244.00 | 245.64 | 244.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 243.99 | 245.31 | 244.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 243.99 | 245.31 | 244.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 246.37 | 245.33 | 244.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 244.00 | 245.33 | 244.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 243.15 | 245.00 | 244.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 243.00 | 245.00 | 244.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 243.90 | 244.78 | 244.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:15:00 | 242.08 | 244.78 | 244.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 242.60 | 244.34 | 244.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 239.92 | 243.05 | 243.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 244.43 | 240.86 | 241.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 244.43 | 240.86 | 241.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 244.43 | 240.86 | 241.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 244.43 | 240.86 | 241.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 243.90 | 241.47 | 242.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 236.80 | 241.47 | 242.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:15:00 | 243.00 | 238.81 | 240.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 246.52 | 241.02 | 240.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 246.52 | 241.02 | 240.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 251.52 | 245.25 | 243.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 277.10 | 277.17 | 269.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 277.10 | 277.17 | 269.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 272.85 | 275.16 | 273.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:45:00 | 272.80 | 275.16 | 273.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 272.00 | 274.53 | 272.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 267.00 | 274.53 | 272.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 264.30 | 272.48 | 272.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 264.15 | 272.48 | 272.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 262.40 | 270.46 | 271.25 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 279.10 | 270.17 | 269.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 282.45 | 275.33 | 273.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 275.80 | 275.84 | 274.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 275.80 | 275.84 | 274.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 275.50 | 275.77 | 274.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:30:00 | 277.30 | 276.38 | 274.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:00:00 | 278.10 | 276.51 | 275.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:30:00 | 279.15 | 277.66 | 276.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 277.05 | 282.21 | 281.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 281.25 | 282.02 | 281.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 282.10 | 282.02 | 281.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 280.05 | 281.62 | 281.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 283.55 | 281.96 | 281.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 11:15:00 | 278.75 | 281.32 | 281.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 278.75 | 281.32 | 281.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 278.00 | 280.38 | 281.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 14:15:00 | 284.00 | 281.10 | 281.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 14:15:00 | 284.00 | 281.10 | 281.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 284.00 | 281.10 | 281.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 284.00 | 281.10 | 281.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 282.50 | 281.38 | 281.39 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 284.65 | 282.04 | 281.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 288.00 | 283.54 | 282.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 285.45 | 286.83 | 284.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:15:00 | 284.00 | 286.83 | 284.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 281.80 | 285.82 | 284.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 281.80 | 285.82 | 284.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 280.15 | 284.69 | 284.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 280.30 | 284.69 | 284.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 279.40 | 283.63 | 283.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 278.90 | 282.02 | 283.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 283.80 | 278.24 | 279.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 283.80 | 278.24 | 279.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 283.80 | 278.24 | 279.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 283.80 | 278.24 | 279.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 283.85 | 279.36 | 280.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 279.45 | 279.36 | 280.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 278.55 | 276.74 | 276.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 278.55 | 276.74 | 276.61 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 271.95 | 276.14 | 276.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 260.95 | 270.88 | 273.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 261.65 | 258.66 | 263.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 261.65 | 258.66 | 263.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 254.70 | 258.28 | 262.06 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 272.30 | 263.43 | 262.31 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 267.60 | 268.40 | 268.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 13:15:00 | 267.20 | 268.16 | 268.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 263.50 | 263.42 | 265.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 265.05 | 263.74 | 265.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 265.05 | 263.74 | 265.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 265.05 | 263.74 | 265.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 264.70 | 263.93 | 265.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 260.00 | 263.96 | 264.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 260.50 | 257.95 | 257.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 260.50 | 257.95 | 257.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 261.80 | 259.82 | 258.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 13:15:00 | 260.85 | 261.39 | 260.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:00:00 | 260.85 | 261.39 | 260.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 265.00 | 262.11 | 260.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 274.90 | 260.40 | 260.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 264.90 | 269.05 | 269.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 264.90 | 269.05 | 269.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 263.40 | 267.21 | 268.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 244.45 | 242.64 | 248.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:30:00 | 243.85 | 242.64 | 248.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 245.90 | 244.19 | 245.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 245.90 | 244.19 | 245.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 246.35 | 244.62 | 245.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:30:00 | 246.40 | 244.62 | 245.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 247.20 | 245.14 | 245.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 247.20 | 245.14 | 245.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 246.50 | 245.41 | 245.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 246.95 | 245.41 | 245.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 253.55 | 247.04 | 246.57 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 246.70 | 247.63 | 247.73 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 249.70 | 248.04 | 247.91 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 245.55 | 247.41 | 247.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 243.90 | 246.22 | 247.00 | Break + close below crossover candle low |

### Cycle 34 — BUY (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 14:15:00 | 254.70 | 247.91 | 247.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 14:15:00 | 258.10 | 253.03 | 250.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 254.65 | 254.70 | 252.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 14:30:00 | 255.55 | 255.64 | 253.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 257.05 | 258.00 | 256.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:45:00 | 256.60 | 258.00 | 256.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 257.10 | 257.82 | 256.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-22 14:15:00 | 255.55 | 257.36 | 256.74 | SL hit (close<ema400) qty=1.00 sl=256.74 alert=retest1 |

### Cycle 35 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 252.60 | 256.16 | 256.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 249.10 | 254.75 | 255.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 248.60 | 245.29 | 247.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 248.60 | 245.29 | 247.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 248.60 | 245.29 | 247.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 248.60 | 245.29 | 247.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 246.70 | 245.57 | 247.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 244.95 | 245.57 | 247.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 245.85 | 244.55 | 245.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 251.25 | 244.18 | 244.75 | SL hit (close>static) qty=1.00 sl=250.80 alert=retest2 |

### Cycle 36 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 251.05 | 245.56 | 245.32 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 246.80 | 248.97 | 249.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 246.56 | 248.49 | 248.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 241.05 | 239.18 | 241.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 241.05 | 239.18 | 241.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 240.53 | 239.45 | 241.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 241.76 | 239.45 | 241.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 239.78 | 238.19 | 239.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 239.78 | 238.19 | 239.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 236.74 | 237.90 | 239.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 235.65 | 237.90 | 239.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 223.87 | 226.36 | 228.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 212.09 | 218.72 | 222.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 38 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 219.92 | 206.67 | 205.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 229.00 | 216.76 | 214.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 222.61 | 226.11 | 222.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 222.61 | 226.11 | 222.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 222.61 | 226.11 | 222.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 222.61 | 226.11 | 222.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 219.32 | 224.75 | 221.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 219.98 | 224.75 | 221.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 222.50 | 224.30 | 221.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:45:00 | 223.45 | 223.94 | 222.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 224.00 | 225.26 | 223.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 14:45:00 | 224.21 | 223.82 | 223.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 220.63 | 222.71 | 222.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 220.63 | 222.71 | 222.88 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 227.00 | 223.46 | 223.14 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 222.19 | 223.82 | 223.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 220.95 | 223.25 | 223.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 219.78 | 218.81 | 220.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 219.78 | 218.81 | 220.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 219.78 | 218.81 | 220.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 218.79 | 219.75 | 220.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 11:00:00 | 218.90 | 219.58 | 220.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 218.95 | 219.19 | 219.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 218.90 | 219.19 | 219.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 218.90 | 219.13 | 219.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 216.75 | 219.13 | 219.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 207.85 | 213.39 | 215.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 207.95 | 213.39 | 215.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 208.00 | 213.39 | 215.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 207.95 | 213.39 | 215.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 12:15:00 | 205.91 | 210.74 | 214.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 216.99 | 211.23 | 213.61 | SL hit (close>ema200) qty=0.50 sl=211.23 alert=retest2 |

### Cycle 42 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 218.96 | 211.65 | 211.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 221.00 | 213.52 | 212.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 219.60 | 220.20 | 218.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 219.60 | 220.20 | 218.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 215.11 | 219.18 | 217.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 214.90 | 219.18 | 217.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 214.87 | 218.32 | 217.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 214.87 | 218.32 | 217.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 214.85 | 217.10 | 217.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 213.86 | 215.57 | 216.39 | Break + close below crossover candle low |

### Cycle 44 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 230.08 | 216.58 | 216.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 232.03 | 227.57 | 224.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 232.70 | 233.44 | 230.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 232.70 | 233.44 | 230.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 231.55 | 232.94 | 230.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:30:00 | 231.39 | 232.94 | 230.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 230.00 | 232.82 | 231.20 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 13:15:00 | 227.94 | 230.08 | 230.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 222.52 | 228.29 | 229.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 10:15:00 | 193.38 | 191.49 | 199.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 11:00:00 | 193.38 | 191.49 | 199.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 197.74 | 193.46 | 198.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 198.96 | 193.46 | 198.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 198.14 | 194.39 | 198.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 198.14 | 194.39 | 198.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 199.85 | 195.49 | 198.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 199.85 | 195.49 | 198.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 200.00 | 196.39 | 198.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 196.43 | 196.39 | 198.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 12:15:00 | 198.00 | 196.89 | 198.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 191.40 | 195.66 | 197.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 186.61 | 191.92 | 193.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 188.10 | 191.92 | 193.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 190.30 | 188.46 | 190.67 | SL hit (close>ema200) qty=0.50 sl=188.46 alert=retest2 |

### Cycle 46 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 192.85 | 191.29 | 191.25 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 11:15:00 | 188.86 | 190.99 | 191.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 183.79 | 188.97 | 190.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 09:15:00 | 180.41 | 179.50 | 181.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 09:45:00 | 180.22 | 179.50 | 181.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 171.81 | 173.70 | 177.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 171.00 | 173.70 | 177.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:30:00 | 169.56 | 173.97 | 175.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 170.65 | 170.66 | 172.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:15:00 | 162.45 | 167.43 | 170.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:15:00 | 162.12 | 167.43 | 170.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 15:15:00 | 161.08 | 164.67 | 168.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 166.64 | 165.06 | 167.95 | SL hit (close>ema200) qty=0.50 sl=165.06 alert=retest2 |

### Cycle 48 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 174.80 | 167.28 | 166.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 186.89 | 174.49 | 171.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 188.08 | 188.79 | 183.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 188.08 | 188.79 | 183.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 187.12 | 190.01 | 187.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 187.78 | 190.01 | 187.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 187.49 | 188.32 | 187.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 188.03 | 188.26 | 187.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 194.02 | 197.45 | 197.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 194.02 | 197.45 | 197.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 193.00 | 194.25 | 195.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 188.42 | 188.35 | 191.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 191.53 | 188.35 | 191.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 192.46 | 189.18 | 191.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 192.46 | 189.18 | 191.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 194.67 | 190.27 | 191.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 194.60 | 190.27 | 191.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 197.94 | 192.80 | 192.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 198.60 | 195.48 | 193.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 09:15:00 | 195.71 | 197.75 | 196.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 195.71 | 197.75 | 196.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 195.71 | 197.75 | 196.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 194.15 | 197.75 | 196.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 195.72 | 197.34 | 196.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:15:00 | 195.55 | 197.34 | 196.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 197.56 | 197.39 | 196.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:45:00 | 195.49 | 197.39 | 196.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 196.00 | 197.11 | 196.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 196.00 | 197.11 | 196.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 195.31 | 196.75 | 196.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 195.31 | 196.75 | 196.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 194.20 | 195.85 | 195.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 188.39 | 194.36 | 195.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 192.03 | 190.94 | 192.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 192.03 | 190.94 | 192.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 192.03 | 190.94 | 192.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 192.03 | 190.94 | 192.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 193.57 | 191.47 | 192.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 193.00 | 191.47 | 192.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 191.91 | 191.56 | 192.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 190.48 | 191.41 | 192.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 193.88 | 191.96 | 192.50 | SL hit (close>static) qty=1.00 sl=193.67 alert=retest2 |

### Cycle 52 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 193.24 | 192.83 | 192.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 196.71 | 193.61 | 193.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 194.00 | 194.43 | 193.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 194.00 | 194.43 | 193.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 194.00 | 194.43 | 193.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:30:00 | 194.03 | 194.43 | 193.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 195.50 | 194.65 | 193.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 203.51 | 194.65 | 193.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 197.42 | 197.40 | 195.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-07 10:15:00 | 217.16 | 203.50 | 199.40 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-12 09:30:00 | 246.42 | 2025-06-13 09:15:00 | 234.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 09:30:00 | 246.42 | 2025-06-16 11:15:00 | 235.23 | STOP_HIT | 0.50 | 4.54% |
| BUY | retest2 | 2025-06-24 15:00:00 | 250.55 | 2025-06-30 10:15:00 | 245.07 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-07-03 14:45:00 | 245.92 | 2025-07-07 13:15:00 | 251.80 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-07-11 12:30:00 | 259.90 | 2025-07-17 09:15:00 | 285.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 244.09 | 2025-08-01 13:15:00 | 231.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 244.09 | 2025-08-04 10:15:00 | 237.31 | STOP_HIT | 0.50 | 2.78% |
| SELL | retest2 | 2025-08-08 09:15:00 | 238.22 | 2025-08-08 12:15:00 | 246.91 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2025-08-22 15:00:00 | 254.92 | 2025-08-26 10:15:00 | 248.77 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-08-25 14:45:00 | 257.34 | 2025-08-26 10:15:00 | 248.77 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-08-28 09:15:00 | 248.45 | 2025-09-05 14:15:00 | 236.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 09:15:00 | 248.45 | 2025-09-08 09:15:00 | 243.60 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2025-09-26 09:15:00 | 236.80 | 2025-09-29 09:15:00 | 246.52 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-09-26 15:15:00 | 243.00 | 2025-09-29 09:15:00 | 246.52 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-10-14 13:30:00 | 277.30 | 2025-10-20 11:15:00 | 278.75 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-10-15 11:00:00 | 278.10 | 2025-10-20 11:15:00 | 278.75 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-10-15 13:30:00 | 279.15 | 2025-10-20 11:15:00 | 278.75 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-10-17 14:30:00 | 277.05 | 2025-10-20 11:15:00 | 278.75 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-10-20 10:45:00 | 283.55 | 2025-10-20 11:15:00 | 278.75 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-10-28 09:15:00 | 279.45 | 2025-10-31 13:15:00 | 278.55 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-19 15:15:00 | 260.00 | 2025-11-26 12:15:00 | 260.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-12-01 09:15:00 | 274.90 | 2025-12-04 13:15:00 | 264.90 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest1 | 2025-12-18 14:30:00 | 255.55 | 2025-12-22 14:15:00 | 255.55 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-22 14:30:00 | 258.30 | 2025-12-22 15:15:00 | 255.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-29 09:15:00 | 244.95 | 2025-12-30 14:15:00 | 251.25 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-12-29 15:15:00 | 245.85 | 2025-12-30 14:15:00 | 251.25 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-01-09 11:15:00 | 235.65 | 2026-01-16 09:15:00 | 223.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:15:00 | 235.65 | 2026-01-19 09:15:00 | 212.09 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-02 13:45:00 | 223.45 | 2026-02-04 10:15:00 | 220.63 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-03 10:45:00 | 224.00 | 2026-02-04 10:15:00 | 220.63 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-03 14:45:00 | 224.21 | 2026-02-04 10:15:00 | 220.63 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-02-11 09:30:00 | 218.79 | 2026-02-13 09:15:00 | 207.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 11:00:00 | 218.90 | 2026-02-13 09:15:00 | 207.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 14:30:00 | 218.95 | 2026-02-13 09:15:00 | 208.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 15:15:00 | 218.90 | 2026-02-13 09:15:00 | 207.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 216.75 | 2026-02-13 12:15:00 | 205.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 218.79 | 2026-02-13 14:15:00 | 216.99 | STOP_HIT | 0.50 | 0.82% |
| SELL | retest2 | 2026-02-11 11:00:00 | 218.90 | 2026-02-13 14:15:00 | 216.99 | STOP_HIT | 0.50 | 0.87% |
| SELL | retest2 | 2026-02-11 14:30:00 | 218.95 | 2026-02-13 14:15:00 | 216.99 | STOP_HIT | 0.50 | 0.90% |
| SELL | retest2 | 2026-02-11 15:15:00 | 218.90 | 2026-02-13 14:15:00 | 216.99 | STOP_HIT | 0.50 | 0.87% |
| SELL | retest2 | 2026-02-12 09:15:00 | 216.75 | 2026-02-13 14:15:00 | 216.99 | STOP_HIT | 0.50 | -0.11% |
| SELL | retest2 | 2026-03-09 09:15:00 | 196.43 | 2026-03-16 09:15:00 | 186.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 12:15:00 | 198.00 | 2026-03-16 09:15:00 | 188.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 196.43 | 2026-03-16 14:15:00 | 190.30 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2026-03-09 12:15:00 | 198.00 | 2026-03-16 14:15:00 | 190.30 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2026-03-10 09:15:00 | 191.40 | 2026-03-17 13:15:00 | 192.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-24 10:15:00 | 171.00 | 2026-03-30 11:15:00 | 162.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 169.56 | 2026-03-30 11:15:00 | 162.12 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2026-03-30 09:15:00 | 170.65 | 2026-03-30 15:15:00 | 161.08 | PARTIAL | 0.50 | 5.61% |
| SELL | retest2 | 2026-03-24 10:15:00 | 171.00 | 2026-04-01 09:15:00 | 166.64 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2026-03-27 09:30:00 | 169.56 | 2026-04-01 09:15:00 | 166.64 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2026-03-30 09:15:00 | 170.65 | 2026-04-01 09:15:00 | 166.64 | STOP_HIT | 0.50 | 2.35% |
| BUY | retest2 | 2026-04-13 10:15:00 | 187.78 | 2026-04-22 09:15:00 | 194.02 | STOP_HIT | 1.00 | 3.32% |
| BUY | retest2 | 2026-04-13 13:30:00 | 187.49 | 2026-04-22 09:15:00 | 194.02 | STOP_HIT | 1.00 | 3.48% |
| BUY | retest2 | 2026-04-13 15:00:00 | 188.03 | 2026-04-22 09:15:00 | 194.02 | STOP_HIT | 1.00 | 3.19% |
| SELL | retest2 | 2026-05-04 12:30:00 | 190.48 | 2026-05-04 14:15:00 | 193.88 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-05-06 09:15:00 | 203.51 | 2026-05-07 10:15:00 | 217.16 | TARGET_HIT | 1.00 | 6.71% |
