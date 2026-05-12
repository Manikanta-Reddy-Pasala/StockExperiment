# AWL Agri Business Ltd. (AWL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 206.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 80 |
| ALERT1 | 49 |
| ALERT2 | 46 |
| ALERT2_SKIP | 26 |
| ALERT3 | 130 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 50 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 35
- **Target hits / Stop hits / Partials:** 1 / 50 / 9
- **Avg / median % per leg:** 0.68% / -0.41%
- **Sum % (uncompounded):** 40.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 1 | 5.3% | 1 | 18 | 0 | -0.73% | -13.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 1 | 5.3% | 1 | 18 | 0 | -0.73% | -13.8% |
| SELL (all) | 41 | 24 | 58.5% | 0 | 32 | 9 | 1.33% | 54.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.60% | -2.6% |
| SELL @ 3rd Alert (retest2) | 40 | 24 | 60.0% | 0 | 31 | 9 | 1.43% | 57.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.60% | -2.6% |
| retest2 (combined) | 59 | 25 | 42.4% | 1 | 49 | 9 | 0.74% | 43.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 263.20 | 259.80 | 259.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 263.80 | 261.77 | 260.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 262.50 | 262.68 | 261.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 262.25 | 262.68 | 261.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 268.50 | 270.62 | 269.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 268.50 | 270.62 | 269.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 269.30 | 270.36 | 269.41 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 267.40 | 268.82 | 268.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 264.60 | 267.63 | 268.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 259.60 | 258.74 | 260.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 259.60 | 258.74 | 260.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 259.60 | 258.74 | 260.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 259.60 | 258.74 | 260.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 260.50 | 259.09 | 260.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 259.55 | 259.09 | 260.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 260.70 | 259.41 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 260.70 | 259.41 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 259.40 | 259.41 | 260.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:45:00 | 258.65 | 259.86 | 260.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:15:00 | 258.50 | 259.73 | 260.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 15:00:00 | 258.60 | 259.50 | 260.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:30:00 | 258.80 | 259.08 | 259.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 272.90 | 261.87 | 260.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 272.90 | 261.87 | 260.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 278.45 | 270.65 | 267.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 267.90 | 270.10 | 267.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 15:15:00 | 267.90 | 270.10 | 267.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 267.90 | 270.10 | 267.58 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 267.80 | 268.44 | 268.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 267.00 | 268.16 | 268.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 266.70 | 264.81 | 265.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 266.70 | 264.81 | 265.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 266.70 | 264.81 | 265.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 267.50 | 264.81 | 265.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 267.45 | 265.34 | 265.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 267.85 | 265.34 | 265.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 267.60 | 266.42 | 266.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 268.00 | 266.94 | 266.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 273.25 | 274.99 | 272.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 14:00:00 | 273.25 | 274.99 | 272.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 272.60 | 274.94 | 273.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 272.60 | 274.94 | 273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 271.25 | 274.20 | 272.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 271.25 | 274.20 | 272.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 267.30 | 271.64 | 272.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 265.85 | 269.40 | 270.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 264.80 | 264.73 | 266.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 264.80 | 264.73 | 266.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 264.50 | 262.51 | 263.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 264.50 | 262.51 | 263.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 263.80 | 262.77 | 263.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 261.95 | 262.77 | 263.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 258.65 | 257.78 | 257.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 258.65 | 257.78 | 257.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 13:15:00 | 259.10 | 258.05 | 257.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 258.00 | 258.50 | 258.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 258.00 | 258.50 | 258.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 258.00 | 258.50 | 258.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 258.35 | 258.50 | 258.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 257.30 | 258.26 | 258.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 257.30 | 258.26 | 258.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 257.50 | 258.11 | 258.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 257.40 | 258.11 | 258.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 257.20 | 257.93 | 257.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 13:15:00 | 256.55 | 257.65 | 257.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 257.80 | 257.66 | 257.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 257.80 | 257.66 | 257.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 257.80 | 257.66 | 257.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 259.80 | 257.66 | 257.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 262.80 | 258.69 | 258.24 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 258.20 | 259.05 | 259.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 258.10 | 258.86 | 258.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 258.60 | 258.45 | 258.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 258.60 | 258.45 | 258.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 258.60 | 258.45 | 258.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 258.25 | 258.45 | 258.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 257.65 | 258.29 | 258.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 257.15 | 258.16 | 258.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 257.20 | 257.87 | 258.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 257.15 | 257.80 | 258.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 257.20 | 257.66 | 258.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 256.90 | 256.65 | 257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 256.90 | 256.65 | 257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 256.70 | 256.54 | 257.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 257.10 | 256.54 | 257.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 256.70 | 256.57 | 256.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 257.55 | 256.57 | 256.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 256.40 | 256.54 | 256.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 258.20 | 257.09 | 257.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 258.20 | 257.09 | 257.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 260.60 | 258.31 | 257.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 262.50 | 262.76 | 260.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:00:00 | 262.50 | 262.76 | 260.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 268.45 | 269.39 | 267.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 268.60 | 269.39 | 267.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 267.70 | 269.05 | 267.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 267.70 | 269.05 | 267.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 266.75 | 268.59 | 267.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 267.00 | 268.59 | 267.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 267.65 | 268.40 | 267.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 269.00 | 268.03 | 267.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:00:00 | 269.00 | 268.24 | 267.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:45:00 | 268.85 | 268.51 | 267.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 263.50 | 267.63 | 267.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 263.50 | 267.63 | 267.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 261.35 | 265.68 | 266.72 | Break + close below crossover candle low |

### Cycle 13 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 277.20 | 265.69 | 265.67 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 276.30 | 277.80 | 277.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 271.40 | 276.06 | 277.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 272.70 | 272.22 | 274.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 272.95 | 272.22 | 274.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 268.00 | 265.60 | 266.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 268.00 | 265.60 | 266.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 265.85 | 265.65 | 266.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 264.70 | 265.65 | 266.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 264.80 | 264.95 | 265.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 251.46 | 252.65 | 254.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 251.56 | 252.65 | 254.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 251.50 | 250.69 | 252.52 | SL hit (close>ema200) qty=0.50 sl=250.69 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 254.60 | 253.21 | 253.21 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 251.30 | 253.09 | 253.18 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 255.00 | 253.19 | 253.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 256.00 | 253.75 | 253.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 253.80 | 253.89 | 253.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 253.80 | 253.89 | 253.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 253.75 | 253.86 | 253.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:45:00 | 253.80 | 253.86 | 253.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 254.00 | 253.89 | 253.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 254.00 | 253.89 | 253.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 253.00 | 253.71 | 253.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 253.00 | 253.71 | 253.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 252.90 | 253.55 | 253.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 252.70 | 253.55 | 253.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 251.95 | 253.23 | 253.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 251.55 | 252.48 | 252.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 252.55 | 251.22 | 251.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 252.55 | 251.22 | 251.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 252.55 | 251.22 | 251.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 250.40 | 251.17 | 251.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 256.05 | 252.44 | 252.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 256.05 | 252.44 | 252.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 259.05 | 253.76 | 252.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 260.20 | 260.73 | 258.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 260.40 | 260.73 | 258.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 258.75 | 260.05 | 259.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 257.85 | 260.05 | 259.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 257.85 | 259.61 | 259.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 257.85 | 259.61 | 259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 259.15 | 259.40 | 259.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 258.85 | 259.40 | 259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 260.70 | 259.66 | 259.17 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 257.40 | 258.85 | 258.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 10:15:00 | 256.65 | 258.41 | 258.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 260.00 | 258.11 | 258.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 260.00 | 258.11 | 258.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 260.00 | 258.11 | 258.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 260.00 | 258.11 | 258.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 258.80 | 258.25 | 258.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 255.75 | 258.25 | 258.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 253.05 | 251.48 | 251.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 253.05 | 251.48 | 251.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 257.40 | 252.66 | 252.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 265.75 | 265.83 | 262.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 09:15:00 | 263.50 | 265.83 | 262.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 262.80 | 265.22 | 262.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 262.45 | 265.22 | 262.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 262.70 | 264.72 | 262.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 263.50 | 264.72 | 262.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 261.65 | 264.10 | 262.56 | SL hit (close<static) qty=1.00 sl=261.80 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 261.30 | 262.23 | 262.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 259.60 | 261.21 | 261.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 259.90 | 259.04 | 260.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:30:00 | 259.60 | 259.04 | 260.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 259.60 | 259.15 | 260.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 260.10 | 259.15 | 260.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 262.10 | 258.33 | 259.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 262.10 | 258.33 | 259.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 262.05 | 259.07 | 259.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 263.00 | 259.07 | 259.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 259.50 | 259.45 | 259.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 258.45 | 259.26 | 259.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 261.50 | 259.75 | 259.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 261.50 | 259.75 | 259.62 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 258.60 | 259.47 | 259.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 258.00 | 258.96 | 259.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 259.05 | 258.87 | 259.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 259.05 | 258.87 | 259.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 259.05 | 258.87 | 259.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 258.45 | 258.88 | 259.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 260.70 | 259.54 | 259.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 260.70 | 259.54 | 259.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 15:15:00 | 261.50 | 259.93 | 259.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 259.00 | 260.42 | 260.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 259.00 | 260.42 | 260.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 259.00 | 260.42 | 260.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 259.00 | 260.42 | 260.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 259.00 | 260.14 | 259.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 259.00 | 260.14 | 259.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 259.70 | 260.02 | 259.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 259.70 | 260.02 | 259.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 259.20 | 259.86 | 259.86 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 260.00 | 259.88 | 259.87 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 259.60 | 259.83 | 259.85 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 261.00 | 259.92 | 259.87 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 257.10 | 259.36 | 259.62 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 262.10 | 259.68 | 259.67 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 257.00 | 259.15 | 259.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 15:15:00 | 254.30 | 258.18 | 258.96 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 267.60 | 259.59 | 259.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 275.70 | 262.81 | 260.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 262.80 | 266.40 | 263.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 10:15:00 | 262.80 | 266.40 | 263.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 262.80 | 266.40 | 263.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 262.80 | 266.40 | 263.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 261.60 | 265.44 | 263.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 261.60 | 265.44 | 263.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 261.35 | 264.62 | 263.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:15:00 | 261.40 | 264.62 | 263.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 260.80 | 263.86 | 263.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 260.75 | 263.86 | 263.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 260.40 | 262.53 | 262.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 256.65 | 261.36 | 262.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 256.00 | 255.38 | 257.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:45:00 | 256.15 | 255.38 | 257.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 257.00 | 255.65 | 257.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 257.00 | 255.65 | 257.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 258.00 | 256.12 | 257.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 258.85 | 256.12 | 257.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 262.00 | 257.30 | 258.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 262.00 | 257.30 | 258.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 262.20 | 258.90 | 258.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 265.00 | 263.15 | 262.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 264.15 | 264.26 | 263.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 263.55 | 264.89 | 264.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 263.55 | 264.89 | 264.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 263.55 | 264.89 | 264.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 264.60 | 264.83 | 264.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 265.75 | 264.87 | 264.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 263.20 | 264.68 | 264.42 | SL hit (close<static) qty=1.00 sl=263.40 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 263.15 | 264.41 | 264.45 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 265.10 | 264.55 | 264.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 266.60 | 264.96 | 264.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 11:15:00 | 267.25 | 267.39 | 266.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 267.25 | 267.39 | 266.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 266.40 | 267.16 | 266.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 266.40 | 267.16 | 266.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 268.55 | 267.44 | 266.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:30:00 | 266.85 | 267.44 | 266.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 268.20 | 267.79 | 266.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 269.15 | 267.72 | 267.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:30:00 | 269.10 | 267.27 | 267.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 268.80 | 267.54 | 267.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 269.25 | 267.64 | 267.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 268.50 | 267.81 | 267.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 265.15 | 267.44 | 267.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 265.15 | 267.44 | 267.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 263.50 | 265.96 | 266.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 261.25 | 260.51 | 262.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 260.85 | 260.51 | 262.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 259.95 | 260.72 | 261.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 261.25 | 260.72 | 261.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 262.00 | 260.51 | 261.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 262.00 | 260.51 | 261.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 263.00 | 261.01 | 261.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 263.00 | 261.01 | 261.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 14:15:00 | 263.25 | 261.78 | 261.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 265.15 | 262.77 | 262.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 15:15:00 | 274.15 | 274.27 | 272.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:15:00 | 273.65 | 274.27 | 272.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 273.40 | 274.10 | 272.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 273.40 | 274.10 | 272.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 272.50 | 273.78 | 272.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 273.25 | 273.78 | 272.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 271.30 | 273.28 | 272.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 272.40 | 273.28 | 272.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 272.00 | 273.03 | 272.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:45:00 | 273.00 | 273.17 | 272.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 273.30 | 273.61 | 273.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 273.30 | 273.55 | 273.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 12:00:00 | 274.70 | 273.78 | 273.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 277.35 | 274.91 | 273.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:30:00 | 274.30 | 274.91 | 273.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 274.45 | 275.28 | 274.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 273.25 | 275.28 | 274.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 273.30 | 274.89 | 274.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 273.30 | 274.89 | 274.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 271.95 | 274.30 | 274.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 271.95 | 274.30 | 274.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 273.00 | 274.04 | 274.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 271.10 | 273.45 | 273.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 271.10 | 273.45 | 273.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 15:15:00 | 268.60 | 270.80 | 271.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 271.55 | 270.52 | 271.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 271.55 | 270.52 | 271.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 271.55 | 270.52 | 271.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 271.55 | 270.52 | 271.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 271.00 | 270.61 | 271.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 270.10 | 270.86 | 271.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 270.00 | 270.86 | 271.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:30:00 | 270.75 | 270.86 | 271.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 11:15:00 | 272.20 | 271.13 | 271.34 | SL hit (close>static) qty=1.00 sl=272.05 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 273.00 | 271.70 | 271.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 274.70 | 272.30 | 271.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 11:15:00 | 272.65 | 272.68 | 272.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 11:15:00 | 272.65 | 272.68 | 272.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 272.65 | 272.68 | 272.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 272.15 | 272.68 | 272.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 272.50 | 272.65 | 272.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:45:00 | 272.70 | 272.65 | 272.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 272.30 | 272.58 | 272.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 272.30 | 272.58 | 272.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 272.85 | 272.63 | 272.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 272.65 | 272.63 | 272.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 273.80 | 272.93 | 272.50 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 272.00 | 272.47 | 272.53 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 274.25 | 272.76 | 272.65 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 270.65 | 272.66 | 272.74 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 274.90 | 273.11 | 272.94 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 272.05 | 272.74 | 272.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 270.20 | 272.23 | 272.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 272.35 | 271.03 | 271.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 272.35 | 271.03 | 271.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 272.35 | 271.03 | 271.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 272.35 | 271.03 | 271.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 274.00 | 271.63 | 271.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 274.00 | 271.63 | 271.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 274.35 | 272.17 | 272.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 276.45 | 273.03 | 272.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 269.35 | 274.38 | 273.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 269.35 | 274.38 | 273.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 269.35 | 274.38 | 273.89 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 269.85 | 273.48 | 273.52 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 273.75 | 273.14 | 273.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 277.65 | 274.13 | 273.55 | Break + close above crossover candle high |

### Cycle 50 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 264.10 | 272.66 | 273.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 262.55 | 270.64 | 272.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 272.00 | 269.66 | 271.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 272.00 | 269.66 | 271.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 272.00 | 269.66 | 271.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 272.00 | 269.66 | 271.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 272.55 | 270.24 | 271.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 272.55 | 270.24 | 271.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 271.65 | 270.52 | 271.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 271.45 | 270.52 | 271.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 270.20 | 270.57 | 271.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 269.75 | 270.47 | 271.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 256.26 | 259.04 | 261.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 248.10 | 247.50 | 249.85 | SL hit (close>ema200) qty=0.50 sl=247.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 252.95 | 250.45 | 250.30 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 246.85 | 249.82 | 250.19 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 248.25 | 247.94 | 247.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 249.70 | 248.35 | 248.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 247.10 | 248.26 | 248.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 247.10 | 248.26 | 248.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 247.10 | 248.26 | 248.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 247.10 | 248.26 | 248.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 247.35 | 248.08 | 248.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 247.30 | 248.08 | 248.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 246.60 | 247.78 | 247.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 246.05 | 247.11 | 247.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 239.60 | 239.46 | 241.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 238.70 | 239.33 | 240.83 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 239.60 | 239.25 | 240.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 239.60 | 239.25 | 240.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 244.90 | 240.53 | 240.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 244.90 | 240.53 | 240.89 | SL hit (close>ema400) qty=1.00 sl=240.89 alert=retest1 |

### Cycle 55 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 246.30 | 241.69 | 241.39 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 240.00 | 241.26 | 241.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 239.50 | 240.38 | 240.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 15:15:00 | 237.10 | 237.08 | 238.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 236.30 | 237.08 | 238.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 237.70 | 236.47 | 237.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 237.70 | 236.47 | 237.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 237.90 | 236.76 | 237.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 236.15 | 236.76 | 237.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 236.60 | 236.77 | 237.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 237.70 | 236.77 | 237.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 237.45 | 236.91 | 237.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 237.45 | 236.91 | 237.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 237.90 | 237.11 | 237.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 237.70 | 237.11 | 237.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 237.60 | 237.20 | 237.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:30:00 | 237.90 | 237.20 | 237.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 240.00 | 237.76 | 237.56 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 237.20 | 238.26 | 238.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 15:15:00 | 234.50 | 237.51 | 238.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 232.30 | 232.27 | 233.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:45:00 | 233.23 | 232.27 | 233.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 220.90 | 220.65 | 223.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 219.93 | 220.63 | 222.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 220.00 | 220.51 | 222.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 220.25 | 220.46 | 222.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 219.69 | 220.30 | 222.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 208.93 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 209.00 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 209.24 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 208.71 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 212.90 | 212.11 | 214.66 | SL hit (close>ema200) qty=0.50 sl=212.11 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 209.46 | 208.53 | 208.47 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 207.56 | 208.35 | 208.43 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 209.17 | 208.58 | 208.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 213.85 | 209.72 | 209.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 212.58 | 213.07 | 211.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:30:00 | 213.08 | 213.07 | 211.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 210.99 | 212.65 | 211.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 211.51 | 212.65 | 211.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 210.91 | 212.30 | 211.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 210.49 | 212.30 | 211.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 208.63 | 211.57 | 211.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 208.63 | 211.57 | 211.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 209.00 | 211.06 | 211.12 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 213.07 | 211.31 | 211.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 215.16 | 213.49 | 212.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 213.68 | 214.08 | 213.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 213.68 | 214.08 | 213.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 213.68 | 214.08 | 213.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 11:15:00 | 215.43 | 213.89 | 213.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 210.10 | 214.39 | 214.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 210.10 | 214.39 | 214.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 207.66 | 209.33 | 210.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 203.88 | 203.07 | 204.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 203.88 | 203.07 | 204.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 204.93 | 203.61 | 204.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 204.99 | 203.61 | 204.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 204.83 | 203.86 | 204.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:15:00 | 204.35 | 203.86 | 204.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 204.07 | 203.94 | 204.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 194.13 | 196.04 | 197.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 193.87 | 196.04 | 197.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 193.98 | 193.09 | 195.10 | SL hit (close>ema200) qty=0.50 sl=193.09 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 180.55 | 178.11 | 177.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 196.28 | 181.81 | 179.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 179.30 | 182.26 | 180.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 179.30 | 182.26 | 180.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 179.30 | 182.26 | 180.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 179.30 | 182.26 | 180.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 178.60 | 181.52 | 180.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 177.60 | 181.52 | 180.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 176.60 | 180.54 | 179.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 176.84 | 180.54 | 179.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 178.20 | 180.07 | 179.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:15:00 | 179.70 | 180.07 | 179.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 12:15:00 | 178.09 | 179.45 | 179.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 178.09 | 179.45 | 179.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 176.00 | 178.53 | 179.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 176.45 | 173.95 | 175.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 176.45 | 173.95 | 175.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 176.45 | 173.95 | 175.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 176.00 | 173.95 | 175.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 176.40 | 174.44 | 175.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 175.75 | 174.44 | 175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 176.19 | 174.79 | 175.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 176.19 | 174.79 | 175.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 175.57 | 175.10 | 175.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 175.57 | 175.10 | 175.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 175.00 | 175.08 | 175.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 175.82 | 175.08 | 175.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 175.00 | 175.07 | 175.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 179.40 | 175.07 | 175.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 177.55 | 175.56 | 175.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 184.49 | 178.20 | 177.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 178.89 | 184.85 | 181.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 178.89 | 184.85 | 181.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 178.89 | 184.85 | 181.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 178.89 | 184.85 | 181.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 177.29 | 183.34 | 181.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 177.29 | 183.34 | 181.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 176.90 | 179.76 | 180.15 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 186.27 | 179.94 | 179.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 188.26 | 181.60 | 180.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 182.88 | 184.51 | 182.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 182.88 | 184.51 | 182.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 182.90 | 184.19 | 182.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 183.14 | 184.19 | 182.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 183.48 | 184.05 | 182.44 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 178.65 | 182.23 | 182.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 178.23 | 181.04 | 181.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 182.61 | 180.13 | 180.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 182.61 | 180.13 | 180.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 182.61 | 180.13 | 180.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 183.52 | 180.13 | 180.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 186.35 | 181.63 | 181.53 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 178.55 | 181.52 | 181.61 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 184.00 | 181.35 | 181.35 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 180.40 | 181.28 | 181.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 179.50 | 180.41 | 180.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 181.52 | 179.92 | 180.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 181.52 | 179.92 | 180.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 181.52 | 179.92 | 180.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:00:00 | 180.16 | 179.97 | 180.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 183.45 | 180.17 | 179.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 11:15:00 | 183.45 | 180.17 | 179.74 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 180.00 | 180.29 | 180.30 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 184.00 | 181.03 | 180.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 185.37 | 182.71 | 182.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 184.17 | 184.19 | 183.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 184.17 | 184.19 | 183.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 184.17 | 184.19 | 183.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 182.80 | 184.19 | 183.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 184.08 | 184.94 | 184.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 185.65 | 184.94 | 184.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-27 09:15:00 | 204.22 | 198.63 | 196.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 200.05 | 201.58 | 201.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 199.20 | 201.00 | 201.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 198.61 | 197.87 | 199.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 198.61 | 197.87 | 199.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 198.61 | 197.87 | 199.22 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 203.40 | 200.35 | 200.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 204.67 | 201.22 | 200.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 205.22 | 205.24 | 203.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:00:00 | 205.22 | 205.24 | 203.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 207.76 | 210.40 | 208.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 208.00 | 210.40 | 208.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 206.86 | 209.69 | 208.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 206.86 | 209.69 | 208.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 207.61 | 208.51 | 208.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 207.90 | 208.51 | 208.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 15:15:00 | 206.00 | 207.62 | 207.72 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 12:00:00 | 263.00 | 2025-05-12 12:15:00 | 263.20 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-05-26 12:45:00 | 258.65 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2025-05-26 14:15:00 | 258.50 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2025-05-26 15:00:00 | 258.60 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.53% |
| SELL | retest2 | 2025-05-27 09:30:00 | 258.80 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.45% |
| SELL | retest2 | 2025-06-19 09:15:00 | 261.95 | 2025-06-25 12:15:00 | 258.65 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2025-07-02 13:15:00 | 257.15 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-03 09:30:00 | 257.20 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-03 12:15:00 | 257.15 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-03 13:45:00 | 257.20 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-14 15:15:00 | 269.00 | 2025-07-15 14:15:00 | 263.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-07-15 12:00:00 | 269.00 | 2025-07-15 14:15:00 | 263.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-07-15 12:45:00 | 268.85 | 2025-07-15 14:15:00 | 263.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-07-31 09:15:00 | 264.70 | 2025-08-07 09:15:00 | 251.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 264.80 | 2025-08-07 09:15:00 | 251.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 264.70 | 2025-08-07 15:15:00 | 251.50 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-07-31 13:45:00 | 264.80 | 2025-08-07 15:15:00 | 251.50 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2025-08-18 12:00:00 | 250.40 | 2025-08-19 10:15:00 | 256.05 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-08-26 09:15:00 | 255.75 | 2025-09-02 11:15:00 | 253.05 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2025-09-05 11:15:00 | 263.50 | 2025-09-05 11:15:00 | 261.65 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-05 14:00:00 | 263.65 | 2025-09-08 11:15:00 | 261.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-11 15:00:00 | 258.45 | 2025-09-12 09:15:00 | 261.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-16 10:45:00 | 258.45 | 2025-09-16 14:15:00 | 260.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-07 15:15:00 | 265.75 | 2025-10-08 09:15:00 | 263.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-08 12:30:00 | 266.50 | 2025-10-09 09:15:00 | 263.05 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-08 14:45:00 | 266.00 | 2025-10-09 09:15:00 | 263.05 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-13 15:00:00 | 269.15 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-14 13:30:00 | 269.10 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-14 14:30:00 | 268.80 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-15 09:15:00 | 269.25 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-03 13:45:00 | 273.00 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-11-04 10:15:00 | 273.30 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-04 11:00:00 | 273.30 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-04 12:00:00 | 274.70 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-10 14:45:00 | 270.10 | 2025-11-11 11:15:00 | 272.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-10 15:15:00 | 270.00 | 2025-11-11 11:15:00 | 272.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-11 10:30:00 | 270.75 | 2025-11-11 11:15:00 | 272.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-26 13:15:00 | 269.75 | 2025-12-01 11:15:00 | 256.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 13:15:00 | 269.75 | 2025-12-04 13:15:00 | 248.10 | STOP_HIT | 0.50 | 8.03% |
| SELL | retest1 | 2025-12-19 10:45:00 | 238.70 | 2025-12-19 14:15:00 | 244.90 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-01-13 12:00:00 | 219.93 | 2026-01-19 09:15:00 | 208.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 220.00 | 2026-01-19 09:15:00 | 209.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 220.25 | 2026-01-19 09:15:00 | 209.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:00:00 | 219.69 | 2026-01-19 09:15:00 | 208.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 219.93 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-01-13 12:45:00 | 220.00 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2026-01-13 14:00:00 | 220.25 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-01-13 15:00:00 | 219.69 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2026-01-22 10:45:00 | 210.00 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-01-22 15:15:00 | 210.05 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2026-01-23 11:45:00 | 209.69 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-28 14:00:00 | 210.11 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-02-04 11:15:00 | 215.43 | 2026-02-06 09:15:00 | 210.10 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-02-17 11:15:00 | 204.35 | 2026-02-24 09:15:00 | 194.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 12:45:00 | 204.07 | 2026-02-24 09:15:00 | 193.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 11:15:00 | 204.35 | 2026-02-25 09:15:00 | 193.98 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-17 12:45:00 | 204.07 | 2026-02-25 09:15:00 | 193.98 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2026-03-12 11:15:00 | 179.70 | 2026-03-12 12:15:00 | 178.09 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-04-08 11:00:00 | 180.16 | 2026-04-10 11:15:00 | 183.45 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-04-21 09:15:00 | 185.65 | 2026-04-27 09:15:00 | 204.22 | TARGET_HIT | 1.00 | 10.00% |
