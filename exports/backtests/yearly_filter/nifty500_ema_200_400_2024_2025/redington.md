# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 223.29
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 40 |
| PARTIAL | 10 |
| TARGET_HIT | 18 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 24
- **Target hits / Stop hits / Partials:** 18 / 24 / 10
- **Avg / median % per leg:** 2.97% / 5.00%
- **Sum % (uncompounded):** 154.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 8 | 13 | 0 | 1.59% | 33.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.05% | -6.1% |
| BUY @ 3rd Alert (retest2) | 19 | 8 | 42.1% | 8 | 11 | 0 | 2.08% | 39.6% |
| SELL (all) | 31 | 20 | 64.5% | 10 | 11 | 10 | 3.91% | 121.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 20 | 64.5% | 10 | 11 | 10 | 3.91% | 121.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.05% | -6.1% |
| retest2 (combined) | 50 | 28 | 56.0% | 18 | 22 | 10 | 3.21% | 160.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 202.53 | 210.62 | 210.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 200.48 | 210.30 | 210.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 13:15:00 | 210.01 | 209.95 | 210.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 13:15:00 | 210.01 | 209.95 | 210.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 210.01 | 209.95 | 210.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 13:45:00 | 210.23 | 209.95 | 210.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 210.50 | 209.96 | 210.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:15:00 | 210.55 | 209.96 | 210.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 210.55 | 209.96 | 210.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 209.00 | 209.96 | 210.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 11:15:00 | 209.26 | 209.95 | 210.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 213.20 | 209.98 | 210.29 | SL hit (close>static) qty=1.00 sl=211.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 11:15:00 | 216.86 | 210.61 | 210.60 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 203.35 | 210.56 | 210.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 197.00 | 209.85 | 210.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 203.23 | 203.22 | 206.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:30:00 | 203.33 | 203.22 | 206.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 205.09 | 203.24 | 206.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 12:15:00 | 203.85 | 204.36 | 206.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 13:00:00 | 203.99 | 204.36 | 206.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 203.92 | 204.30 | 206.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:30:00 | 204.04 | 204.30 | 206.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 205.64 | 204.26 | 206.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:30:00 | 206.30 | 204.26 | 206.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 207.11 | 204.30 | 206.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-28 13:15:00 | 207.11 | 204.30 | 206.11 | SL hit (close>static) qty=1.00 sl=206.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 197.30 | 190.24 | 190.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 202.17 | 190.50 | 190.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 202.70 | 203.06 | 198.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 15:00:00 | 202.70 | 203.06 | 198.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 198.53 | 202.97 | 198.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 198.53 | 202.97 | 198.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 198.79 | 202.93 | 198.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:45:00 | 198.85 | 202.93 | 198.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 194.94 | 202.85 | 198.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 194.94 | 202.85 | 198.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 188.28 | 202.71 | 198.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 188.28 | 202.71 | 198.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 200.49 | 202.31 | 198.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 206.27 | 202.31 | 198.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 196.89 | 201.86 | 198.76 | SL hit (close<static) qty=1.00 sl=197.33 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 14:15:00 | 242.90 | 281.46 | 281.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 237.70 | 280.64 | 281.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 266.00 | 248.62 | 258.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 266.00 | 248.62 | 258.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 266.00 | 248.62 | 258.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 266.00 | 248.62 | 258.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 275.70 | 248.89 | 258.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 275.70 | 248.89 | 258.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 262.76 | 264.16 | 264.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 263.40 | 264.16 | 264.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 265.14 | 264.14 | 264.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 265.14 | 264.14 | 264.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 265.50 | 264.15 | 264.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 269.60 | 264.15 | 264.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 270.51 | 264.21 | 264.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 264.67 | 264.24 | 264.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 293.01 | 264.69 | 264.98 | SL hit (close>static) qty=1.00 sl=274.44 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 287.84 | 265.40 | 265.33 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 253.10 | 267.49 | 267.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 13:15:00 | 252.40 | 266.70 | 267.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 279.90 | 265.51 | 266.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 282.30 | 265.67 | 266.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 282.30 | 265.67 | 266.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 289.65 | 267.60 | 267.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 298.80 | 271.34 | 269.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 281.20 | 282.61 | 277.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 285.70 | 282.45 | 277.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:00:00 | 285.85 | 282.49 | 277.20 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 277.80 | 282.40 | 277.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 277.40 | 282.40 | 277.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 277.05 | 282.34 | 277.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 277.05 | 282.34 | 277.51 | SL hit (close<ema400) qty=1.00 sl=277.51 alert=retest1 |

### Cycle 9 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 272.00 | 275.38 | 275.39 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 280.10 | 275.42 | 275.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 289.60 | 275.65 | 275.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 275.25 | 277.58 | 276.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 276.25 | 277.56 | 276.55 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 267.45 | 275.66 | 275.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 263.50 | 275.08 | 275.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 269.40 | 268.36 | 271.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 269.40 | 268.36 | 271.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 270.40 | 268.38 | 271.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 270.40 | 268.38 | 271.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 271.00 | 268.40 | 271.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 268.20 | 268.40 | 271.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 269.20 | 268.42 | 271.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 274.20 | 268.23 | 271.19 | SL hit (close>static) qty=1.00 sl=273.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 11:15:00 | 201.40 | 2024-06-10 12:15:00 | 221.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-03 10:00:00 | 201.45 | 2024-06-10 12:15:00 | 221.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-03 11:00:00 | 201.30 | 2024-06-10 12:15:00 | 221.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 10:45:00 | 210.05 | 2024-07-19 09:15:00 | 200.10 | STOP_HIT | 1.00 | -4.74% |
| BUY | retest2 | 2024-06-07 12:15:00 | 210.15 | 2024-07-19 09:15:00 | 200.10 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2024-06-10 09:15:00 | 214.40 | 2024-07-19 09:15:00 | 200.10 | STOP_HIT | 1.00 | -6.67% |
| BUY | retest2 | 2024-07-09 09:30:00 | 210.00 | 2024-07-19 09:15:00 | 200.10 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2024-07-25 09:15:00 | 209.00 | 2024-07-26 09:15:00 | 213.20 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-07-25 11:15:00 | 209.26 | 2024-07-26 09:15:00 | 213.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-08-26 12:15:00 | 203.85 | 2024-08-28 13:15:00 | 207.11 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-08-26 13:00:00 | 203.99 | 2024-08-28 13:15:00 | 207.11 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-27 09:45:00 | 203.92 | 2024-08-28 13:15:00 | 207.11 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-08-27 11:30:00 | 204.04 | 2024-08-28 13:15:00 | 207.11 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-08-28 15:15:00 | 204.30 | 2024-09-09 09:15:00 | 194.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 09:30:00 | 203.28 | 2024-09-09 09:15:00 | 193.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 12:00:00 | 203.52 | 2024-09-09 09:15:00 | 193.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-28 15:15:00 | 204.30 | 2024-09-25 11:15:00 | 183.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-30 09:30:00 | 203.28 | 2024-09-25 11:15:00 | 182.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-30 12:00:00 | 203.52 | 2024-09-25 11:15:00 | 183.17 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-31 14:15:00 | 206.27 | 2025-01-06 10:15:00 | 196.89 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-01-07 15:15:00 | 205.72 | 2025-01-21 09:15:00 | 226.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-09 09:30:00 | 205.01 | 2025-01-21 09:15:00 | 225.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-10 11:00:00 | 205.44 | 2025-01-21 09:15:00 | 225.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-29 09:30:00 | 208.37 | 2025-01-29 12:15:00 | 203.90 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-01-29 10:45:00 | 208.06 | 2025-01-29 12:15:00 | 203.90 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-01-29 11:30:00 | 208.34 | 2025-01-29 12:15:00 | 203.90 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-01-30 09:45:00 | 209.70 | 2025-02-01 12:15:00 | 202.81 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-02-05 14:00:00 | 210.40 | 2025-02-06 15:15:00 | 231.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-09 12:30:00 | 213.21 | 2025-04-09 15:15:00 | 204.20 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-04-11 09:15:00 | 216.05 | 2025-04-23 12:15:00 | 237.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-29 11:30:00 | 264.67 | 2025-09-29 14:15:00 | 293.01 | STOP_HIT | 1.00 | -10.71% |
| BUY | retest1 | 2025-12-01 09:15:00 | 285.70 | 2025-12-03 10:15:00 | 277.05 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2025-12-01 10:00:00 | 285.85 | 2025-12-03 10:15:00 | 277.05 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-12-24 10:30:00 | 277.20 | 2025-12-24 12:15:00 | 273.85 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-01 09:15:00 | 268.20 | 2026-02-03 09:15:00 | 274.20 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-02-01 11:30:00 | 269.20 | 2026-02-03 09:15:00 | 274.20 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-02-05 10:00:00 | 268.25 | 2026-02-10 09:15:00 | 274.40 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-09 12:00:00 | 270.00 | 2026-02-10 09:15:00 | 274.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-10 13:45:00 | 273.00 | 2026-02-16 09:15:00 | 259.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 14:30:00 | 271.80 | 2026-02-16 09:15:00 | 258.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 13:45:00 | 273.00 | 2026-02-23 13:15:00 | 245.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-10 14:30:00 | 271.80 | 2026-02-23 14:15:00 | 244.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 272.25 | 2026-03-04 09:15:00 | 258.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:00:00 | 268.80 | 2026-03-04 09:15:00 | 255.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 12:30:00 | 262.75 | 2026-03-04 11:15:00 | 249.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 14:15:00 | 263.40 | 2026-03-04 11:15:00 | 250.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 272.25 | 2026-03-06 15:15:00 | 245.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 10:00:00 | 268.80 | 2026-03-09 09:15:00 | 241.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 12:30:00 | 262.75 | 2026-03-09 09:15:00 | 236.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 14:15:00 | 263.40 | 2026-03-09 09:15:00 | 237.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-10 13:00:00 | 262.20 | 2026-03-11 09:15:00 | 249.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 13:00:00 | 262.20 | 2026-03-13 13:15:00 | 235.98 | TARGET_HIT | 0.50 | 10.00% |
