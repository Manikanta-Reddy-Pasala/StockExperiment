# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 223.29
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 20 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 15
- **Target hits / Stop hits / Partials:** 7 / 15 / 7
- **Avg / median % per leg:** 2.54% / -1.21%
- **Sum % (uncompounded):** 73.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.12% | -23.3% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.05% | -6.1% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.91% | -17.2% |
| SELL (all) | 18 | 14 | 77.8% | 7 | 4 | 7 | 5.39% | 97.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 14 | 77.8% | 7 | 4 | 7 | 5.39% | 97.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.05% | -6.1% |
| retest2 (combined) | 27 | 14 | 51.9% | 7 | 13 | 7 | 2.96% | 79.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 242.50 | 279.12 | 279.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 238.35 | 275.59 | 277.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 266.00 | 248.61 | 257.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 266.00 | 248.61 | 257.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 266.00 | 248.61 | 257.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 266.00 | 248.61 | 257.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 275.70 | 248.88 | 257.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 275.70 | 248.88 | 257.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 262.76 | 264.16 | 264.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 263.40 | 264.16 | 264.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 263.73 | 264.12 | 264.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 263.73 | 264.12 | 264.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 265.14 | 264.13 | 264.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 265.14 | 264.13 | 264.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 265.50 | 264.15 | 264.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 269.60 | 264.15 | 264.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 270.91 | 264.31 | 264.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 270.91 | 264.31 | 264.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 273.98 | 264.40 | 264.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 273.98 | 264.40 | 264.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 293.01 | 264.69 | 264.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 15:15:00 | 295.00 | 264.99 | 264.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 269.05 | 269.54 | 267.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 09:30:00 | 268.75 | 269.54 | 267.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 267.40 | 269.51 | 267.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 267.30 | 269.51 | 267.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 267.55 | 269.49 | 267.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 267.05 | 269.49 | 267.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 267.80 | 269.48 | 267.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 267.10 | 269.48 | 267.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 267.70 | 269.46 | 267.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 267.70 | 269.46 | 267.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 268.60 | 269.45 | 267.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 276.70 | 269.44 | 267.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:45:00 | 269.70 | 271.35 | 268.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 269.85 | 271.71 | 269.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:45:00 | 269.25 | 271.63 | 269.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 268.15 | 271.59 | 269.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 268.15 | 271.59 | 269.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 267.80 | 271.56 | 269.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 266.65 | 271.56 | 269.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 265.20 | 271.45 | 268.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 265.20 | 271.45 | 268.97 | SL hit (close<static) qty=1.00 sl=267.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 256.30 | 267.24 | 267.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 253.45 | 266.99 | 267.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 277.45 | 265.51 | 266.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 279.90 | 265.51 | 266.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 282.30 | 265.67 | 266.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 282.30 | 265.67 | 266.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 296.25 | 267.38 | 267.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 298.80 | 271.34 | 269.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 281.20 | 282.61 | 276.98 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:15:00 | 285.70 | 282.45 | 277.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:00:00 | 285.85 | 282.49 | 277.11 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 277.80 | 282.40 | 277.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 277.40 | 282.40 | 277.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 277.05 | 282.34 | 277.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 277.05 | 282.34 | 277.43 | SL hit (close<ema400) qty=1.00 sl=277.43 alert=retest1 |

### Cycle 5 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 272.45 | 275.31 | 275.32 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 279.05 | 275.33 | 275.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 279.90 | 275.38 | 275.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 275.25 | 277.58 | 276.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 275.25 | 277.58 | 276.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 276.25 | 277.56 | 276.52 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 267.45 | 275.66 | 275.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 263.50 | 275.08 | 275.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 269.40 | 268.36 | 271.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 14:00:00 | 269.40 | 268.36 | 271.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 270.40 | 268.38 | 271.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 270.40 | 268.38 | 271.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 271.00 | 268.40 | 271.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 268.20 | 268.40 | 271.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 269.20 | 268.42 | 271.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 274.20 | 268.23 | 271.17 | SL hit (close>static) qty=1.00 sl=273.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-10 09:15:00 | 276.70 | 2025-10-20 10:15:00 | 265.20 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2025-10-14 12:45:00 | 269.70 | 2025-10-20 10:15:00 | 265.20 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-10-17 11:00:00 | 269.85 | 2025-10-20 10:15:00 | 265.20 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-17 13:45:00 | 269.25 | 2025-10-20 10:15:00 | 265.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest1 | 2025-12-01 09:15:00 | 285.70 | 2025-12-03 10:15:00 | 277.05 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2025-12-01 10:00:00 | 285.85 | 2025-12-03 10:15:00 | 277.05 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-12-04 09:15:00 | 276.90 | 2025-12-05 09:15:00 | 269.85 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-12-09 15:00:00 | 276.40 | 2025-12-10 10:15:00 | 271.70 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-12-11 11:30:00 | 276.20 | 2025-12-16 12:15:00 | 272.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-16 10:30:00 | 276.40 | 2025-12-16 12:15:00 | 272.60 | STOP_HIT | 1.00 | -1.37% |
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
