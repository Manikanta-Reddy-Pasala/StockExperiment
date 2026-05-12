# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 339.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 6 |
| ALERT3 | 107 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 97 |
| PARTIAL | 14 |
| TARGET_HIT | 10 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 115 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 78
- **Target hits / Stop hits / Partials:** 10 / 91 / 14
- **Avg / median % per leg:** 0.24% / -1.40%
- **Sum % (uncompounded):** 27.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 9 | 28.1% | 9 | 23 | 0 | 0.99% | 31.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 9 | 28.1% | 9 | 23 | 0 | 0.99% | 31.6% |
| SELL (all) | 83 | 28 | 33.7% | 1 | 68 | 14 | -0.05% | -4.5% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.41% | -9.6% |
| SELL @ 3rd Alert (retest2) | 79 | 28 | 35.4% | 1 | 64 | 14 | 0.07% | 5.2% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.41% | -9.6% |
| retest2 (combined) | 111 | 37 | 33.3% | 10 | 87 | 14 | 0.33% | 36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 15:15:00 | 273.90 | 296.79 | 296.84 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 11:15:00 | 304.00 | 296.40 | 296.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-20 09:15:00 | 310.55 | 297.49 | 296.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 12:15:00 | 298.80 | 299.13 | 297.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 12:15:00 | 298.80 | 299.13 | 297.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 298.80 | 299.13 | 297.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 13:00:00 | 298.80 | 299.13 | 297.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 297.95 | 299.12 | 297.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 297.95 | 299.12 | 297.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 297.00 | 299.10 | 297.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 298.40 | 299.10 | 297.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 295.75 | 299.07 | 297.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:45:00 | 301.80 | 299.00 | 297.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 10:45:00 | 301.75 | 299.00 | 297.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 14:15:00 | 291.05 | 298.79 | 297.86 | SL hit (close<static) qty=1.00 sl=291.30 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 13:15:00 | 290.20 | 297.12 | 297.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 287.25 | 296.86 | 297.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 275.45 | 273.18 | 281.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 09:15:00 | 270.75 | 273.63 | 280.91 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:45:00 | 272.70 | 273.48 | 280.55 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:00:00 | 273.00 | 273.47 | 280.51 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 10:15:00 | 273.30 | 273.62 | 280.15 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 276.50 | 273.12 | 278.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-04 10:15:00 | 279.90 | 273.12 | 278.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 279.00 | 273.18 | 278.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-04 10:15:00 | 279.00 | 273.18 | 278.43 | SL hit (close>ema400) qty=1.00 sl=278.43 alert=retest1 |

### Cycle 4 — BUY (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 15:15:00 | 289.90 | 277.60 | 277.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 11:15:00 | 291.50 | 279.70 | 278.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 12:15:00 | 282.00 | 282.55 | 280.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 13:00:00 | 282.00 | 282.55 | 280.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 280.40 | 282.53 | 280.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 15:00:00 | 280.40 | 282.53 | 280.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 279.75 | 282.50 | 280.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 278.10 | 282.50 | 280.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 282.60 | 282.51 | 280.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:45:00 | 282.00 | 282.51 | 280.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 283.45 | 286.40 | 283.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 283.45 | 286.40 | 283.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 281.30 | 286.35 | 283.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 281.30 | 286.35 | 283.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 284.80 | 286.34 | 283.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 286.55 | 285.98 | 283.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 279.80 | 286.39 | 283.88 | SL hit (close<static) qty=1.00 sl=281.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 259.70 | 282.03 | 282.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 259.10 | 279.78 | 280.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 11:15:00 | 277.45 | 276.77 | 279.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-27 12:00:00 | 277.45 | 276.77 | 279.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 279.40 | 276.76 | 279.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:00:00 | 279.40 | 276.76 | 279.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 278.10 | 276.77 | 279.02 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 299.90 | 280.96 | 280.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 301.65 | 282.50 | 281.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 284.55 | 287.71 | 284.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 284.55 | 287.71 | 284.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 284.55 | 287.71 | 284.81 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 13:15:00 | 256.35 | 282.45 | 282.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 255.15 | 274.69 | 278.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 267.95 | 267.66 | 272.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 13:00:00 | 267.95 | 267.66 | 272.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 271.75 | 267.78 | 272.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:45:00 | 272.90 | 267.78 | 272.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 273.85 | 267.76 | 271.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 273.85 | 267.76 | 271.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 273.15 | 267.81 | 271.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 271.25 | 267.87 | 271.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 266.05 | 268.07 | 271.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 257.69 | 267.93 | 271.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 10:15:00 | 268.35 | 267.68 | 271.35 | SL hit (close>ema200) qty=0.50 sl=267.68 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 14:15:00 | 293.00 | 274.29 | 274.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 298.00 | 274.71 | 274.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 296.15 | 296.17 | 289.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 09:30:00 | 294.75 | 296.17 | 289.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 289.60 | 295.87 | 289.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:00:00 | 289.60 | 295.87 | 289.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 288.80 | 295.80 | 289.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 288.80 | 295.80 | 289.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 290.30 | 295.75 | 289.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:15:00 | 290.05 | 295.75 | 289.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 290.05 | 295.69 | 289.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 288.35 | 295.69 | 289.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 290.50 | 295.64 | 289.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:30:00 | 291.40 | 295.58 | 289.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 12:45:00 | 291.60 | 295.48 | 289.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 09:45:00 | 290.90 | 295.33 | 289.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:00:00 | 291.05 | 295.25 | 289.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 288.00 | 295.18 | 289.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 289.70 | 295.18 | 289.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 289.50 | 295.12 | 289.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:15:00 | 287.35 | 295.12 | 289.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 295.20 | 295.12 | 289.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 302.70 | 295.14 | 289.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 12:00:00 | 301.05 | 295.30 | 289.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 13:00:00 | 299.50 | 295.34 | 289.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 13:30:00 | 300.00 | 295.37 | 290.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 292.20 | 295.40 | 290.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:00:00 | 292.20 | 295.40 | 290.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 288.15 | 295.28 | 290.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 288.15 | 295.28 | 290.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 290.10 | 295.23 | 290.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 292.00 | 295.05 | 290.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:30:00 | 290.50 | 297.54 | 292.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 13:00:00 | 292.05 | 297.54 | 292.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 287.35 | 297.47 | 293.48 | SL hit (close<static) qty=1.00 sl=287.80 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 289.50 | 308.71 | 308.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 286.30 | 308.29 | 308.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 278.00 | 275.21 | 286.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 11:00:00 | 278.00 | 275.21 | 286.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 286.95 | 274.63 | 283.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 286.95 | 274.63 | 283.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 286.05 | 274.74 | 283.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 11:30:00 | 284.90 | 274.85 | 283.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:45:00 | 284.80 | 274.94 | 283.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 14:30:00 | 284.70 | 275.16 | 283.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 289.90 | 275.40 | 283.77 | SL hit (close>static) qty=1.00 sl=287.45 alert=retest2 |

### Cycle 10 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 299.35 | 277.02 | 276.96 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 272.95 | 277.76 | 277.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 271.20 | 277.70 | 277.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 279.45 | 277.47 | 277.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 279.45 | 277.47 | 277.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 279.45 | 277.47 | 277.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:30:00 | 279.70 | 277.47 | 277.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 282.70 | 277.52 | 277.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:45:00 | 282.40 | 277.52 | 277.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 276.15 | 275.93 | 276.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 276.95 | 275.93 | 276.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 276.20 | 275.93 | 276.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:30:00 | 276.40 | 275.93 | 276.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 276.55 | 275.94 | 276.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:30:00 | 277.30 | 275.94 | 276.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 276.25 | 275.94 | 276.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:45:00 | 276.25 | 275.94 | 276.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 279.35 | 275.98 | 276.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 12:00:00 | 276.40 | 276.21 | 276.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 12:45:00 | 276.00 | 276.20 | 276.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 11:00:00 | 276.25 | 276.19 | 276.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 11:45:00 | 276.30 | 276.18 | 276.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 273.95 | 274.82 | 276.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 282.05 | 275.11 | 276.12 | SL hit (close>static) qty=1.00 sl=281.15 alert=retest2 |

### Cycle 12 — BUY (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 10:15:00 | 288.30 | 277.04 | 277.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 13:15:00 | 291.45 | 277.41 | 277.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 275.80 | 280.93 | 279.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 275.80 | 280.93 | 279.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 275.80 | 280.93 | 279.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 275.80 | 280.93 | 279.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 273.20 | 280.86 | 279.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:30:00 | 276.05 | 280.86 | 279.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 274.50 | 280.45 | 279.03 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 262.30 | 277.73 | 277.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 245.50 | 277.41 | 277.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 09:15:00 | 275.00 | 272.39 | 274.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 275.00 | 272.39 | 274.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 275.00 | 272.39 | 274.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:00:00 | 275.00 | 272.39 | 274.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 275.00 | 272.42 | 274.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:15:00 | 275.45 | 272.42 | 274.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 276.00 | 272.45 | 274.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 12:00:00 | 276.00 | 272.45 | 274.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 273.35 | 272.46 | 274.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 13:15:00 | 271.20 | 272.46 | 274.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 11:15:00 | 276.40 | 272.85 | 274.76 | SL hit (close>static) qty=1.00 sl=276.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 272.05 | 262.83 | 262.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 276.50 | 263.98 | 263.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 272.35 | 273.03 | 268.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 272.35 | 273.03 | 268.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 348.00 | 368.03 | 348.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 348.00 | 368.03 | 348.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 348.45 | 367.84 | 348.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:30:00 | 352.75 | 366.79 | 348.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 15:15:00 | 353.80 | 363.99 | 350.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 361.80 | 362.91 | 350.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 353.00 | 364.58 | 353.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 348.50 | 364.42 | 353.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 348.50 | 364.42 | 353.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-02 11:15:00 | 344.60 | 364.22 | 353.71 | SL hit (close<static) qty=1.00 sl=346.80 alert=retest2 |

### Cycle 15 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 317.80 | 360.01 | 360.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 314.50 | 355.56 | 357.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 314.75 | 309.36 | 324.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 316.55 | 309.36 | 324.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 320.80 | 310.73 | 322.35 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-28 09:45:00 | 301.80 | 2023-09-28 14:15:00 | 291.05 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2023-09-28 10:45:00 | 301.75 | 2023-09-28 14:15:00 | 291.05 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2023-09-29 09:30:00 | 301.90 | 2023-10-04 09:15:00 | 288.65 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2023-10-03 09:30:00 | 300.80 | 2023-10-04 09:15:00 | 288.65 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2023-10-03 13:30:00 | 299.25 | 2023-10-04 09:15:00 | 288.65 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest1 | 2023-11-17 09:15:00 | 270.75 | 2023-12-04 10:15:00 | 279.00 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest1 | 2023-11-20 09:45:00 | 272.70 | 2023-12-04 10:15:00 | 279.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest1 | 2023-11-20 11:00:00 | 273.00 | 2023-12-04 10:15:00 | 279.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest1 | 2023-11-22 10:15:00 | 273.30 | 2023-12-04 10:15:00 | 279.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2023-12-05 09:15:00 | 273.25 | 2023-12-14 09:15:00 | 282.55 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2023-12-07 12:15:00 | 276.35 | 2023-12-14 09:15:00 | 282.55 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2023-12-20 13:00:00 | 276.30 | 2023-12-21 09:15:00 | 262.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 13:00:00 | 276.30 | 2023-12-27 09:15:00 | 276.75 | STOP_HIT | 0.50 | -0.16% |
| SELL | retest2 | 2023-12-27 12:15:00 | 275.10 | 2023-12-28 13:15:00 | 280.90 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2023-12-28 15:00:00 | 272.85 | 2023-12-29 10:15:00 | 278.65 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-01-02 09:15:00 | 273.70 | 2024-01-03 10:15:00 | 278.40 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-01-09 09:45:00 | 274.00 | 2024-01-11 12:15:00 | 279.15 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-01-09 10:45:00 | 272.80 | 2024-01-11 12:15:00 | 279.15 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-01-12 09:15:00 | 275.90 | 2024-01-12 11:15:00 | 279.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-01-17 10:30:00 | 276.10 | 2024-01-17 12:15:00 | 280.05 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-01-23 10:15:00 | 275.20 | 2024-01-30 10:15:00 | 281.85 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-01-23 12:00:00 | 276.30 | 2024-01-30 10:15:00 | 281.85 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-03-01 09:15:00 | 286.55 | 2024-03-06 11:15:00 | 279.80 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-03-07 09:15:00 | 285.90 | 2024-03-12 15:15:00 | 281.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-03-07 10:45:00 | 287.40 | 2024-03-12 15:15:00 | 281.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-06-03 11:30:00 | 271.25 | 2024-06-04 12:15:00 | 257.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:30:00 | 271.25 | 2024-06-05 10:15:00 | 268.35 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2024-06-04 09:15:00 | 266.05 | 2024-06-05 12:15:00 | 275.05 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2024-07-22 10:30:00 | 291.40 | 2024-08-14 09:15:00 | 287.35 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-07-22 12:45:00 | 291.60 | 2024-08-14 09:15:00 | 287.35 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-07-23 09:45:00 | 290.90 | 2024-08-14 09:15:00 | 287.35 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-23 12:00:00 | 291.05 | 2024-08-27 09:15:00 | 320.54 | TARGET_HIT | 1.00 | 10.13% |
| BUY | retest2 | 2024-07-24 09:15:00 | 302.70 | 2024-08-27 09:15:00 | 320.76 | TARGET_HIT | 1.00 | 5.97% |
| BUY | retest2 | 2024-07-24 12:00:00 | 301.05 | 2024-08-27 09:15:00 | 319.99 | TARGET_HIT | 1.00 | 6.29% |
| BUY | retest2 | 2024-07-24 13:00:00 | 299.50 | 2024-08-27 09:15:00 | 320.16 | TARGET_HIT | 1.00 | 6.90% |
| BUY | retest2 | 2024-07-24 13:30:00 | 300.00 | 2024-08-27 09:15:00 | 320.54 | TARGET_HIT | 1.00 | 6.85% |
| BUY | retest2 | 2024-07-26 09:15:00 | 292.00 | 2024-09-03 09:15:00 | 329.45 | TARGET_HIT | 1.00 | 12.83% |
| BUY | retest2 | 2024-08-05 12:30:00 | 290.50 | 2024-09-03 10:15:00 | 332.97 | TARGET_HIT | 1.00 | 14.62% |
| BUY | retest2 | 2024-08-05 13:00:00 | 292.05 | 2024-09-03 10:15:00 | 331.16 | TARGET_HIT | 1.00 | 13.39% |
| BUY | retest2 | 2024-08-16 09:15:00 | 291.40 | 2024-09-03 10:15:00 | 330.00 | TARGET_HIT | 1.00 | 13.25% |
| SELL | retest2 | 2024-12-03 11:30:00 | 284.90 | 2024-12-04 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-03 12:45:00 | 284.80 | 2024-12-04 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-12-03 14:30:00 | 284.70 | 2024-12-04 09:15:00 | 289.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-12-04 11:15:00 | 284.80 | 2024-12-10 09:15:00 | 285.40 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-12-06 11:45:00 | 279.80 | 2024-12-10 09:15:00 | 285.40 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-12-09 12:30:00 | 279.70 | 2024-12-10 09:15:00 | 285.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-12-09 13:15:00 | 279.65 | 2024-12-13 10:15:00 | 270.56 | PARTIAL | 0.50 | 3.25% |
| SELL | retest2 | 2024-12-09 13:15:00 | 279.65 | 2024-12-16 14:15:00 | 278.30 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2024-12-12 10:00:00 | 279.60 | 2024-12-19 09:15:00 | 265.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 10:00:00 | 279.60 | 2025-01-02 12:15:00 | 273.80 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-01-03 12:00:00 | 277.65 | 2025-01-03 12:15:00 | 280.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-01-03 14:00:00 | 277.80 | 2025-01-06 09:15:00 | 280.05 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-01-06 10:15:00 | 276.15 | 2025-01-09 09:15:00 | 280.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-01-08 15:15:00 | 277.90 | 2025-01-09 09:15:00 | 280.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-10 09:15:00 | 276.90 | 2025-01-13 13:15:00 | 264.10 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2025-01-10 11:15:00 | 278.00 | 2025-01-13 14:15:00 | 263.05 | PARTIAL | 0.50 | 5.38% |
| SELL | retest2 | 2025-01-10 09:15:00 | 276.90 | 2025-01-16 09:15:00 | 273.45 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-01-10 11:15:00 | 278.00 | 2025-01-16 09:15:00 | 273.45 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2025-03-07 12:00:00 | 276.40 | 2025-03-18 14:15:00 | 282.05 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-03-07 12:45:00 | 276.00 | 2025-03-18 14:15:00 | 282.05 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-03-10 11:00:00 | 276.25 | 2025-03-18 14:15:00 | 282.05 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-03-10 11:45:00 | 276.30 | 2025-03-18 14:15:00 | 282.05 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-16 13:15:00 | 271.20 | 2025-04-22 11:15:00 | 276.40 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-04-23 09:30:00 | 272.40 | 2025-04-30 09:15:00 | 258.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-23 09:30:00 | 272.40 | 2025-05-09 11:15:00 | 245.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-13 09:15:00 | 271.05 | 2025-06-26 14:15:00 | 269.30 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-06-16 14:15:00 | 272.50 | 2025-06-26 14:15:00 | 269.30 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-06-18 11:15:00 | 267.25 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-06-18 12:15:00 | 267.20 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-06-24 13:30:00 | 267.35 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-25 13:15:00 | 267.30 | 2025-06-27 10:15:00 | 272.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-06-26 11:00:00 | 266.15 | 2025-07-03 09:15:00 | 268.30 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-26 12:30:00 | 266.05 | 2025-07-03 09:15:00 | 268.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-01 12:00:00 | 266.20 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-02 09:15:00 | 264.60 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-07-02 10:15:00 | 260.45 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-07-02 13:00:00 | 261.55 | 2025-07-03 10:15:00 | 270.15 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-07-02 14:30:00 | 261.35 | 2025-07-07 09:15:00 | 268.55 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-07-02 15:15:00 | 261.20 | 2025-07-07 09:15:00 | 268.55 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-04 11:30:00 | 266.05 | 2025-07-10 13:15:00 | 268.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-04 15:15:00 | 266.35 | 2025-07-10 13:15:00 | 268.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-10 11:45:00 | 266.40 | 2025-07-18 10:15:00 | 258.88 | PARTIAL | 0.50 | 2.82% |
| SELL | retest2 | 2025-07-10 11:45:00 | 266.40 | 2025-07-22 15:15:00 | 266.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2025-07-10 12:45:00 | 266.45 | 2025-07-23 09:15:00 | 257.50 | PARTIAL | 0.50 | 3.36% |
| SELL | retest2 | 2025-07-11 10:30:00 | 265.00 | 2025-07-28 09:15:00 | 251.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 09:30:00 | 264.95 | 2025-07-28 09:15:00 | 251.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 10:30:00 | 265.15 | 2025-07-28 09:15:00 | 251.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-15 11:15:00 | 265.05 | 2025-07-28 09:15:00 | 251.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 261.00 | 2025-07-28 11:15:00 | 247.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 12:45:00 | 266.45 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2025-07-11 10:30:00 | 265.00 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2025-07-15 09:30:00 | 264.95 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2025-07-15 10:30:00 | 265.15 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2025-07-15 11:15:00 | 265.05 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-07-23 09:15:00 | 261.00 | 2025-08-13 11:15:00 | 258.95 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2025-08-19 09:15:00 | 262.65 | 2025-08-20 14:15:00 | 268.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-08-20 14:15:00 | 264.55 | 2025-08-20 14:15:00 | 268.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-08-21 13:00:00 | 264.50 | 2025-08-25 13:15:00 | 267.75 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-28 09:15:00 | 258.20 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-08-28 14:45:00 | 259.90 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-09-02 14:15:00 | 260.00 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-03 09:30:00 | 260.00 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-03 14:30:00 | 259.50 | 2025-09-04 09:15:00 | 263.65 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-01-12 13:30:00 | 352.75 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-01-20 15:15:00 | 353.80 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-01-22 09:15:00 | 361.80 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2026-02-02 09:45:00 | 353.00 | 2026-02-02 11:15:00 | 344.60 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-24 13:15:00 | 364.55 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-24 13:45:00 | 365.00 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-02-24 14:30:00 | 365.00 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-03-02 09:30:00 | 367.50 | 2026-03-04 11:15:00 | 362.25 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-03-04 14:30:00 | 363.45 | 2026-03-09 09:15:00 | 348.25 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-04 15:15:00 | 362.85 | 2026-03-09 09:15:00 | 348.25 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2026-03-05 12:15:00 | 363.30 | 2026-03-09 09:15:00 | 348.25 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2026-03-10 11:45:00 | 363.50 | 2026-03-10 12:15:00 | 359.25 | STOP_HIT | 1.00 | -1.17% |
