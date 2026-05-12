# Aditya Birla Fashion and Retail Ltd. (ABFRL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 66.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 106 |
| ALERT2 | 106 |
| ALERT2_SKIP | 56 |
| ALERT3 | 306 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 131 |
| PARTIAL | 12 |
| TARGET_HIT | 8 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 143 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 101
- **Target hits / Stop hits / Partials:** 8 / 123 / 12
- **Avg / median % per leg:** 0.31% / -0.87%
- **Sum % (uncompounded):** 44.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 14 | 23.0% | 8 | 53 | 0 | 0.34% | 20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 61 | 14 | 23.0% | 8 | 53 | 0 | 0.34% | 20.5% |
| SELL (all) | 82 | 28 | 34.1% | 0 | 70 | 12 | 0.30% | 24.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 82 | 28 | 34.1% | 0 | 70 | 12 | 0.30% | 24.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 143 | 42 | 29.4% | 8 | 123 | 12 | 0.31% | 44.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 12:15:00 | 288.00 | 291.65 | 292.00 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 293.95 | 291.91 | 291.76 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 287.25 | 291.17 | 291.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 268.75 | 286.69 | 289.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 285.45 | 275.41 | 281.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 285.45 | 275.41 | 281.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 285.45 | 275.41 | 281.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 285.45 | 275.41 | 281.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 291.40 | 278.61 | 281.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 291.40 | 278.61 | 281.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 298.10 | 284.81 | 284.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 13:15:00 | 300.00 | 287.85 | 285.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 325.05 | 325.20 | 317.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:00:00 | 325.05 | 325.20 | 317.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 323.00 | 326.42 | 322.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 323.00 | 326.42 | 322.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 324.50 | 326.03 | 323.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:45:00 | 325.40 | 326.07 | 323.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:00:00 | 324.95 | 325.84 | 323.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:30:00 | 325.00 | 325.91 | 323.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 10:15:00 | 323.90 | 327.94 | 328.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 323.90 | 327.94 | 328.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 10:15:00 | 320.15 | 322.77 | 324.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 321.75 | 320.14 | 322.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 321.75 | 320.14 | 322.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 321.75 | 320.14 | 322.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:45:00 | 322.35 | 320.14 | 322.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 323.40 | 320.79 | 322.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:00:00 | 323.40 | 320.79 | 322.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 318.25 | 320.28 | 322.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:15:00 | 318.00 | 320.28 | 322.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:00:00 | 316.75 | 317.85 | 318.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 317.90 | 317.93 | 318.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 15:15:00 | 317.80 | 317.93 | 318.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 317.80 | 317.90 | 318.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 320.00 | 317.90 | 318.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 320.00 | 318.32 | 318.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:15:00 | 321.05 | 318.32 | 318.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 319.90 | 318.64 | 318.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 10:15:00 | 319.90 | 318.64 | 318.63 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 10:15:00 | 316.70 | 318.76 | 318.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 314.25 | 317.24 | 318.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 317.80 | 317.35 | 318.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 317.80 | 317.35 | 318.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 319.75 | 317.83 | 318.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 319.25 | 317.83 | 318.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 320.35 | 318.33 | 318.41 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 320.90 | 318.85 | 318.64 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 11:15:00 | 314.80 | 318.04 | 318.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 13:15:00 | 314.00 | 317.04 | 317.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 316.75 | 315.57 | 316.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 316.75 | 315.57 | 316.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 316.75 | 315.57 | 316.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:15:00 | 318.85 | 315.57 | 316.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 319.60 | 316.38 | 317.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 319.60 | 316.38 | 317.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 319.25 | 316.95 | 317.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 318.35 | 316.95 | 317.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:00:00 | 318.20 | 317.20 | 317.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 319.05 | 317.57 | 317.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 319.05 | 317.57 | 317.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 322.90 | 318.64 | 317.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 319.30 | 321.08 | 319.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 11:15:00 | 319.30 | 321.08 | 319.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 319.30 | 321.08 | 319.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 319.30 | 321.08 | 319.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 319.30 | 320.72 | 319.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 319.30 | 320.72 | 319.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 321.10 | 320.80 | 319.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 325.05 | 320.80 | 319.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 325.45 | 328.29 | 328.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 325.45 | 328.29 | 328.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 322.00 | 327.03 | 327.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 326.80 | 325.05 | 326.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 10:15:00 | 326.80 | 325.05 | 326.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 326.80 | 325.05 | 326.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 326.80 | 325.05 | 326.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 325.00 | 325.04 | 326.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 12:15:00 | 322.90 | 325.04 | 326.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 326.35 | 323.65 | 323.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 326.35 | 323.65 | 323.63 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 322.00 | 323.32 | 323.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 13:15:00 | 320.90 | 322.58 | 323.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 14:15:00 | 323.30 | 322.73 | 323.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 14:15:00 | 323.30 | 322.73 | 323.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 323.30 | 322.73 | 323.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 323.30 | 322.73 | 323.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 324.60 | 323.10 | 323.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 326.90 | 323.10 | 323.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 322.25 | 322.93 | 323.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 326.50 | 322.93 | 323.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 323.00 | 322.95 | 323.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 323.00 | 322.95 | 323.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 322.55 | 322.87 | 323.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:00:00 | 321.00 | 322.49 | 322.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 14:15:00 | 328.05 | 323.39 | 323.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 14:15:00 | 328.05 | 323.39 | 323.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 328.95 | 325.24 | 324.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 329.20 | 329.39 | 327.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 329.20 | 329.39 | 327.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 329.20 | 329.39 | 327.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 328.25 | 329.39 | 327.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 322.60 | 328.03 | 326.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 322.60 | 328.03 | 326.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 325.00 | 327.43 | 326.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 325.50 | 327.04 | 326.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 14:15:00 | 323.60 | 325.99 | 326.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 323.60 | 325.99 | 326.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 314.30 | 323.23 | 324.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 315.00 | 312.76 | 315.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 315.00 | 312.76 | 315.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 315.00 | 312.76 | 315.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 315.00 | 312.76 | 315.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 314.50 | 313.11 | 315.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 315.65 | 313.11 | 315.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 317.30 | 314.02 | 315.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:15:00 | 319.85 | 314.02 | 315.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 320.60 | 315.33 | 315.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 320.60 | 315.33 | 315.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 319.50 | 316.17 | 316.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 322.30 | 318.30 | 317.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 318.75 | 318.98 | 317.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 318.75 | 318.98 | 317.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 318.75 | 318.98 | 317.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:45:00 | 322.00 | 319.99 | 318.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 334.95 | 338.87 | 339.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 334.95 | 338.87 | 339.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 318.45 | 325.52 | 329.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 325.40 | 324.03 | 327.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:00:00 | 325.40 | 324.03 | 327.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 326.90 | 324.48 | 326.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 326.90 | 324.48 | 326.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 323.60 | 324.31 | 326.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:45:00 | 329.05 | 324.31 | 326.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 318.95 | 318.04 | 320.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:30:00 | 319.55 | 318.04 | 320.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 325.00 | 319.43 | 320.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 325.00 | 319.43 | 320.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 323.85 | 320.31 | 320.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:30:00 | 322.20 | 320.73 | 321.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 10:15:00 | 323.20 | 321.22 | 321.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 323.20 | 321.22 | 321.21 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 319.35 | 321.19 | 321.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 312.75 | 319.09 | 320.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 15:15:00 | 313.00 | 312.78 | 315.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 09:15:00 | 313.50 | 312.78 | 315.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 315.25 | 313.27 | 315.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 315.55 | 313.27 | 315.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 315.40 | 313.70 | 315.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 316.45 | 313.70 | 315.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 316.00 | 314.16 | 315.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:30:00 | 316.80 | 314.16 | 315.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 317.95 | 314.92 | 315.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 317.95 | 314.92 | 315.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 320.50 | 316.58 | 316.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 322.20 | 319.22 | 318.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 318.95 | 319.36 | 318.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 318.95 | 319.36 | 318.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 318.95 | 319.36 | 318.71 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 14:15:00 | 317.00 | 318.39 | 318.44 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 319.40 | 318.50 | 318.45 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 12:15:00 | 317.20 | 318.24 | 318.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 14:15:00 | 314.30 | 317.21 | 317.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 13:15:00 | 318.15 | 315.57 | 316.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 13:15:00 | 318.15 | 315.57 | 316.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 318.15 | 315.57 | 316.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 318.15 | 315.57 | 316.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 318.65 | 316.18 | 316.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 318.65 | 316.18 | 316.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 319.35 | 317.16 | 317.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 12:15:00 | 320.90 | 318.65 | 317.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 316.00 | 320.49 | 319.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 316.00 | 320.49 | 319.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 316.00 | 320.49 | 319.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 316.00 | 320.49 | 319.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 316.20 | 319.63 | 319.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:15:00 | 315.20 | 319.63 | 319.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 11:15:00 | 313.80 | 318.46 | 319.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 311.50 | 315.46 | 317.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 313.85 | 313.47 | 315.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 313.85 | 313.47 | 315.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 315.50 | 313.87 | 315.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 317.20 | 313.87 | 315.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 318.00 | 314.70 | 315.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:30:00 | 315.00 | 315.13 | 315.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:30:00 | 314.95 | 315.10 | 315.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:15:00 | 313.70 | 313.59 | 314.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 14:15:00 | 320.60 | 315.26 | 315.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 320.60 | 315.26 | 315.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 323.20 | 317.69 | 316.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 12:15:00 | 317.40 | 318.11 | 316.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 13:00:00 | 317.40 | 318.11 | 316.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 316.55 | 317.80 | 316.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 316.55 | 317.80 | 316.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 315.85 | 317.41 | 316.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 315.85 | 317.41 | 316.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 316.25 | 317.18 | 316.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 316.65 | 317.18 | 316.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 314.00 | 316.67 | 316.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:00:00 | 314.00 | 316.67 | 316.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 312.00 | 315.74 | 316.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 310.70 | 314.73 | 315.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 314.30 | 312.87 | 314.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 314.30 | 312.87 | 314.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 314.30 | 312.87 | 314.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:00:00 | 314.30 | 312.87 | 314.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 315.00 | 313.30 | 314.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:45:00 | 314.55 | 313.30 | 314.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 315.15 | 313.67 | 314.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:30:00 | 315.15 | 313.67 | 314.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 314.85 | 313.90 | 314.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 311.65 | 313.74 | 314.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 15:15:00 | 312.70 | 311.19 | 311.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:30:00 | 313.15 | 311.79 | 312.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 316.60 | 312.75 | 312.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 316.60 | 312.75 | 312.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 317.90 | 314.78 | 313.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 314.40 | 315.37 | 314.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 314.40 | 315.37 | 314.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 314.40 | 315.37 | 314.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 314.40 | 315.37 | 314.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 316.75 | 317.19 | 315.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:15:00 | 321.60 | 317.40 | 315.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 13:15:00 | 325.55 | 330.37 | 330.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 13:15:00 | 325.55 | 330.37 | 330.52 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 339.75 | 331.79 | 330.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 345.40 | 334.51 | 332.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 344.85 | 345.73 | 341.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:45:00 | 348.25 | 345.73 | 341.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 339.60 | 344.33 | 341.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 340.30 | 344.33 | 341.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 341.00 | 343.66 | 341.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:00:00 | 342.25 | 343.38 | 341.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:45:00 | 344.15 | 343.44 | 341.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 13:45:00 | 342.40 | 343.78 | 342.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 342.35 | 343.49 | 342.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 342.00 | 343.19 | 342.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 356.45 | 343.19 | 342.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 14:15:00 | 343.70 | 348.50 | 349.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 343.70 | 348.50 | 349.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 340.40 | 346.13 | 347.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 330.90 | 327.67 | 332.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 330.90 | 327.67 | 332.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 328.70 | 327.88 | 332.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 329.85 | 327.88 | 332.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 332.55 | 329.24 | 332.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:30:00 | 333.50 | 329.24 | 332.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 333.10 | 330.01 | 332.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 333.10 | 330.01 | 332.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 334.05 | 330.82 | 332.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 333.15 | 330.82 | 332.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 342.80 | 333.96 | 333.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 348.05 | 341.29 | 339.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 343.60 | 345.56 | 343.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 343.60 | 345.56 | 343.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 343.60 | 345.56 | 343.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 343.60 | 345.56 | 343.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 345.00 | 345.45 | 343.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:30:00 | 345.90 | 346.06 | 344.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 11:45:00 | 346.00 | 346.88 | 345.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 15:15:00 | 343.00 | 345.12 | 344.79 | SL hit (close<static) qty=1.00 sl=343.05 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 337.35 | 343.57 | 344.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 334.05 | 340.49 | 342.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 335.05 | 334.46 | 337.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 335.05 | 334.46 | 337.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 306.35 | 302.70 | 306.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:45:00 | 306.10 | 302.70 | 306.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 306.30 | 303.42 | 306.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 303.60 | 304.16 | 305.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 308.55 | 304.68 | 304.97 | SL hit (close>static) qty=1.00 sl=307.65 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 308.00 | 305.34 | 305.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 309.70 | 306.80 | 305.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 305.65 | 306.57 | 305.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 13:15:00 | 305.65 | 306.57 | 305.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 305.65 | 306.57 | 305.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 305.65 | 306.57 | 305.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 307.25 | 306.71 | 306.05 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 304.40 | 305.59 | 305.74 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 308.05 | 306.08 | 305.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 316.80 | 308.58 | 307.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 305.35 | 308.80 | 307.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 305.35 | 308.80 | 307.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 305.35 | 308.80 | 307.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 305.35 | 308.80 | 307.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 301.20 | 307.28 | 306.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 301.20 | 307.28 | 306.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 301.30 | 306.08 | 306.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 299.20 | 304.04 | 305.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 302.45 | 301.29 | 303.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 302.45 | 301.29 | 303.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 302.45 | 301.29 | 303.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 302.45 | 301.29 | 303.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 302.10 | 301.45 | 302.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:30:00 | 302.25 | 301.45 | 302.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 302.30 | 301.62 | 302.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 304.40 | 301.62 | 302.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 303.40 | 301.98 | 302.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 302.55 | 301.98 | 302.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 300.90 | 301.76 | 302.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 295.85 | 300.74 | 301.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 304.45 | 300.43 | 301.43 | SL hit (close>static) qty=1.00 sl=303.75 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 11:15:00 | 304.50 | 301.38 | 301.05 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 295.15 | 300.19 | 300.68 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 303.55 | 301.22 | 301.09 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 296.90 | 300.21 | 300.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 294.80 | 297.59 | 298.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 293.50 | 291.93 | 294.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 293.50 | 291.93 | 294.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 293.50 | 291.93 | 294.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 293.50 | 291.93 | 294.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 291.60 | 291.87 | 293.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 294.55 | 291.87 | 293.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 293.50 | 292.19 | 293.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 293.50 | 292.19 | 293.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 292.95 | 291.21 | 292.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 292.95 | 291.21 | 292.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 292.55 | 291.48 | 292.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 293.25 | 291.48 | 292.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 293.15 | 291.82 | 292.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:45:00 | 293.30 | 291.82 | 292.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 291.80 | 291.81 | 292.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 291.80 | 291.81 | 292.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 296.55 | 292.54 | 292.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 296.60 | 292.54 | 292.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 297.30 | 293.49 | 293.02 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 288.30 | 292.34 | 292.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 286.70 | 291.21 | 292.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 288.50 | 287.35 | 289.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 288.50 | 287.35 | 289.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 288.90 | 287.66 | 289.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 288.90 | 287.66 | 289.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 286.30 | 287.39 | 289.06 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 299.70 | 290.52 | 289.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 302.10 | 292.84 | 291.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 313.10 | 313.96 | 310.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 313.10 | 313.96 | 310.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 312.30 | 313.44 | 311.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:45:00 | 312.45 | 313.44 | 311.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 315.90 | 313.95 | 312.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:30:00 | 318.10 | 315.22 | 312.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:45:00 | 317.80 | 318.93 | 316.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 13:15:00 | 313.15 | 316.24 | 316.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 313.15 | 316.24 | 316.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 312.30 | 315.45 | 316.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 308.00 | 307.49 | 310.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 13:00:00 | 308.00 | 307.49 | 310.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 308.45 | 307.31 | 309.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:30:00 | 308.80 | 307.31 | 309.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 307.65 | 307.38 | 308.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 308.15 | 307.38 | 308.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 311.35 | 308.17 | 309.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:15:00 | 311.00 | 308.17 | 309.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 308.35 | 308.21 | 309.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 307.40 | 308.21 | 309.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:15:00 | 307.25 | 308.95 | 309.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 14:30:00 | 307.70 | 308.32 | 308.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 312.30 | 309.12 | 309.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 312.30 | 309.12 | 309.08 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 305.20 | 309.08 | 309.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 303.95 | 307.51 | 308.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 301.35 | 301.22 | 303.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 301.35 | 301.22 | 303.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 301.25 | 300.52 | 301.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:30:00 | 301.65 | 300.52 | 301.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 289.45 | 290.40 | 292.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 293.30 | 290.40 | 292.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 281.35 | 281.67 | 282.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 285.15 | 281.67 | 282.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 284.25 | 282.19 | 283.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:00:00 | 282.05 | 282.37 | 282.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 281.70 | 282.66 | 282.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:15:00 | 281.50 | 282.66 | 282.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 282.45 | 280.40 | 280.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 282.45 | 280.40 | 280.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 283.50 | 281.39 | 280.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 279.55 | 282.90 | 282.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 13:15:00 | 279.55 | 282.90 | 282.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 279.55 | 282.90 | 282.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 279.55 | 282.90 | 282.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 280.50 | 282.42 | 281.99 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 277.55 | 281.04 | 281.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 273.60 | 279.55 | 280.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 270.70 | 270.24 | 273.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:30:00 | 270.65 | 270.24 | 273.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 272.50 | 268.28 | 269.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:00:00 | 272.50 | 268.28 | 269.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 274.45 | 269.51 | 269.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:15:00 | 275.00 | 269.51 | 269.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 14:15:00 | 276.50 | 270.91 | 270.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 15:15:00 | 277.50 | 272.23 | 271.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 269.35 | 271.65 | 270.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 269.35 | 271.65 | 270.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 269.35 | 271.65 | 270.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 269.35 | 271.65 | 270.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 279.05 | 273.13 | 271.71 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 265.90 | 271.75 | 271.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 264.55 | 269.39 | 270.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 267.80 | 265.80 | 268.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 267.80 | 265.80 | 268.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 267.80 | 265.80 | 268.24 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 270.20 | 269.07 | 269.05 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 13:15:00 | 265.85 | 268.95 | 269.05 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 270.75 | 269.11 | 269.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 273.00 | 269.89 | 269.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 274.25 | 274.31 | 272.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:45:00 | 274.45 | 274.31 | 272.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 273.00 | 273.96 | 272.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 273.00 | 273.96 | 272.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 273.85 | 273.94 | 272.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 273.15 | 273.94 | 272.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 277.00 | 280.11 | 278.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 277.00 | 280.11 | 278.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 277.05 | 279.50 | 278.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 278.35 | 279.50 | 278.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 276.15 | 278.99 | 278.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 276.15 | 278.99 | 278.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 275.25 | 278.24 | 278.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:00:00 | 275.25 | 278.24 | 278.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 274.80 | 277.55 | 277.80 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 279.80 | 278.00 | 277.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 280.20 | 278.81 | 278.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 280.25 | 280.60 | 279.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 280.25 | 280.60 | 279.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 280.25 | 280.60 | 279.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 281.35 | 280.60 | 279.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 283.35 | 281.15 | 279.93 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 268.75 | 277.71 | 278.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 261.40 | 269.59 | 273.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 268.90 | 268.51 | 272.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 268.90 | 268.51 | 272.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 271.30 | 268.17 | 270.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 272.35 | 268.17 | 270.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 267.55 | 268.04 | 270.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 267.00 | 268.04 | 270.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 267.05 | 267.90 | 269.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:15:00 | 266.60 | 267.22 | 268.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 266.20 | 267.16 | 268.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 264.75 | 266.58 | 267.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:30:00 | 264.10 | 266.41 | 267.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 272.65 | 267.59 | 268.04 | SL hit (close>static) qty=1.00 sl=271.55 alert=retest2 |

### Cycle 58 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 272.90 | 268.66 | 268.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 274.05 | 269.73 | 268.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 09:15:00 | 284.15 | 284.50 | 280.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:00:00 | 284.15 | 284.50 | 280.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 285.10 | 283.74 | 281.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:45:00 | 286.00 | 284.15 | 281.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:45:00 | 286.15 | 286.08 | 284.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 10:15:00 | 280.65 | 283.79 | 283.53 | SL hit (close<static) qty=1.00 sl=281.40 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 279.45 | 282.92 | 283.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 277.80 | 281.40 | 282.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 256.55 | 255.91 | 261.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 256.55 | 255.91 | 261.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 258.00 | 256.83 | 260.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 253.50 | 257.26 | 259.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 256.10 | 258.12 | 259.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 240.82 | 250.03 | 253.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 243.30 | 250.03 | 253.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 10:15:00 | 252.50 | 250.52 | 253.63 | SL hit (close>ema200) qty=0.50 sl=250.52 alert=retest2 |

### Cycle 60 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 255.15 | 252.48 | 252.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 256.20 | 253.46 | 252.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 251.25 | 253.59 | 253.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 251.25 | 253.59 | 253.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 251.25 | 253.59 | 253.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 251.25 | 253.59 | 253.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 250.65 | 253.00 | 252.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 250.65 | 253.00 | 252.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 250.70 | 252.54 | 252.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 249.80 | 251.09 | 251.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 252.00 | 251.22 | 251.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 252.00 | 251.22 | 251.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 252.00 | 251.22 | 251.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 252.30 | 251.22 | 251.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 252.95 | 251.57 | 251.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 252.55 | 251.57 | 251.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 253.00 | 251.95 | 251.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:45:00 | 253.50 | 251.95 | 251.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 252.00 | 251.96 | 251.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 250.85 | 251.96 | 251.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 251.40 | 251.85 | 251.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 249.80 | 251.21 | 251.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 12:30:00 | 249.85 | 250.92 | 251.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 237.31 | 243.87 | 246.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 237.36 | 243.87 | 246.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 239.14 | 238.42 | 241.57 | SL hit (close>ema200) qty=0.50 sl=238.42 alert=retest2 |

### Cycle 62 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 244.11 | 241.48 | 241.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 245.16 | 243.65 | 242.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 241.11 | 243.14 | 242.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 241.11 | 243.14 | 242.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 241.11 | 243.14 | 242.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 241.11 | 243.14 | 242.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 240.64 | 242.64 | 242.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 240.64 | 242.64 | 242.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 241.20 | 242.35 | 242.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 10:15:00 | 240.33 | 241.92 | 242.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 239.95 | 239.05 | 240.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 239.95 | 239.05 | 240.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 239.95 | 239.05 | 240.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:45:00 | 240.09 | 239.05 | 240.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 240.98 | 239.43 | 240.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 240.98 | 239.43 | 240.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 241.45 | 239.84 | 240.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 14:15:00 | 240.71 | 240.07 | 240.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:00:00 | 240.43 | 240.15 | 240.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 239.64 | 240.12 | 240.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 243.73 | 241.14 | 240.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 243.73 | 241.14 | 240.83 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 239.30 | 240.83 | 240.85 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 242.39 | 240.77 | 240.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 243.72 | 241.49 | 241.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 246.17 | 248.03 | 245.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 246.17 | 248.03 | 245.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 246.17 | 248.03 | 245.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 246.17 | 248.03 | 245.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 255.95 | 255.08 | 253.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 252.64 | 255.08 | 253.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 252.86 | 254.63 | 253.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 252.86 | 254.63 | 253.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 257.39 | 255.19 | 253.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 258.22 | 255.24 | 254.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 257.91 | 255.66 | 254.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 258.12 | 257.33 | 255.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 262.79 | 255.95 | 255.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 257.30 | 258.19 | 257.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 257.30 | 258.19 | 257.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 256.31 | 257.82 | 257.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 256.31 | 257.82 | 257.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 256.20 | 257.49 | 256.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 256.20 | 257.49 | 256.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 254.60 | 256.91 | 256.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 261.40 | 256.91 | 256.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:00:00 | 257.70 | 257.51 | 257.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 13:30:00 | 257.85 | 257.85 | 257.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 13:15:00 | 257.65 | 260.59 | 260.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 257.65 | 260.59 | 260.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 241.15 | 256.25 | 258.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 248.00 | 247.31 | 251.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 248.00 | 247.31 | 251.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 251.60 | 247.58 | 248.51 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 254.15 | 250.02 | 249.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 260.85 | 253.92 | 251.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 266.55 | 266.85 | 263.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 266.55 | 266.85 | 263.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 271.20 | 268.88 | 266.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:45:00 | 271.95 | 270.13 | 267.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 272.35 | 271.26 | 269.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:15:00 | 271.90 | 270.94 | 269.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:00:00 | 272.10 | 271.29 | 269.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 273.50 | 272.42 | 271.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 263.40 | 269.75 | 270.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 263.40 | 269.75 | 270.35 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 273.60 | 269.21 | 269.07 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 268.20 | 270.14 | 270.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 265.55 | 268.65 | 269.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 259.60 | 259.08 | 262.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:45:00 | 260.00 | 259.08 | 262.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 261.65 | 259.60 | 262.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 261.45 | 259.60 | 262.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 263.00 | 260.90 | 262.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:30:00 | 263.25 | 260.90 | 262.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 262.85 | 261.29 | 262.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 262.85 | 261.29 | 262.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 263.40 | 261.71 | 262.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 261.50 | 261.71 | 262.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 260.00 | 259.09 | 260.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 260.85 | 259.09 | 260.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 263.50 | 259.97 | 260.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 263.50 | 259.97 | 260.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 265.30 | 261.04 | 261.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 264.30 | 261.04 | 261.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 263.20 | 261.53 | 261.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 14:15:00 | 264.70 | 262.16 | 261.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 260.50 | 264.64 | 263.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 260.50 | 264.64 | 263.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 260.50 | 264.64 | 263.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 260.50 | 264.64 | 263.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 259.05 | 263.52 | 263.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 259.05 | 263.52 | 263.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 257.40 | 262.30 | 262.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 255.15 | 260.87 | 261.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 263.70 | 257.28 | 258.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 263.70 | 257.28 | 258.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 263.70 | 257.28 | 258.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 263.70 | 257.28 | 258.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 266.05 | 260.37 | 260.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 268.60 | 263.81 | 261.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 268.50 | 268.51 | 265.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 268.50 | 268.51 | 265.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 290.85 | 280.98 | 277.95 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 273.60 | 280.53 | 280.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 271.00 | 277.55 | 279.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 89.10 | 89.10 | 109.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 11:00:00 | 89.10 | 89.10 | 109.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 87.50 | 87.18 | 87.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 87.72 | 87.18 | 87.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 77.87 | 77.15 | 77.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 77.87 | 77.15 | 77.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 77.69 | 77.26 | 77.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 77.42 | 77.29 | 77.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 73.55 | 75.71 | 76.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 74.41 | 74.34 | 75.31 | SL hit (close>ema200) qty=0.50 sl=74.34 alert=retest2 |

### Cycle 76 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 74.65 | 73.95 | 73.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 74.80 | 74.12 | 73.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 74.45 | 74.53 | 74.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 11:00:00 | 74.45 | 74.53 | 74.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 74.84 | 74.62 | 74.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:45:00 | 74.44 | 74.62 | 74.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 74.36 | 74.56 | 74.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 74.36 | 74.56 | 74.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 74.07 | 74.47 | 74.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 74.07 | 74.47 | 74.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 73.95 | 74.36 | 74.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 74.39 | 74.36 | 74.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 73.81 | 74.33 | 74.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 73.81 | 74.33 | 74.37 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 74.83 | 74.34 | 74.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 75.38 | 74.78 | 74.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 74.61 | 74.82 | 74.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 74.61 | 74.82 | 74.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 74.61 | 74.82 | 74.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 74.61 | 74.82 | 74.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 74.54 | 74.76 | 74.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 74.43 | 74.76 | 74.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 75.02 | 74.81 | 74.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 75.01 | 74.81 | 74.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 73.80 | 74.64 | 74.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 73.80 | 74.64 | 74.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 74.45 | 74.61 | 74.61 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 75.08 | 74.71 | 74.66 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 74.45 | 74.61 | 74.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 74.19 | 74.51 | 74.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 74.66 | 74.54 | 74.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 74.66 | 74.54 | 74.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 74.66 | 74.54 | 74.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 74.66 | 74.54 | 74.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 13:15:00 | 76.31 | 74.89 | 74.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 78.08 | 75.53 | 75.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 76.64 | 76.81 | 76.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:30:00 | 76.68 | 76.81 | 76.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 77.67 | 76.93 | 76.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:15:00 | 78.38 | 76.93 | 76.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:45:00 | 77.84 | 77.54 | 76.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 78.07 | 77.59 | 76.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 77.97 | 77.75 | 76.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 77.61 | 77.92 | 77.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 77.61 | 77.92 | 77.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 78.86 | 78.10 | 77.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 77.75 | 78.10 | 77.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 78.41 | 78.74 | 78.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 78.41 | 78.74 | 78.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 78.96 | 78.79 | 78.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 77.70 | 78.25 | 78.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 77.70 | 78.25 | 78.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 76.59 | 77.77 | 78.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 77.04 | 76.99 | 77.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 77.04 | 76.99 | 77.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 77.23 | 77.04 | 77.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 77.35 | 77.04 | 77.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 76.70 | 76.97 | 77.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 76.56 | 76.89 | 77.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 77.39 | 77.02 | 77.13 | SL hit (close>static) qty=1.00 sl=77.31 alert=retest2 |

### Cycle 84 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 75.65 | 75.44 | 75.43 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 74.94 | 75.34 | 75.39 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 75.70 | 75.39 | 75.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 75.92 | 75.50 | 75.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 75.37 | 75.52 | 75.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 11:15:00 | 75.37 | 75.52 | 75.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 75.37 | 75.52 | 75.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 75.37 | 75.52 | 75.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 75.17 | 75.45 | 75.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 75.17 | 75.45 | 75.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 75.16 | 75.39 | 75.40 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 75.80 | 75.47 | 75.43 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 74.97 | 75.34 | 75.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 74.67 | 75.20 | 75.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 74.50 | 74.36 | 74.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 74.37 | 74.36 | 74.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 74.62 | 73.93 | 74.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 74.62 | 73.93 | 74.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 74.09 | 73.96 | 74.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:15:00 | 74.63 | 73.96 | 74.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 74.55 | 74.08 | 74.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 74.54 | 74.08 | 74.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 75.66 | 74.39 | 74.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 75.66 | 74.39 | 74.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 75.27 | 74.57 | 74.50 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 73.68 | 74.54 | 74.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 12:15:00 | 73.58 | 74.13 | 74.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 13:15:00 | 74.16 | 73.59 | 73.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 74.16 | 73.59 | 73.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 74.16 | 73.59 | 73.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 74.16 | 73.59 | 73.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 73.45 | 73.56 | 73.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:30:00 | 74.06 | 73.56 | 73.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 73.54 | 73.47 | 73.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:15:00 | 74.00 | 73.47 | 73.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 74.05 | 73.59 | 73.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:30:00 | 74.07 | 73.59 | 73.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 74.69 | 73.81 | 73.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 74.69 | 73.81 | 73.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 75.02 | 74.05 | 73.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 75.55 | 74.35 | 74.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 75.29 | 75.42 | 74.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:00:00 | 75.29 | 75.42 | 74.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 75.32 | 75.40 | 74.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 75.32 | 75.40 | 74.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 75.93 | 75.51 | 75.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:45:00 | 77.01 | 75.91 | 75.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 76.24 | 75.86 | 75.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 75.02 | 75.76 | 75.64 | SL hit (close<static) qty=1.00 sl=75.04 alert=retest2 |

### Cycle 93 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 75.23 | 75.55 | 75.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 74.77 | 75.35 | 75.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 74.71 | 74.71 | 74.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 14:45:00 | 74.77 | 74.71 | 74.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 74.70 | 74.66 | 74.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:15:00 | 74.36 | 74.57 | 74.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:45:00 | 74.25 | 74.47 | 74.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 76.10 | 74.90 | 74.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 76.10 | 74.90 | 74.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 77.61 | 75.65 | 75.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 76.90 | 77.06 | 76.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 76.90 | 77.06 | 76.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 76.90 | 77.06 | 76.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 76.60 | 77.06 | 76.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 76.62 | 76.95 | 76.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 76.57 | 76.95 | 76.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 76.70 | 76.90 | 76.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:15:00 | 76.46 | 76.90 | 76.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 76.66 | 76.85 | 76.64 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 75.90 | 76.49 | 76.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 75.62 | 76.16 | 76.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 15:15:00 | 76.10 | 75.88 | 76.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 76.81 | 75.88 | 76.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 96 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 79.94 | 76.70 | 76.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 11:15:00 | 81.18 | 77.97 | 77.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 82.36 | 82.40 | 81.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 81.06 | 82.40 | 81.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 81.36 | 82.19 | 81.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 81.69 | 82.19 | 81.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 82.08 | 82.17 | 81.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 81.19 | 82.17 | 81.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 79.71 | 81.68 | 81.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 79.71 | 81.68 | 81.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 79.85 | 81.31 | 81.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:15:00 | 79.48 | 81.31 | 81.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 78.86 | 80.82 | 80.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 78.01 | 80.26 | 80.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 77.97 | 77.95 | 78.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 77.97 | 77.95 | 78.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 81.12 | 78.59 | 79.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 81.12 | 78.59 | 79.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 81.24 | 79.12 | 79.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 81.44 | 79.12 | 79.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 83.37 | 79.97 | 79.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 83.58 | 80.69 | 79.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 83.00 | 83.11 | 81.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 82.12 | 83.11 | 81.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 82.37 | 82.96 | 82.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 84.36 | 81.82 | 81.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:30:00 | 83.28 | 83.17 | 82.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 83.38 | 83.24 | 82.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:00:00 | 83.29 | 83.25 | 82.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-08 12:15:00 | 92.80 | 87.16 | 85.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 87.57 | 88.56 | 88.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 86.79 | 88.20 | 88.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 88.28 | 87.95 | 88.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 88.28 | 87.95 | 88.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 88.28 | 87.95 | 88.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 88.26 | 87.95 | 88.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 88.45 | 88.05 | 88.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 87.94 | 88.05 | 88.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 87.82 | 88.00 | 88.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 87.80 | 87.99 | 88.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 87.99 | 87.74 | 87.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 87.44 | 87.68 | 87.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 87.28 | 87.68 | 87.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 88.85 | 87.41 | 87.62 | SL hit (close>static) qty=1.00 sl=88.73 alert=retest2 |

### Cycle 100 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 89.80 | 87.89 | 87.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 90.71 | 88.45 | 88.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 15:15:00 | 89.74 | 89.88 | 89.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:15:00 | 89.96 | 89.88 | 89.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 89.58 | 89.82 | 89.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 89.58 | 89.82 | 89.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 91.82 | 90.22 | 89.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 93.64 | 91.34 | 90.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 93.20 | 91.82 | 90.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 15:00:00 | 93.52 | 93.01 | 91.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 91.22 | 91.66 | 91.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 91.22 | 91.66 | 91.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 89.91 | 91.07 | 91.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 85.27 | 84.82 | 86.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 85.17 | 84.82 | 86.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 85.38 | 84.80 | 85.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 85.03 | 84.80 | 85.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 85.48 | 84.93 | 85.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 85.48 | 84.93 | 85.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 85.00 | 84.95 | 85.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 84.99 | 85.09 | 85.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 86.26 | 85.33 | 85.43 | SL hit (close>static) qty=1.00 sl=85.55 alert=retest2 |

### Cycle 102 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 87.00 | 85.66 | 85.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 87.57 | 86.04 | 85.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 87.50 | 87.60 | 86.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 87.75 | 87.60 | 86.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 86.91 | 87.49 | 86.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 86.97 | 87.49 | 86.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 87.00 | 87.39 | 86.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 86.91 | 87.39 | 86.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 86.90 | 87.30 | 86.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 86.97 | 87.30 | 86.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 86.60 | 87.16 | 86.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 86.60 | 87.16 | 86.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 86.23 | 86.97 | 86.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 86.30 | 86.97 | 86.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 86.06 | 86.79 | 86.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 85.80 | 86.79 | 86.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 86.02 | 86.64 | 86.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 85.25 | 85.88 | 86.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 86.22 | 85.94 | 86.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 86.22 | 85.94 | 86.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 85.75 | 85.91 | 86.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 85.37 | 85.91 | 86.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 81.10 | 81.90 | 82.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 81.32 | 81.09 | 82.04 | SL hit (close>ema200) qty=0.50 sl=81.09 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 83.35 | 82.52 | 82.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 83.75 | 83.16 | 82.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 82.82 | 83.09 | 82.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 82.82 | 83.09 | 82.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 82.82 | 83.09 | 82.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 83.06 | 83.09 | 82.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 82.75 | 83.02 | 82.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 82.76 | 83.02 | 82.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 82.69 | 82.96 | 82.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 82.52 | 82.96 | 82.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 82.10 | 82.79 | 82.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 82.10 | 82.79 | 82.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 81.95 | 82.62 | 82.69 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 82.98 | 82.70 | 82.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 83.68 | 83.00 | 82.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 83.58 | 83.67 | 83.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:45:00 | 83.43 | 83.67 | 83.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 83.32 | 83.55 | 83.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 83.19 | 83.55 | 83.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 83.00 | 83.44 | 83.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 83.11 | 83.44 | 83.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 82.85 | 83.32 | 83.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 82.85 | 83.32 | 83.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 81.91 | 83.04 | 83.12 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 84.33 | 82.84 | 82.65 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 82.85 | 83.25 | 83.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 82.60 | 83.12 | 83.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 84.15 | 82.52 | 82.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 84.15 | 82.52 | 82.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 84.15 | 82.52 | 82.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 84.15 | 82.52 | 82.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 83.23 | 82.66 | 82.75 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 83.65 | 82.86 | 82.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 84.36 | 83.16 | 82.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 84.03 | 84.36 | 83.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 14:15:00 | 84.03 | 84.36 | 83.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 84.03 | 84.36 | 83.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 84.03 | 84.36 | 83.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 84.05 | 84.30 | 83.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 80.60 | 84.30 | 83.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 79.06 | 83.25 | 83.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 78.00 | 80.94 | 82.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 79.25 | 78.44 | 79.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 79.25 | 78.44 | 79.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 79.25 | 78.44 | 79.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 79.73 | 78.44 | 79.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 79.76 | 78.71 | 79.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 79.76 | 78.71 | 79.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 79.64 | 78.89 | 79.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 79.22 | 78.89 | 79.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 79.39 | 79.28 | 79.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 80.20 | 79.46 | 79.81 | SL hit (close>static) qty=1.00 sl=79.98 alert=retest2 |

### Cycle 112 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 81.17 | 79.80 | 79.62 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 79.50 | 79.79 | 79.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 78.88 | 79.61 | 79.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 78.54 | 78.51 | 78.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:45:00 | 78.46 | 78.51 | 78.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 78.04 | 78.03 | 78.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:30:00 | 77.55 | 78.00 | 78.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 77.70 | 77.96 | 78.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 78.77 | 78.08 | 78.34 | SL hit (close>static) qty=1.00 sl=78.50 alert=retest2 |

### Cycle 114 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 76.94 | 76.47 | 76.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 77.15 | 76.82 | 76.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 77.85 | 77.88 | 77.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 77.76 | 77.88 | 77.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 77.49 | 77.80 | 77.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 77.49 | 77.80 | 77.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 76.87 | 77.61 | 77.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 76.87 | 77.61 | 77.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 76.89 | 77.47 | 77.38 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 76.51 | 77.20 | 77.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 76.39 | 76.91 | 77.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 76.85 | 76.77 | 76.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:30:00 | 76.59 | 76.77 | 76.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 76.75 | 76.78 | 76.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:15:00 | 76.94 | 76.78 | 76.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 77.07 | 76.84 | 76.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 77.07 | 76.84 | 76.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 76.50 | 76.77 | 76.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:00:00 | 76.32 | 76.68 | 76.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 76.41 | 76.55 | 76.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 77.30 | 76.82 | 76.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 77.30 | 76.82 | 76.79 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 76.06 | 76.71 | 76.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 75.78 | 76.52 | 76.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 75.62 | 75.54 | 76.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:30:00 | 75.28 | 75.54 | 76.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 76.00 | 75.63 | 75.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 76.00 | 75.63 | 75.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 76.00 | 75.71 | 75.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 76.66 | 75.71 | 75.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 77.20 | 76.00 | 76.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 77.54 | 76.00 | 76.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 76.26 | 76.06 | 76.07 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 76.51 | 76.15 | 76.11 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 75.72 | 76.05 | 76.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 75.59 | 75.96 | 76.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 75.57 | 75.36 | 75.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 14:15:00 | 75.57 | 75.36 | 75.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 75.57 | 75.36 | 75.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 75.57 | 75.36 | 75.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 75.45 | 75.38 | 75.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 76.23 | 75.38 | 75.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 76.64 | 75.63 | 75.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 76.64 | 75.63 | 75.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 76.60 | 75.83 | 75.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 77.41 | 76.27 | 75.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 76.34 | 76.83 | 76.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 76.34 | 76.83 | 76.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 76.34 | 76.83 | 76.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 76.34 | 76.83 | 76.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 76.06 | 76.67 | 76.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 76.06 | 76.67 | 76.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 75.38 | 76.22 | 76.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 15:15:00 | 75.32 | 76.04 | 76.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 76.52 | 74.96 | 75.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 76.52 | 74.96 | 75.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 76.52 | 74.96 | 75.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 76.52 | 74.96 | 75.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 77.42 | 75.46 | 75.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 77.87 | 75.46 | 75.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 77.67 | 75.90 | 75.72 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 75.43 | 75.80 | 75.83 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 76.09 | 75.79 | 75.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 76.32 | 75.92 | 75.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 77.75 | 78.16 | 77.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 77.75 | 78.16 | 77.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 78.91 | 78.31 | 77.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 78.26 | 78.31 | 77.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 77.87 | 78.28 | 77.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 77.87 | 78.28 | 77.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 77.71 | 78.17 | 77.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 77.65 | 78.17 | 77.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 77.75 | 78.08 | 77.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:15:00 | 77.36 | 78.08 | 77.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 77.30 | 77.93 | 77.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 77.30 | 77.93 | 77.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 77.01 | 77.60 | 77.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 76.77 | 77.44 | 77.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 77.28 | 76.95 | 77.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 77.28 | 76.95 | 77.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 77.28 | 76.95 | 77.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 77.28 | 76.95 | 77.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 77.11 | 76.98 | 77.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 77.14 | 76.98 | 77.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 76.80 | 76.95 | 77.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 76.49 | 76.95 | 77.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 76.26 | 76.78 | 77.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 76.61 | 76.33 | 76.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 77.22 | 76.51 | 76.64 | SL hit (close>static) qty=1.00 sl=77.20 alert=retest2 |

### Cycle 126 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 77.20 | 76.80 | 76.76 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 76.45 | 76.73 | 76.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 75.97 | 76.56 | 76.65 | Break + close below crossover candle low |

### Cycle 128 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 77.72 | 76.67 | 76.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 78.23 | 77.45 | 77.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 77.31 | 77.48 | 77.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 77.31 | 77.48 | 77.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 77.24 | 77.43 | 77.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 77.05 | 77.43 | 77.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 77.20 | 77.38 | 77.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:15:00 | 77.50 | 77.38 | 77.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:45:00 | 77.32 | 77.20 | 77.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 76.17 | 76.97 | 77.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 76.17 | 76.97 | 77.03 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 77.50 | 77.06 | 77.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 77.86 | 77.29 | 77.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 77.26 | 77.33 | 77.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 11:15:00 | 77.26 | 77.33 | 77.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 77.26 | 77.33 | 77.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 77.26 | 77.33 | 77.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 77.25 | 77.32 | 77.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 76.81 | 77.32 | 77.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 76.85 | 77.22 | 77.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:15:00 | 76.80 | 77.22 | 77.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 77.21 | 77.22 | 77.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 77.06 | 77.22 | 77.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 77.48 | 77.27 | 77.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 77.15 | 77.27 | 77.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 78.27 | 77.47 | 77.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 77.65 | 77.47 | 77.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 77.11 | 77.40 | 77.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 77.11 | 77.40 | 77.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 76.22 | 77.16 | 77.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 75.90 | 76.91 | 77.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 76.34 | 76.15 | 76.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 76.34 | 76.15 | 76.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 76.34 | 76.15 | 76.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 76.65 | 76.15 | 76.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 75.55 | 76.03 | 76.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 75.43 | 75.88 | 76.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 71.66 | 72.16 | 72.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 71.43 | 70.74 | 71.65 | SL hit (close>ema200) qty=0.50 sl=70.74 alert=retest2 |

### Cycle 132 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 64.72 | 63.31 | 63.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 65.26 | 63.70 | 63.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 63.73 | 63.97 | 63.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 63.73 | 63.97 | 63.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 63.73 | 63.97 | 63.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 63.74 | 63.97 | 63.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 63.64 | 63.90 | 63.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:30:00 | 64.02 | 63.89 | 63.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 64.29 | 64.03 | 63.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 65.19 | 64.55 | 64.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 63.94 | 65.15 | 65.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 63.94 | 65.15 | 65.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 63.19 | 64.76 | 65.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 64.12 | 64.08 | 64.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 64.12 | 64.08 | 64.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 66.91 | 64.63 | 64.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 66.71 | 64.63 | 64.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 66.45 | 65.00 | 64.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 66.45 | 65.00 | 64.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 67.77 | 65.94 | 65.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 66.80 | 69.39 | 68.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 66.80 | 69.39 | 68.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 66.80 | 69.39 | 68.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 66.80 | 69.39 | 68.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 67.25 | 68.96 | 68.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 66.80 | 68.96 | 68.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 69.89 | 68.90 | 68.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:30:00 | 70.90 | 69.28 | 68.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 71.99 | 70.15 | 69.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 70.07 | 71.41 | 71.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 70.07 | 71.41 | 71.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 69.98 | 70.40 | 70.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 74.20 | 70.80 | 70.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 74.20 | 70.80 | 70.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 74.20 | 70.80 | 70.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:45:00 | 74.15 | 70.80 | 70.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 72.44 | 71.13 | 70.98 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 71.65 | 72.08 | 72.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 71.25 | 71.77 | 71.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 70.74 | 70.14 | 70.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 70.74 | 70.14 | 70.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 70.74 | 70.14 | 70.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 69.10 | 69.79 | 70.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 69.14 | 69.79 | 70.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 69.26 | 69.67 | 70.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 68.82 | 69.15 | 69.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 69.66 | 69.05 | 69.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 69.85 | 69.05 | 69.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 69.12 | 69.06 | 69.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 68.72 | 68.92 | 69.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.64 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.68 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.80 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.38 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.28 | 67.10 | 67.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 66.20 | 66.09 | 66.91 | SL hit (close>ema200) qty=0.50 sl=66.09 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 15:15:00 | 62.15 | 62.00 | 62.00 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 61.07 | 61.82 | 61.92 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 62.80 | 62.12 | 62.03 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 60.10 | 61.73 | 61.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 59.87 | 60.91 | 61.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 58.63 | 58.54 | 59.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 58.63 | 58.54 | 59.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 59.39 | 58.64 | 59.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 59.61 | 58.64 | 59.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 58.86 | 58.68 | 59.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 58.27 | 59.19 | 59.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 62.34 | 58.64 | 58.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 62.34 | 58.64 | 58.61 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 56.34 | 58.59 | 58.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 55.92 | 58.06 | 58.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 57.79 | 57.29 | 57.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 57.79 | 57.29 | 57.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 57.79 | 57.29 | 57.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 57.77 | 57.29 | 57.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 57.87 | 57.41 | 57.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:00:00 | 57.87 | 57.41 | 57.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 57.85 | 57.50 | 57.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 57.85 | 57.50 | 57.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 58.95 | 57.79 | 57.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 58.97 | 57.79 | 57.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 58.81 | 57.99 | 58.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:15:00 | 58.42 | 57.99 | 58.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 58.20 | 58.07 | 58.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 58.20 | 58.07 | 58.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 60.22 | 58.50 | 58.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 59.04 | 59.19 | 58.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 59.04 | 59.19 | 58.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 58.78 | 59.11 | 58.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 58.21 | 59.11 | 58.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 57.58 | 58.80 | 58.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 57.58 | 58.80 | 58.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 57.05 | 58.45 | 58.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 56.20 | 57.36 | 57.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 57.49 | 55.71 | 56.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 57.49 | 55.71 | 56.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 57.49 | 55.71 | 56.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 57.49 | 55.71 | 56.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 57.28 | 56.02 | 56.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 57.55 | 56.02 | 56.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 58.03 | 57.18 | 57.07 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 55.66 | 56.93 | 56.98 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 57.71 | 56.94 | 56.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 59.07 | 57.70 | 57.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 57.91 | 58.26 | 57.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 57.91 | 58.26 | 57.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 58.16 | 58.24 | 57.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 59.20 | 57.99 | 57.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 58.35 | 58.82 | 58.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 65.12 | 61.42 | 60.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 64.00 | 64.59 | 64.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 63.63 | 64.30 | 64.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 64.00 | 62.63 | 63.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 64.00 | 62.63 | 63.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 64.00 | 62.63 | 63.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 64.00 | 62.63 | 63.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 64.31 | 62.97 | 63.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 64.36 | 62.97 | 63.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 63.99 | 63.56 | 63.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 64.05 | 63.74 | 63.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 63.70 | 63.73 | 63.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 63.70 | 63.73 | 63.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 63.70 | 63.73 | 63.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 63.67 | 63.73 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 63.55 | 63.69 | 63.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 63.60 | 63.69 | 63.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 63.70 | 63.70 | 63.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 63.52 | 63.70 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 63.58 | 63.67 | 63.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 63.58 | 63.67 | 63.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 63.64 | 63.67 | 63.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 64.37 | 63.67 | 63.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 63.75 | 63.67 | 63.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 64.19 | 63.78 | 63.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 63.75 | 64.25 | 64.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 63.05 | 64.01 | 64.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 63.05 | 64.01 | 64.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 62.86 | 63.78 | 63.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 66.00 | 64.07 | 64.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 14:15:00 | 66.00 | 64.07 | 64.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 66.00 | 64.07 | 64.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 66.00 | 64.07 | 64.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 65.93 | 64.44 | 64.24 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 63.90 | 64.10 | 64.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 63.81 | 64.00 | 64.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 11:15:00 | 64.00 | 63.98 | 64.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:30:00 | 63.99 | 63.98 | 64.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 64.38 | 64.06 | 64.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:15:00 | 65.36 | 64.06 | 64.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 66.15 | 64.48 | 64.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 67.04 | 65.23 | 64.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 66.77 | 67.16 | 66.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:45:00 | 66.85 | 67.16 | 66.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 66.65 | 67.01 | 66.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 66.65 | 67.01 | 66.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 66.39 | 66.88 | 66.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 66.39 | 66.88 | 66.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 66.37 | 66.78 | 66.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 66.15 | 66.78 | 66.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 66.15 | 66.66 | 66.34 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 14:30:00 | 256.75 | 2024-05-21 11:15:00 | 282.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 11:15:00 | 256.65 | 2024-05-21 11:15:00 | 282.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 09:45:00 | 325.40 | 2024-06-19 10:15:00 | 323.90 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-06-12 11:00:00 | 324.95 | 2024-06-19 10:15:00 | 323.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-06-12 11:30:00 | 325.00 | 2024-06-19 10:15:00 | 323.90 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-06-21 12:15:00 | 318.00 | 2024-06-26 10:15:00 | 319.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-06-25 12:00:00 | 316.75 | 2024-06-26 10:15:00 | 319.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-06-25 14:45:00 | 317.90 | 2024-06-26 10:15:00 | 319.90 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-06-25 15:15:00 | 317.80 | 2024-06-26 10:15:00 | 319.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-07-01 12:15:00 | 318.35 | 2024-07-01 13:15:00 | 319.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-01 13:00:00 | 318.20 | 2024-07-01 13:15:00 | 319.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-07-02 14:15:00 | 325.05 | 2024-07-08 10:15:00 | 325.45 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-07-09 12:15:00 | 322.90 | 2024-07-11 10:15:00 | 326.35 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-07-15 11:00:00 | 321.00 | 2024-07-15 14:15:00 | 328.05 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-07-18 13:00:00 | 325.50 | 2024-07-18 14:15:00 | 323.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-07-25 13:45:00 | 322.00 | 2024-08-02 09:15:00 | 334.95 | STOP_HIT | 1.00 | 4.02% |
| SELL | retest2 | 2024-08-12 09:30:00 | 322.20 | 2024-08-12 10:15:00 | 323.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-08-30 12:30:00 | 315.00 | 2024-09-02 14:15:00 | 320.60 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-08-30 13:30:00 | 314.95 | 2024-09-02 14:15:00 | 320.60 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-09-02 11:15:00 | 313.70 | 2024-09-02 14:15:00 | 320.60 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-09-06 09:30:00 | 311.65 | 2024-09-10 10:15:00 | 316.60 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-09-09 15:15:00 | 312.70 | 2024-09-10 10:15:00 | 316.60 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-09-10 09:30:00 | 313.15 | 2024-09-10 10:15:00 | 316.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-09-12 10:15:00 | 321.60 | 2024-09-20 13:15:00 | 325.55 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-09-25 14:00:00 | 342.25 | 2024-10-03 14:15:00 | 343.70 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-09-25 14:45:00 | 344.15 | 2024-10-03 14:15:00 | 343.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-09-26 13:45:00 | 342.40 | 2024-10-03 14:15:00 | 343.70 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-09-26 15:00:00 | 342.35 | 2024-10-03 14:15:00 | 343.70 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-09-27 09:15:00 | 356.45 | 2024-10-03 14:15:00 | 343.70 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2024-10-15 14:30:00 | 345.90 | 2024-10-16 15:15:00 | 343.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-10-16 11:45:00 | 346.00 | 2024-10-16 15:15:00 | 343.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-10-29 09:30:00 | 303.60 | 2024-10-30 09:15:00 | 308.55 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-11-07 09:15:00 | 295.85 | 2024-11-07 10:15:00 | 304.45 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-11-07 12:45:00 | 297.40 | 2024-11-08 09:15:00 | 304.35 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-11-07 13:15:00 | 296.70 | 2024-11-08 09:15:00 | 304.35 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-12-02 10:30:00 | 318.10 | 2024-12-04 13:15:00 | 313.15 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-12-03 09:45:00 | 317.80 | 2024-12-04 13:15:00 | 313.15 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-12-09 14:15:00 | 307.40 | 2024-12-10 15:15:00 | 312.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-12-10 12:15:00 | 307.25 | 2024-12-10 15:15:00 | 312.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-12-10 14:30:00 | 307.70 | 2024-12-10 15:15:00 | 312.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-27 12:00:00 | 282.05 | 2025-01-01 13:15:00 | 282.45 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-12-27 14:45:00 | 281.70 | 2025-01-01 13:15:00 | 282.45 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-12-27 15:15:00 | 281.50 | 2025-01-01 13:15:00 | 282.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-01-29 11:15:00 | 267.00 | 2025-01-31 09:15:00 | 272.65 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-01-29 12:15:00 | 267.05 | 2025-01-31 09:15:00 | 272.65 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-01-30 09:15:00 | 266.60 | 2025-01-31 09:15:00 | 272.65 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-01-30 10:45:00 | 266.20 | 2025-01-31 09:15:00 | 272.65 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-01-30 14:30:00 | 264.10 | 2025-01-31 09:15:00 | 272.65 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-02-04 14:45:00 | 286.00 | 2025-02-06 10:15:00 | 280.65 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-02-05 13:45:00 | 286.15 | 2025-02-06 10:15:00 | 280.65 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-02-13 11:30:00 | 253.50 | 2025-02-17 09:15:00 | 240.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 256.10 | 2025-02-17 09:15:00 | 243.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 253.50 | 2025-02-17 10:15:00 | 252.50 | STOP_HIT | 0.50 | 0.39% |
| SELL | retest2 | 2025-02-14 09:15:00 | 256.10 | 2025-02-17 10:15:00 | 252.50 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2025-02-25 11:45:00 | 249.80 | 2025-02-28 09:15:00 | 237.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 12:30:00 | 249.85 | 2025-02-28 09:15:00 | 237.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 249.80 | 2025-03-03 12:15:00 | 239.14 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2025-02-25 12:30:00 | 249.85 | 2025-03-03 12:15:00 | 239.14 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2025-03-11 14:15:00 | 240.71 | 2025-03-12 14:15:00 | 243.73 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-03-12 10:00:00 | 240.43 | 2025-03-12 14:15:00 | 243.73 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-03-12 10:45:00 | 239.64 | 2025-03-12 14:15:00 | 243.73 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-03-26 09:15:00 | 258.22 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-03-26 10:15:00 | 257.91 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-03-27 09:15:00 | 258.12 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-03-28 09:15:00 | 262.79 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-04-01 09:15:00 | 261.40 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-04-01 12:00:00 | 257.70 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-04-01 13:30:00 | 257.85 | 2025-04-04 13:15:00 | 257.65 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-04-22 11:45:00 | 271.95 | 2025-04-25 10:15:00 | 263.40 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-04-23 09:15:00 | 272.35 | 2025-04-25 10:15:00 | 263.40 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-04-23 11:15:00 | 271.90 | 2025-04-25 10:15:00 | 263.40 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-04-23 14:00:00 | 272.10 | 2025-04-25 10:15:00 | 263.40 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-06-11 12:00:00 | 77.42 | 2025-06-13 09:15:00 | 73.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:00:00 | 77.42 | 2025-06-13 15:15:00 | 74.41 | STOP_HIT | 0.50 | 3.89% |
| BUY | retest2 | 2025-06-25 09:15:00 | 74.39 | 2025-06-26 11:15:00 | 73.81 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-04 10:15:00 | 78.38 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-04 12:45:00 | 77.84 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-07-04 13:45:00 | 78.07 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-07-04 14:30:00 | 77.97 | 2025-07-10 12:15:00 | 77.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-14 15:00:00 | 76.56 | 2025-07-15 14:15:00 | 77.39 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-16 14:00:00 | 76.47 | 2025-07-22 15:15:00 | 75.65 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-08-06 13:45:00 | 77.01 | 2025-08-08 09:15:00 | 75.02 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-08-07 15:15:00 | 76.24 | 2025-08-08 09:15:00 | 75.02 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-12 14:15:00 | 74.36 | 2025-08-14 09:15:00 | 76.10 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-08-12 14:45:00 | 74.25 | 2025-08-14 09:15:00 | 76.10 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-09-04 09:15:00 | 84.36 | 2025-09-08 12:15:00 | 92.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-04 14:30:00 | 83.28 | 2025-09-08 12:15:00 | 91.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 09:30:00 | 83.38 | 2025-09-08 12:15:00 | 91.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 13:00:00 | 83.29 | 2025-09-08 12:15:00 | 91.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-11 09:30:00 | 89.59 | 2025-09-12 11:15:00 | 87.57 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-09-11 11:45:00 | 89.00 | 2025-09-12 11:15:00 | 87.57 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-09-11 15:00:00 | 89.28 | 2025-09-12 11:15:00 | 87.57 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-09-15 09:15:00 | 87.94 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-15 10:00:00 | 87.82 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-15 12:00:00 | 87.80 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-16 09:30:00 | 87.99 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-16 11:15:00 | 87.28 | 2025-09-17 09:15:00 | 88.85 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-22 09:45:00 | 93.64 | 2025-09-24 11:15:00 | 91.22 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-09-22 10:45:00 | 93.20 | 2025-09-24 11:15:00 | 91.22 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-09-22 15:00:00 | 93.52 | 2025-09-24 11:15:00 | 91.22 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-10-01 13:45:00 | 84.99 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-08 11:15:00 | 85.37 | 2025-10-14 11:15:00 | 81.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 11:15:00 | 85.37 | 2025-10-15 09:15:00 | 81.32 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2025-11-10 09:15:00 | 79.22 | 2025-11-10 13:15:00 | 80.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-11-10 13:00:00 | 79.39 | 2025-11-10 13:15:00 | 80.20 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-11-10 13:30:00 | 79.36 | 2025-11-10 14:15:00 | 80.04 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-11 09:15:00 | 79.17 | 2025-11-12 10:15:00 | 81.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-11-18 13:30:00 | 77.55 | 2025-11-19 09:15:00 | 78.77 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-11-18 15:15:00 | 77.70 | 2025-11-19 09:15:00 | 78.77 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-21 09:15:00 | 77.41 | 2025-11-26 13:15:00 | 76.94 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-12-04 13:00:00 | 76.32 | 2025-12-05 14:15:00 | 77.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-05 09:15:00 | 76.41 | 2025-12-05 14:15:00 | 77.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-30 09:15:00 | 76.49 | 2025-12-31 11:15:00 | 77.22 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-30 09:45:00 | 76.26 | 2025-12-31 11:15:00 | 77.22 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-12-31 11:00:00 | 76.61 | 2025-12-31 11:15:00 | 77.22 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-01-05 12:15:00 | 77.50 | 2026-01-06 09:15:00 | 76.17 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-05 14:45:00 | 77.32 | 2026-01-06 09:15:00 | 76.17 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-09 11:30:00 | 75.43 | 2026-01-16 14:15:00 | 71.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:30:00 | 75.43 | 2026-01-19 14:15:00 | 71.43 | STOP_HIT | 0.50 | 5.30% |
| BUY | retest2 | 2026-01-29 11:30:00 | 64.02 | 2026-02-02 09:15:00 | 63.94 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-01-29 13:30:00 | 64.29 | 2026-02-02 09:15:00 | 63.94 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-01-30 09:30:00 | 65.19 | 2026-02-02 09:15:00 | 63.94 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-02-03 10:15:00 | 66.71 | 2026-02-03 10:15:00 | 66.45 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-02-09 10:30:00 | 70.90 | 2026-02-11 12:15:00 | 70.07 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-02-09 11:45:00 | 71.99 | 2026-02-11 12:15:00 | 70.07 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-02-24 09:30:00 | 69.10 | 2026-03-02 09:15:00 | 65.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:15:00 | 69.14 | 2026-03-02 09:15:00 | 65.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:45:00 | 69.26 | 2026-03-02 09:15:00 | 65.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 68.82 | 2026-03-02 09:15:00 | 65.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 68.72 | 2026-03-02 09:15:00 | 65.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:30:00 | 69.10 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-02-24 10:15:00 | 69.14 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2026-02-24 10:45:00 | 69.26 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2026-02-25 09:30:00 | 68.82 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-02-26 11:30:00 | 68.72 | 2026-03-02 15:15:00 | 66.20 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2026-03-19 09:15:00 | 58.27 | 2026-03-20 11:15:00 | 62.34 | STOP_HIT | 1.00 | -6.98% |
| SELL | retest2 | 2026-03-24 14:15:00 | 58.42 | 2026-03-24 15:15:00 | 58.20 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2026-04-08 09:15:00 | 59.20 | 2026-04-15 09:15:00 | 65.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 14:45:00 | 58.35 | 2026-04-15 09:15:00 | 64.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 64.37 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-29 10:15:00 | 63.75 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-04-29 10:45:00 | 64.19 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-05-04 11:15:00 | 63.75 | 2026-05-04 11:15:00 | 63.05 | STOP_HIT | 1.00 | -1.10% |
