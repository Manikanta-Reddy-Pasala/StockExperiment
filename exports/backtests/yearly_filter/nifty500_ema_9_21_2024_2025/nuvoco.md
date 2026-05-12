# Nuvoco Vistas Corporation Ltd. (NUVOCO)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 328.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 161 |
| ALERT1 | 97 |
| ALERT2 | 96 |
| ALERT2_SKIP | 55 |
| ALERT3 | 247 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 119 |
| PARTIAL | 13 |
| TARGET_HIT | 5 |
| STOP_HIT | 122 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 102
- **Target hits / Stop hits / Partials:** 5 / 119 / 13
- **Avg / median % per leg:** 0.13% / -0.61%
- **Sum % (uncompounded):** 18.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 12 | 23.1% | 5 | 45 | 2 | 0.17% | 8.8% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.08% | 24.5% |
| BUY @ 3rd Alert (retest2) | 46 | 8 | 17.4% | 3 | 43 | 0 | -0.34% | -15.7% |
| SELL (all) | 85 | 23 | 27.1% | 0 | 74 | 11 | 0.11% | 9.5% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.54% | 1.5% |
| SELL @ 3rd Alert (retest2) | 84 | 22 | 26.2% | 0 | 73 | 11 | 0.10% | 8.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 3 | 2 | 3.72% | 26.0% |
| retest2 (combined) | 130 | 30 | 23.1% | 3 | 116 | 11 | -0.06% | -7.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 318.40 | 315.60 | 315.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 11:15:00 | 321.45 | 318.60 | 317.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 12:15:00 | 322.75 | 323.37 | 321.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:00:00 | 322.75 | 323.37 | 321.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 322.80 | 323.26 | 321.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:45:00 | 322.10 | 323.26 | 321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 323.70 | 323.56 | 322.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 322.55 | 323.56 | 322.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 322.15 | 323.20 | 322.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:30:00 | 322.50 | 323.20 | 322.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 322.00 | 322.96 | 322.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 320.25 | 322.96 | 322.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 319.30 | 321.81 | 322.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 11:15:00 | 318.45 | 321.14 | 321.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 10:15:00 | 320.05 | 319.52 | 320.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 10:15:00 | 320.05 | 319.52 | 320.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 320.05 | 319.52 | 320.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 318.55 | 319.54 | 320.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:15:00 | 318.55 | 316.90 | 317.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 15:00:00 | 318.00 | 317.78 | 318.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 317.70 | 318.13 | 318.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 317.65 | 318.03 | 318.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 319.30 | 318.36 | 318.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 319.30 | 318.36 | 318.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 12:15:00 | 319.95 | 318.68 | 318.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 317.00 | 318.55 | 318.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 317.00 | 318.55 | 318.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 317.00 | 318.55 | 318.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 316.40 | 318.55 | 318.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 15:15:00 | 315.95 | 318.03 | 318.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 313.90 | 317.20 | 317.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 314.00 | 312.62 | 314.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 314.00 | 312.62 | 314.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 314.00 | 312.62 | 314.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 314.00 | 312.62 | 314.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 314.60 | 313.01 | 314.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:15:00 | 314.40 | 313.01 | 314.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 314.40 | 313.29 | 314.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 319.70 | 313.29 | 314.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 323.35 | 315.30 | 315.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 330.90 | 320.97 | 317.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 319.25 | 324.12 | 320.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 319.25 | 324.12 | 320.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 319.25 | 324.12 | 320.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 316.40 | 324.12 | 320.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 303.50 | 320.00 | 319.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 303.50 | 320.00 | 319.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 306.45 | 317.29 | 318.03 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 320.25 | 315.65 | 315.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 15:15:00 | 321.00 | 316.72 | 316.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 357.25 | 361.23 | 355.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 10:00:00 | 357.25 | 361.23 | 355.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 358.00 | 360.59 | 355.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:30:00 | 359.00 | 360.58 | 356.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:30:00 | 360.00 | 359.73 | 356.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 353.40 | 357.76 | 356.42 | SL hit (close<static) qty=1.00 sl=355.20 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 351.75 | 354.94 | 355.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 350.45 | 353.79 | 354.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 12:15:00 | 353.50 | 352.34 | 353.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 12:15:00 | 353.50 | 352.34 | 353.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 353.50 | 352.34 | 353.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 353.50 | 352.34 | 353.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 353.00 | 352.47 | 353.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 15:15:00 | 351.45 | 352.50 | 353.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 352.10 | 351.97 | 352.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 353.65 | 351.50 | 351.81 | SL hit (close>static) qty=1.00 sl=353.60 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 353.40 | 352.17 | 352.04 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 348.70 | 351.37 | 351.69 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 367.00 | 354.04 | 352.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 373.65 | 366.57 | 362.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 10:15:00 | 368.00 | 368.16 | 364.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 363.00 | 366.79 | 364.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 363.00 | 366.79 | 364.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 363.00 | 366.79 | 364.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 361.75 | 365.78 | 364.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:30:00 | 363.00 | 365.78 | 364.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 09:15:00 | 358.65 | 362.92 | 363.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 15:15:00 | 355.00 | 359.40 | 361.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 12:15:00 | 360.40 | 359.00 | 360.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 12:15:00 | 360.40 | 359.00 | 360.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 360.40 | 359.00 | 360.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:45:00 | 360.45 | 359.00 | 360.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 358.10 | 358.82 | 360.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 355.45 | 358.55 | 359.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 14:15:00 | 373.10 | 359.54 | 359.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 373.10 | 359.54 | 359.37 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 360.50 | 361.35 | 361.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 357.00 | 360.48 | 360.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 360.40 | 356.90 | 358.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 360.40 | 356.90 | 358.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 360.40 | 356.90 | 358.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 362.20 | 356.90 | 358.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 360.00 | 357.52 | 358.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:30:00 | 356.70 | 357.46 | 358.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 357.00 | 356.66 | 357.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:00:00 | 356.95 | 356.72 | 357.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 357.10 | 357.34 | 357.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 358.55 | 357.58 | 357.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 358.60 | 357.58 | 357.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 358.10 | 357.69 | 357.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:15:00 | 355.50 | 357.69 | 357.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 10:45:00 | 356.25 | 354.99 | 355.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:30:00 | 355.80 | 355.27 | 355.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:00:00 | 355.00 | 355.27 | 355.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 354.45 | 355.11 | 355.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:30:00 | 354.50 | 355.11 | 355.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 357.25 | 355.54 | 355.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 357.45 | 355.54 | 355.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-16 15:15:00 | 357.15 | 355.86 | 355.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 15:15:00 | 357.15 | 355.86 | 355.85 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 354.00 | 355.49 | 355.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 349.75 | 354.34 | 355.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 340.05 | 339.72 | 343.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:45:00 | 340.25 | 339.72 | 343.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 342.55 | 341.10 | 343.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 340.50 | 341.19 | 343.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 340.70 | 341.45 | 343.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 341.00 | 341.76 | 342.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:30:00 | 341.85 | 341.72 | 342.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 345.25 | 342.43 | 342.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 345.25 | 342.43 | 342.79 | SL hit (close>static) qty=1.00 sl=345.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 345.40 | 343.49 | 343.24 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 341.50 | 343.07 | 343.26 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 345.00 | 343.46 | 343.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 347.00 | 344.89 | 344.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 351.70 | 352.39 | 349.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:00:00 | 351.70 | 352.39 | 349.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 349.30 | 351.77 | 349.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:00:00 | 349.30 | 351.77 | 349.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 355.75 | 352.57 | 349.89 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 348.15 | 351.16 | 351.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 340.05 | 346.54 | 348.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 334.55 | 331.19 | 336.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 334.55 | 331.19 | 336.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 336.45 | 332.24 | 336.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:45:00 | 336.80 | 332.24 | 336.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 333.60 | 332.51 | 335.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 332.80 | 332.51 | 335.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 333.45 | 331.36 | 332.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 332.85 | 332.33 | 333.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 332.20 | 332.37 | 332.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 333.80 | 332.66 | 332.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 333.40 | 332.81 | 332.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 333.40 | 332.81 | 332.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 334.20 | 333.28 | 333.01 | Break + close above crossover candle high |

### Cycle 22 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 329.80 | 332.79 | 332.87 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 334.15 | 333.13 | 333.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 15:15:00 | 336.25 | 334.21 | 333.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 332.70 | 333.91 | 333.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 332.70 | 333.91 | 333.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 332.70 | 333.91 | 333.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 332.70 | 333.91 | 333.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 332.50 | 333.62 | 333.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 332.35 | 333.62 | 333.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 329.75 | 332.85 | 333.06 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 14:15:00 | 335.30 | 333.38 | 333.26 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 332.00 | 333.17 | 333.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 330.20 | 332.26 | 332.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 332.60 | 331.68 | 332.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 332.60 | 331.68 | 332.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 332.60 | 331.68 | 332.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:30:00 | 332.05 | 331.68 | 332.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 332.70 | 331.88 | 332.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 332.35 | 331.88 | 332.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 333.60 | 332.23 | 332.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:45:00 | 334.05 | 332.23 | 332.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 333.15 | 332.41 | 332.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 340.00 | 334.41 | 333.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 337.15 | 337.59 | 335.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 337.15 | 337.59 | 335.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 337.90 | 337.65 | 335.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 341.25 | 337.65 | 335.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 343.85 | 340.27 | 338.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 338.00 | 342.18 | 342.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 338.00 | 342.18 | 342.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 11:15:00 | 337.50 | 338.99 | 340.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 340.70 | 339.34 | 340.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 12:15:00 | 340.70 | 339.34 | 340.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 340.70 | 339.34 | 340.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 340.70 | 339.34 | 340.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 340.30 | 339.53 | 340.22 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 341.00 | 340.47 | 340.47 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 12:15:00 | 337.95 | 339.97 | 340.24 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 342.40 | 340.46 | 340.25 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 338.70 | 340.04 | 340.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 15:15:00 | 338.00 | 339.38 | 339.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 10:15:00 | 340.10 | 339.48 | 339.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 10:15:00 | 340.10 | 339.48 | 339.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 340.10 | 339.48 | 339.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 341.15 | 339.48 | 339.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 338.45 | 339.27 | 339.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 12:45:00 | 337.00 | 338.95 | 339.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 340.85 | 338.92 | 339.19 | SL hit (close>static) qty=1.00 sl=340.50 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 341.00 | 339.55 | 339.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 11:15:00 | 343.65 | 340.37 | 339.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 13:15:00 | 349.25 | 349.81 | 347.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 14:00:00 | 349.25 | 349.81 | 347.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 346.65 | 349.21 | 347.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 345.70 | 349.21 | 347.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 345.65 | 348.50 | 347.65 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 345.00 | 346.84 | 347.02 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 351.85 | 347.54 | 347.26 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 348.00 | 350.23 | 350.27 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 352.45 | 350.31 | 350.16 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 12:15:00 | 350.10 | 351.31 | 351.38 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 352.00 | 351.45 | 351.43 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 350.50 | 351.26 | 351.35 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 374.50 | 355.91 | 353.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 374.95 | 359.72 | 355.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 15:15:00 | 364.15 | 364.65 | 359.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:15:00 | 372.15 | 364.65 | 359.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 360.65 | 367.33 | 364.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 360.65 | 367.33 | 364.63 | SL hit (close<ema400) qty=1.00 sl=364.63 alert=retest1 |

### Cycle 42 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 355.95 | 363.00 | 363.00 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 358.70 | 356.47 | 356.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 11:15:00 | 360.00 | 357.83 | 357.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 356.00 | 357.77 | 357.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 356.00 | 357.77 | 357.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 356.00 | 357.77 | 357.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 356.00 | 357.77 | 357.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 356.50 | 357.52 | 357.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 353.05 | 357.52 | 357.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 357.35 | 357.40 | 357.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 356.05 | 357.40 | 357.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 359.90 | 357.90 | 357.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 357.70 | 357.90 | 357.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 359.40 | 358.20 | 357.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 358.85 | 358.20 | 357.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 358.30 | 358.59 | 358.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 357.35 | 358.59 | 358.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 358.70 | 358.61 | 358.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 358.00 | 358.61 | 358.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 357.05 | 358.30 | 358.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 357.05 | 358.30 | 358.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 357.90 | 358.22 | 357.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 13:30:00 | 360.60 | 358.70 | 358.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 356.00 | 359.00 | 358.56 | SL hit (close<static) qty=1.00 sl=357.05 alert=retest2 |

### Cycle 44 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 357.15 | 358.12 | 358.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 10:15:00 | 356.00 | 357.20 | 357.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 357.60 | 357.28 | 357.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 357.60 | 357.28 | 357.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 357.60 | 357.28 | 357.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 354.55 | 356.56 | 357.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 355.90 | 354.35 | 354.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 355.90 | 354.35 | 354.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 09:15:00 | 361.55 | 357.76 | 356.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 14:15:00 | 358.25 | 358.50 | 357.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 14:15:00 | 358.25 | 358.50 | 357.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 358.25 | 358.50 | 357.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:30:00 | 357.70 | 358.50 | 357.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 354.20 | 358.04 | 357.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 354.20 | 358.04 | 357.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 352.35 | 356.90 | 356.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 15:15:00 | 351.55 | 353.75 | 355.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 12:15:00 | 359.95 | 351.01 | 352.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 12:15:00 | 359.95 | 351.01 | 352.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 359.95 | 351.01 | 352.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:45:00 | 360.05 | 351.01 | 352.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 13:15:00 | 366.20 | 354.05 | 353.29 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 352.40 | 357.06 | 357.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 348.00 | 354.58 | 356.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 12:15:00 | 349.50 | 347.36 | 349.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 12:15:00 | 349.50 | 347.36 | 349.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 349.50 | 347.36 | 349.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:00:00 | 349.50 | 347.36 | 349.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 343.35 | 346.56 | 349.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 14:15:00 | 340.00 | 346.56 | 349.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 336.40 | 343.89 | 347.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 323.00 | 339.83 | 343.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 341.75 | 338.82 | 341.20 | SL hit (close>ema200) qty=0.50 sl=338.82 alert=retest2 |

### Cycle 49 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 340.40 | 335.43 | 335.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 345.45 | 337.43 | 336.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 344.25 | 344.54 | 341.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 344.25 | 344.54 | 341.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 354.00 | 346.09 | 342.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 360.95 | 353.11 | 348.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:45:00 | 358.95 | 356.50 | 352.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 12:15:00 | 352.50 | 354.02 | 354.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 352.50 | 354.02 | 354.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 351.65 | 353.41 | 353.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 352.00 | 351.03 | 352.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 14:15:00 | 352.00 | 351.03 | 352.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 352.00 | 351.03 | 352.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 352.00 | 351.03 | 352.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 350.00 | 350.83 | 352.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 346.65 | 350.83 | 352.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 11:00:00 | 346.95 | 349.44 | 351.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 329.32 | 334.14 | 339.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 329.60 | 334.14 | 339.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 335.60 | 333.46 | 338.05 | SL hit (close>ema200) qty=0.50 sl=333.46 alert=retest2 |

### Cycle 51 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 341.85 | 336.62 | 336.23 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 330.85 | 335.71 | 336.25 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 339.70 | 336.54 | 336.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 344.65 | 338.16 | 337.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 14:15:00 | 338.70 | 338.98 | 337.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 14:15:00 | 338.70 | 338.98 | 337.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 338.70 | 338.98 | 337.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 338.70 | 338.98 | 337.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 336.40 | 338.46 | 337.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 340.25 | 338.46 | 337.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 339.20 | 340.46 | 340.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-10 09:15:00 | 373.12 | 369.09 | 368.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 366.50 | 369.96 | 370.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 362.85 | 367.33 | 368.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 366.05 | 365.26 | 367.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 366.05 | 365.26 | 367.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 366.05 | 365.26 | 367.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 366.05 | 365.26 | 367.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 366.80 | 365.57 | 367.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 365.55 | 366.06 | 366.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 347.27 | 350.64 | 351.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 14:15:00 | 351.00 | 349.83 | 350.64 | SL hit (close>ema200) qty=0.50 sl=349.83 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 353.45 | 351.07 | 351.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 354.90 | 352.63 | 351.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 356.30 | 356.42 | 354.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 356.30 | 356.42 | 354.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 353.95 | 355.93 | 354.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 353.95 | 355.93 | 354.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 352.95 | 355.33 | 354.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 352.95 | 355.33 | 354.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 352.20 | 354.55 | 354.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 352.20 | 354.55 | 354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 353.50 | 354.34 | 354.35 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 12:15:00 | 355.55 | 354.58 | 354.46 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 352.35 | 354.14 | 354.27 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 354.95 | 354.32 | 354.24 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 13:15:00 | 352.55 | 353.94 | 354.09 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 355.45 | 354.24 | 354.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 10:15:00 | 356.50 | 355.02 | 354.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 14:15:00 | 354.85 | 355.09 | 354.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 354.85 | 355.09 | 354.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 354.85 | 355.09 | 354.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:30:00 | 355.05 | 355.09 | 354.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 356.00 | 355.27 | 354.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 354.50 | 355.27 | 354.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 354.80 | 355.18 | 354.88 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 354.20 | 354.71 | 354.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 13:15:00 | 353.05 | 354.37 | 354.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 15:15:00 | 346.00 | 345.66 | 347.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 09:15:00 | 345.95 | 343.17 | 344.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 345.95 | 343.17 | 344.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:30:00 | 345.85 | 343.17 | 344.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 350.35 | 344.61 | 345.46 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 350.00 | 346.63 | 346.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 352.20 | 347.77 | 346.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 11:15:00 | 346.45 | 347.62 | 347.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 11:15:00 | 346.45 | 347.62 | 347.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 346.45 | 347.62 | 347.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:00:00 | 346.45 | 347.62 | 347.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 347.40 | 347.58 | 347.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 15:00:00 | 348.00 | 347.64 | 347.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 13:15:00 | 340.45 | 348.96 | 350.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 340.45 | 348.96 | 350.03 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 09:15:00 | 349.25 | 348.59 | 348.54 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 344.45 | 348.72 | 349.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 15:15:00 | 342.85 | 347.00 | 348.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 339.45 | 337.48 | 341.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 339.45 | 337.48 | 341.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 339.45 | 337.48 | 341.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 340.10 | 337.48 | 341.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 343.40 | 338.67 | 341.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 343.40 | 338.67 | 341.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 344.00 | 339.73 | 341.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 344.00 | 339.73 | 341.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 347.20 | 343.27 | 342.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 350.45 | 346.92 | 345.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 348.10 | 349.06 | 347.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 348.10 | 349.06 | 347.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 348.10 | 349.06 | 347.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:45:00 | 347.70 | 349.06 | 347.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 347.30 | 348.71 | 347.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 347.30 | 348.71 | 347.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 345.50 | 348.07 | 347.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 344.00 | 348.07 | 347.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 341.55 | 346.76 | 346.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 341.55 | 346.76 | 346.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 344.15 | 346.24 | 346.48 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 353.00 | 347.67 | 346.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 354.00 | 352.23 | 350.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 359.30 | 361.09 | 357.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:00:00 | 359.30 | 361.09 | 357.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 360.00 | 360.87 | 358.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 360.50 | 360.87 | 358.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 351.20 | 357.91 | 357.74 | SL hit (close<static) qty=1.00 sl=356.55 alert=retest2 |

### Cycle 70 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 350.05 | 356.33 | 357.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 349.05 | 354.88 | 356.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 317.15 | 314.80 | 320.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 317.15 | 314.80 | 320.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 319.25 | 316.26 | 320.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 319.25 | 316.26 | 320.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 318.30 | 316.67 | 319.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 314.55 | 316.67 | 319.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:45:00 | 318.20 | 314.75 | 316.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 13:15:00 | 322.50 | 318.17 | 317.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 322.50 | 318.17 | 317.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 322.95 | 319.13 | 318.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 321.15 | 321.25 | 319.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 321.15 | 321.25 | 319.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 321.40 | 321.30 | 320.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:45:00 | 320.25 | 321.30 | 320.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 321.00 | 321.57 | 320.96 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 315.00 | 320.26 | 320.41 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 13:15:00 | 320.85 | 320.57 | 320.53 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 316.85 | 319.82 | 320.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 12:15:00 | 310.85 | 316.55 | 318.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 306.00 | 304.70 | 308.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 306.00 | 304.70 | 308.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 306.00 | 304.70 | 308.54 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 312.00 | 308.38 | 308.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 316.75 | 310.64 | 309.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 316.60 | 316.62 | 314.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:00:00 | 316.60 | 316.62 | 314.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 318.50 | 318.68 | 316.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 315.15 | 318.68 | 316.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 314.80 | 317.90 | 316.37 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 311.45 | 314.86 | 315.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 307.65 | 313.42 | 314.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 296.00 | 295.40 | 298.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-17 10:00:00 | 296.00 | 295.40 | 298.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 296.00 | 292.17 | 295.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 296.00 | 292.17 | 295.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 294.65 | 292.67 | 294.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 294.95 | 292.67 | 294.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 294.80 | 293.09 | 294.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:45:00 | 295.00 | 293.09 | 294.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 295.75 | 293.62 | 295.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 295.75 | 293.62 | 295.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 296.55 | 294.21 | 295.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:00:00 | 296.55 | 294.21 | 295.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 297.10 | 294.69 | 295.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 302.90 | 294.69 | 295.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 302.50 | 296.25 | 295.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 307.10 | 298.42 | 296.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 302.50 | 302.72 | 300.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 305.15 | 302.72 | 300.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:30:00 | 303.85 | 302.92 | 300.59 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:15:00 | 320.41 | 309.12 | 304.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:15:00 | 319.04 | 309.12 | 304.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-03-21 10:15:00 | 335.67 | 314.07 | 307.28 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 78 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 315.50 | 318.52 | 318.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 308.75 | 316.16 | 317.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 318.05 | 314.56 | 316.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 318.05 | 314.56 | 316.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 318.05 | 314.56 | 316.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 318.05 | 314.56 | 316.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 309.50 | 313.55 | 315.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 324.95 | 313.55 | 315.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 306.00 | 312.04 | 314.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 307.90 | 312.04 | 314.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 308.80 | 311.39 | 314.15 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 313.50 | 312.33 | 312.30 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 13:15:00 | 309.55 | 312.30 | 312.41 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 313.95 | 312.54 | 312.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 317.55 | 313.55 | 312.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 314.10 | 315.11 | 313.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 314.10 | 315.11 | 313.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 314.10 | 315.11 | 313.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:15:00 | 312.05 | 315.11 | 313.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 315.95 | 315.28 | 314.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 314.30 | 315.28 | 314.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 313.20 | 314.86 | 313.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 313.20 | 314.86 | 313.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 315.40 | 314.97 | 314.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:30:00 | 313.45 | 314.97 | 314.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 316.85 | 315.35 | 314.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:15:00 | 317.05 | 315.35 | 314.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 317.85 | 315.72 | 314.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 298.85 | 312.38 | 313.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 298.85 | 312.38 | 313.29 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 13:15:00 | 312.80 | 307.65 | 306.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 314.35 | 308.99 | 307.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 312.60 | 313.35 | 310.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 13:15:00 | 312.60 | 313.35 | 310.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 312.60 | 313.35 | 310.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:00:00 | 312.60 | 313.35 | 310.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 318.70 | 320.55 | 318.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 324.00 | 321.19 | 319.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 331.15 | 333.28 | 333.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 331.15 | 333.28 | 333.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 10:15:00 | 329.15 | 332.45 | 333.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 343.75 | 326.74 | 327.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 343.75 | 326.74 | 327.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 343.75 | 326.74 | 327.73 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 341.95 | 329.78 | 329.02 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 336.00 | 341.10 | 341.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 334.40 | 339.76 | 341.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 342.50 | 335.10 | 337.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 342.50 | 335.10 | 337.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 342.50 | 335.10 | 337.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 343.00 | 335.10 | 337.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 341.45 | 336.37 | 337.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:45:00 | 339.70 | 337.10 | 337.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 15:15:00 | 341.75 | 338.87 | 338.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 341.75 | 338.87 | 338.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 342.70 | 339.64 | 338.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 338.35 | 340.11 | 339.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 338.35 | 340.11 | 339.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 338.35 | 340.11 | 339.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 338.35 | 340.11 | 339.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 339.50 | 339.99 | 339.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 15:00:00 | 344.45 | 340.88 | 339.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 357.10 | 358.19 | 358.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 12:15:00 | 357.10 | 358.19 | 358.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 356.40 | 357.83 | 358.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 15:15:00 | 356.35 | 356.01 | 356.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:15:00 | 355.85 | 356.01 | 356.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 355.35 | 355.88 | 356.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 10:15:00 | 354.00 | 355.88 | 356.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 357.35 | 352.96 | 353.04 | SL hit (close>static) qty=1.00 sl=357.15 alert=retest2 |

### Cycle 89 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 357.95 | 353.96 | 353.49 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 353.80 | 354.43 | 354.49 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 15:15:00 | 355.45 | 354.64 | 354.58 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 353.95 | 354.50 | 354.52 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 357.00 | 355.00 | 354.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 361.15 | 356.70 | 355.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 10:15:00 | 356.30 | 356.62 | 355.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 11:00:00 | 356.30 | 356.62 | 355.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 357.15 | 356.72 | 355.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 359.95 | 357.32 | 356.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 15:00:00 | 358.95 | 358.05 | 356.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 353.90 | 357.53 | 357.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 353.90 | 357.53 | 357.83 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 359.75 | 357.85 | 357.69 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 356.95 | 358.16 | 358.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 354.00 | 357.33 | 357.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 355.00 | 354.94 | 356.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:45:00 | 355.20 | 354.94 | 356.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 354.80 | 351.79 | 352.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 354.80 | 351.79 | 352.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 350.15 | 351.46 | 352.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 347.15 | 350.42 | 351.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 347.00 | 349.15 | 350.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 346.75 | 347.59 | 349.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 347.40 | 342.45 | 341.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 347.40 | 342.45 | 341.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 347.80 | 343.52 | 342.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 353.20 | 354.22 | 352.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 353.20 | 354.22 | 352.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 352.45 | 353.86 | 352.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 352.10 | 353.86 | 352.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 355.20 | 354.13 | 352.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 357.30 | 354.16 | 352.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:45:00 | 356.15 | 354.74 | 353.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 356.35 | 358.75 | 358.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 355.15 | 358.03 | 358.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 355.15 | 358.03 | 358.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 352.80 | 356.98 | 357.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 357.75 | 357.14 | 357.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 357.75 | 357.14 | 357.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 357.75 | 357.14 | 357.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 357.75 | 357.14 | 357.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 358.00 | 357.31 | 357.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:45:00 | 358.30 | 357.31 | 357.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 358.20 | 357.49 | 357.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 358.20 | 357.49 | 357.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 356.95 | 357.38 | 357.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 359.90 | 357.38 | 357.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 357.60 | 357.42 | 357.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 358.70 | 357.42 | 357.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 356.40 | 357.22 | 357.62 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 359.25 | 357.99 | 357.87 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 355.25 | 357.63 | 357.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 354.95 | 357.09 | 357.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 358.65 | 356.77 | 357.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 358.65 | 356.77 | 357.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 358.65 | 356.77 | 357.31 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 12:15:00 | 359.10 | 357.87 | 357.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 362.10 | 358.90 | 358.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 362.00 | 362.13 | 360.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 362.00 | 362.13 | 360.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 362.00 | 362.13 | 360.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 362.15 | 362.13 | 360.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 360.75 | 361.86 | 360.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 360.75 | 361.86 | 360.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 361.00 | 361.69 | 360.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 361.90 | 362.29 | 360.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 359.00 | 361.92 | 361.05 | SL hit (close<static) qty=1.00 sl=360.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 402.50 | 412.00 | 412.11 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 416.45 | 412.89 | 412.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 426.35 | 419.20 | 416.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 426.95 | 429.55 | 424.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:15:00 | 432.80 | 429.55 | 424.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 434.75 | 430.59 | 425.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:30:00 | 438.70 | 430.92 | 428.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 438.00 | 431.65 | 429.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:00:00 | 438.60 | 433.04 | 430.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 439.50 | 435.67 | 433.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 433.00 | 435.14 | 433.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 433.90 | 435.14 | 433.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 434.90 | 435.09 | 433.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 433.50 | 435.09 | 433.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 434.60 | 434.96 | 433.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:15:00 | 433.90 | 434.96 | 433.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 433.90 | 434.75 | 433.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 433.00 | 434.75 | 433.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 432.95 | 434.39 | 433.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-07 11:15:00 | 430.00 | 432.71 | 433.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 430.00 | 432.71 | 433.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 13:15:00 | 429.10 | 431.50 | 432.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 436.15 | 432.16 | 432.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 436.15 | 432.16 | 432.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 436.15 | 432.16 | 432.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:30:00 | 437.50 | 432.16 | 432.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 435.45 | 432.82 | 432.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 13:15:00 | 438.80 | 434.38 | 433.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 448.20 | 449.05 | 445.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:15:00 | 460.25 | 449.05 | 445.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 451.40 | 453.06 | 449.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 451.40 | 453.06 | 449.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 449.00 | 452.25 | 449.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 449.00 | 452.25 | 449.83 | SL hit (close<ema400) qty=1.00 sl=449.83 alert=retest1 |

### Cycle 106 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 455.25 | 460.17 | 460.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 453.40 | 458.27 | 459.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 464.65 | 458.62 | 459.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 464.65 | 458.62 | 459.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 464.65 | 458.62 | 459.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 465.45 | 458.62 | 459.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 464.95 | 459.89 | 459.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 464.95 | 459.89 | 459.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 464.95 | 460.90 | 460.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 09:15:00 | 466.45 | 463.90 | 462.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 10:15:00 | 463.55 | 463.83 | 462.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 463.55 | 463.83 | 462.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 463.55 | 463.83 | 462.37 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 456.95 | 461.40 | 461.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 13:15:00 | 456.20 | 460.36 | 461.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 464.95 | 460.17 | 461.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 464.95 | 460.17 | 461.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 464.95 | 460.17 | 461.00 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 461.25 | 460.24 | 460.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 12:15:00 | 462.00 | 460.59 | 460.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 467.55 | 468.58 | 465.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 467.55 | 468.58 | 465.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 467.55 | 468.58 | 465.98 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 461.55 | 464.63 | 464.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 458.50 | 462.76 | 463.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 434.50 | 434.21 | 440.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:00:00 | 434.50 | 434.21 | 440.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 435.60 | 431.96 | 434.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 435.60 | 431.96 | 434.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 434.60 | 432.49 | 434.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 439.45 | 432.49 | 434.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 435.95 | 433.18 | 434.54 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 440.45 | 435.99 | 435.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 442.50 | 438.81 | 437.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 15:15:00 | 447.70 | 447.84 | 446.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 15:15:00 | 447.70 | 447.84 | 446.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 447.70 | 447.84 | 446.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 452.00 | 447.84 | 446.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:30:00 | 448.65 | 448.92 | 447.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:00:00 | 449.70 | 448.92 | 447.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:30:00 | 449.25 | 448.79 | 447.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 447.05 | 448.25 | 447.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 447.05 | 448.25 | 447.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 448.10 | 448.22 | 447.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 448.35 | 448.22 | 447.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 448.75 | 448.33 | 447.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 445.05 | 446.89 | 447.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 445.05 | 446.89 | 447.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 442.45 | 446.22 | 446.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 447.20 | 446.06 | 446.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 447.20 | 446.06 | 446.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 447.20 | 446.06 | 446.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:15:00 | 449.50 | 446.06 | 446.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 448.90 | 446.62 | 446.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 448.70 | 446.62 | 446.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 451.00 | 447.50 | 447.13 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 440.50 | 446.95 | 447.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 431.05 | 442.98 | 445.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 431.85 | 427.61 | 433.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 431.85 | 427.61 | 433.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 431.00 | 428.29 | 432.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 416.65 | 426.92 | 431.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 427.40 | 422.94 | 422.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 427.40 | 422.94 | 422.75 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 422.00 | 422.66 | 422.73 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 425.35 | 423.20 | 422.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 430.40 | 425.53 | 424.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 427.10 | 427.34 | 425.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 425.85 | 427.34 | 425.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 427.80 | 427.43 | 425.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:30:00 | 431.35 | 427.54 | 425.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:45:00 | 431.20 | 428.27 | 426.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 430.80 | 428.27 | 426.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 430.85 | 428.63 | 427.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 425.35 | 428.33 | 427.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 425.35 | 428.33 | 427.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 428.05 | 428.27 | 427.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 432.00 | 428.70 | 427.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 430.40 | 432.35 | 431.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 14:15:00 | 429.60 | 430.61 | 430.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 429.60 | 430.61 | 430.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 422.85 | 428.96 | 429.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 12:15:00 | 426.45 | 425.93 | 428.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 426.45 | 425.93 | 428.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 426.45 | 425.93 | 428.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 426.95 | 425.93 | 428.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 428.30 | 426.58 | 427.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 428.30 | 426.58 | 427.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 425.20 | 426.30 | 427.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 429.40 | 426.60 | 427.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 425.80 | 426.44 | 427.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 428.65 | 426.44 | 427.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 424.85 | 426.12 | 427.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:45:00 | 413.30 | 423.89 | 426.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 418.65 | 414.96 | 414.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 418.65 | 414.96 | 414.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 423.00 | 416.57 | 415.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 421.65 | 422.07 | 419.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:15:00 | 417.30 | 422.07 | 419.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 417.85 | 421.23 | 419.65 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 414.55 | 418.15 | 418.59 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 426.05 | 419.09 | 418.90 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 417.05 | 420.92 | 421.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 13:15:00 | 415.50 | 418.95 | 420.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 423.50 | 419.06 | 419.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 423.50 | 419.06 | 419.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 423.50 | 419.06 | 419.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 422.55 | 419.06 | 419.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 424.95 | 420.24 | 420.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 424.80 | 420.24 | 420.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 424.35 | 421.06 | 420.63 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 414.30 | 419.54 | 420.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 414.20 | 418.47 | 419.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 382.70 | 381.87 | 388.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 09:15:00 | 375.80 | 381.87 | 388.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 375.30 | 375.69 | 378.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 368.15 | 371.25 | 373.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:30:00 | 368.40 | 370.35 | 372.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 15:15:00 | 367.70 | 369.32 | 371.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 368.15 | 368.99 | 370.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 370.00 | 368.22 | 369.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 15:15:00 | 370.00 | 368.22 | 369.70 | SL hit (close>ema400) qty=1.00 sl=369.70 alert=retest1 |

### Cycle 125 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 378.45 | 362.13 | 361.82 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 14:15:00 | 354.50 | 361.39 | 361.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 15:15:00 | 350.05 | 359.12 | 360.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 353.30 | 350.26 | 353.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 353.30 | 350.26 | 353.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 353.30 | 350.26 | 353.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 353.30 | 350.26 | 353.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 361.20 | 352.45 | 353.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 361.20 | 352.45 | 353.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 15:15:00 | 364.65 | 354.89 | 354.82 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 352.95 | 357.09 | 357.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 350.45 | 353.15 | 354.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 352.30 | 351.73 | 353.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 15:00:00 | 352.30 | 351.73 | 353.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 349.85 | 349.80 | 351.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 349.85 | 349.80 | 351.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 349.30 | 349.70 | 351.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 348.00 | 349.05 | 350.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 09:15:00 | 330.60 | 336.81 | 340.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 336.15 | 335.93 | 339.22 | SL hit (close>ema200) qty=0.50 sl=335.93 alert=retest2 |

### Cycle 129 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 343.85 | 340.07 | 339.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 347.00 | 343.44 | 341.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 356.00 | 358.06 | 355.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 356.00 | 358.06 | 355.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 356.00 | 358.06 | 355.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 354.45 | 358.06 | 355.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 354.45 | 357.34 | 355.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 353.95 | 357.34 | 355.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 355.90 | 357.05 | 355.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 353.85 | 357.05 | 355.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 355.05 | 356.13 | 355.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 354.20 | 356.13 | 355.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 352.20 | 355.34 | 354.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 352.20 | 355.34 | 354.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 352.25 | 354.72 | 354.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 352.35 | 354.72 | 354.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 11:15:00 | 354.30 | 354.64 | 354.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 350.05 | 352.95 | 353.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 352.85 | 352.83 | 353.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:30:00 | 352.95 | 352.83 | 353.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 356.40 | 353.55 | 353.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 356.40 | 353.55 | 353.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 360.65 | 354.97 | 354.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 366.10 | 361.26 | 358.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 362.70 | 362.75 | 360.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 362.70 | 362.75 | 360.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 364.25 | 363.04 | 360.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 366.70 | 363.04 | 360.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 371.00 | 365.46 | 363.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 366.00 | 367.27 | 366.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 361.55 | 364.87 | 365.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 361.55 | 364.87 | 365.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 355.75 | 362.64 | 364.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 360.95 | 360.53 | 362.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 360.95 | 360.53 | 362.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 354.85 | 356.33 | 358.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 353.35 | 356.33 | 358.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 352.70 | 355.23 | 357.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 15:15:00 | 358.00 | 356.04 | 355.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 358.00 | 356.04 | 355.86 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 353.40 | 355.32 | 355.56 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 358.10 | 356.08 | 355.87 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 354.60 | 355.60 | 355.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 352.80 | 354.81 | 355.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 353.20 | 352.99 | 354.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 353.20 | 352.99 | 354.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 354.00 | 353.20 | 354.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 350.40 | 353.20 | 354.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 353.70 | 349.71 | 349.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 353.70 | 349.71 | 349.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 355.95 | 353.29 | 351.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 354.00 | 354.35 | 352.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 352.00 | 353.88 | 352.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 352.00 | 353.88 | 352.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 352.00 | 353.88 | 352.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 352.60 | 353.63 | 352.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 352.60 | 353.63 | 352.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 351.40 | 353.18 | 352.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 351.40 | 353.18 | 352.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 351.05 | 352.76 | 352.59 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 348.60 | 351.92 | 352.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 346.25 | 350.45 | 351.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 12:15:00 | 348.90 | 348.59 | 350.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 13:00:00 | 348.90 | 348.59 | 350.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 347.25 | 348.29 | 349.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 349.15 | 348.29 | 349.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 348.45 | 348.11 | 349.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 349.80 | 348.11 | 349.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 349.50 | 348.39 | 349.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 349.50 | 348.39 | 349.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 349.85 | 348.68 | 349.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:30:00 | 349.35 | 348.68 | 349.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 347.95 | 348.54 | 349.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 345.55 | 348.12 | 349.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:00:00 | 346.45 | 348.12 | 349.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 15:15:00 | 345.00 | 348.10 | 348.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 350.00 | 346.44 | 347.09 | SL hit (close>static) qty=1.00 sl=349.95 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 353.55 | 347.86 | 347.68 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 344.60 | 348.25 | 348.65 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 349.75 | 347.26 | 347.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 351.15 | 348.51 | 347.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 348.25 | 349.79 | 348.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 348.25 | 349.79 | 348.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 348.25 | 349.79 | 348.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 348.15 | 349.79 | 348.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 349.05 | 349.64 | 348.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 348.60 | 349.64 | 348.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 350.10 | 349.74 | 349.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 350.50 | 350.11 | 349.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 348.25 | 350.02 | 349.38 | SL hit (close<static) qty=1.00 sl=348.60 alert=retest2 |

### Cycle 142 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 346.70 | 348.88 | 348.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 344.00 | 347.41 | 348.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 337.20 | 336.93 | 340.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 337.20 | 336.93 | 340.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 338.00 | 337.83 | 340.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 339.45 | 337.83 | 340.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 336.45 | 337.56 | 339.83 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 347.60 | 342.30 | 341.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 348.95 | 343.63 | 342.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 343.45 | 343.92 | 342.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 343.45 | 343.92 | 342.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 344.40 | 344.02 | 342.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 344.10 | 344.02 | 342.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 344.45 | 345.29 | 344.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 344.45 | 345.29 | 344.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 347.50 | 345.73 | 344.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 344.50 | 345.73 | 344.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 343.05 | 346.70 | 345.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 343.05 | 346.70 | 345.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 343.80 | 346.12 | 345.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 342.15 | 346.12 | 345.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 342.95 | 344.99 | 345.02 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 345.55 | 345.13 | 345.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 348.60 | 345.84 | 345.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 347.45 | 348.33 | 347.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 15:15:00 | 347.45 | 348.33 | 347.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 347.45 | 348.33 | 347.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 345.85 | 348.33 | 347.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 342.60 | 347.18 | 347.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 342.60 | 347.18 | 347.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 348.50 | 347.45 | 347.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 350.45 | 347.45 | 347.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 348.85 | 351.20 | 350.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:15:00 | 349.50 | 350.66 | 350.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 15:15:00 | 347.00 | 349.33 | 349.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 347.00 | 349.33 | 349.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 341.15 | 347.69 | 348.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 13:15:00 | 336.50 | 336.42 | 338.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 13:30:00 | 337.10 | 336.42 | 338.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 338.50 | 337.07 | 338.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 340.45 | 337.07 | 338.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 339.45 | 337.55 | 338.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:15:00 | 340.65 | 337.55 | 338.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 341.50 | 338.34 | 339.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 341.50 | 338.34 | 339.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 345.10 | 340.52 | 339.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 347.60 | 343.42 | 341.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 342.55 | 343.24 | 341.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:15:00 | 342.50 | 343.24 | 341.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 340.20 | 342.63 | 341.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 340.20 | 342.63 | 341.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 338.00 | 341.71 | 341.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 338.00 | 341.71 | 341.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 337.15 | 340.80 | 340.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 335.00 | 338.83 | 339.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 335.05 | 334.54 | 336.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 335.05 | 334.54 | 336.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 335.05 | 334.54 | 336.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 330.45 | 333.87 | 335.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 331.20 | 333.87 | 335.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 331.75 | 333.92 | 334.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 331.00 | 333.18 | 334.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 334.20 | 333.04 | 333.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 334.70 | 333.04 | 333.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 334.10 | 333.25 | 333.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 334.80 | 333.25 | 333.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 332.70 | 333.14 | 333.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 331.55 | 332.94 | 333.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 331.35 | 332.57 | 333.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 313.93 | 317.47 | 322.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 314.64 | 317.47 | 322.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 315.16 | 317.47 | 322.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 314.45 | 317.47 | 322.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 314.97 | 317.47 | 322.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 314.78 | 317.47 | 322.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 308.80 | 308.69 | 313.69 | SL hit (close>ema200) qty=0.50 sl=308.69 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 306.15 | 296.04 | 294.80 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 289.15 | 293.88 | 294.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 286.50 | 292.41 | 293.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 292.05 | 291.60 | 292.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:00:00 | 292.05 | 291.60 | 292.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 284.15 | 290.26 | 292.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 282.35 | 290.26 | 292.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 294.00 | 289.69 | 289.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 294.00 | 289.69 | 289.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 302.80 | 292.31 | 290.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 294.70 | 300.31 | 296.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 294.70 | 300.31 | 296.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 294.70 | 300.31 | 296.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 294.70 | 300.31 | 296.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 292.85 | 298.82 | 296.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:45:00 | 292.80 | 298.82 | 296.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 287.95 | 294.55 | 294.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 285.35 | 292.71 | 294.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 285.10 | 282.06 | 284.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 285.10 | 282.06 | 284.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 285.10 | 282.06 | 284.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 285.10 | 282.06 | 284.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 287.20 | 283.09 | 284.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 287.20 | 283.09 | 284.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 290.10 | 284.49 | 285.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 290.00 | 284.49 | 285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 305.70 | 289.65 | 287.55 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 289.00 | 294.26 | 294.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 285.65 | 292.54 | 293.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 288.40 | 287.42 | 290.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 288.40 | 287.42 | 290.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 288.40 | 287.42 | 290.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 288.45 | 287.42 | 290.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 291.65 | 288.04 | 290.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 291.65 | 288.04 | 290.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 290.00 | 288.43 | 290.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:30:00 | 291.55 | 288.43 | 290.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 289.55 | 288.63 | 290.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 289.55 | 288.63 | 290.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 290.00 | 288.91 | 290.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 282.75 | 288.91 | 290.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 294.55 | 288.66 | 289.04 | SL hit (close>static) qty=1.00 sl=292.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 292.00 | 289.65 | 289.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 295.60 | 290.84 | 289.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 292.20 | 292.83 | 291.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 292.20 | 292.83 | 291.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 291.95 | 292.95 | 291.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 292.10 | 292.95 | 291.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 290.15 | 292.39 | 291.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:00:00 | 290.15 | 292.39 | 291.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 292.20 | 292.35 | 291.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 293.90 | 292.35 | 291.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 300.60 | 304.83 | 305.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 10:15:00 | 300.60 | 304.83 | 305.20 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 311.70 | 306.01 | 305.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 312.40 | 310.66 | 309.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 308.35 | 310.22 | 309.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 308.35 | 310.22 | 309.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 308.35 | 310.22 | 309.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:45:00 | 308.25 | 310.22 | 309.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 308.30 | 309.83 | 309.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:45:00 | 308.35 | 309.83 | 309.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 307.85 | 309.44 | 309.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 305.20 | 309.44 | 309.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 306.35 | 308.82 | 308.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 297.00 | 304.10 | 306.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 301.90 | 301.81 | 304.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 14:00:00 | 301.90 | 301.81 | 304.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 295.75 | 294.53 | 296.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 292.45 | 294.98 | 296.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 293.75 | 294.76 | 296.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 14:15:00 | 297.30 | 293.20 | 292.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 297.30 | 293.20 | 292.75 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 291.65 | 292.48 | 292.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 290.10 | 292.00 | 292.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 09:15:00 | 291.80 | 291.13 | 291.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 291.80 | 291.13 | 291.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 291.80 | 291.13 | 291.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:45:00 | 292.65 | 291.13 | 291.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 295.35 | 291.98 | 292.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 295.35 | 291.98 | 292.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 296.40 | 292.86 | 292.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 305.65 | 296.68 | 294.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 327.00 | 327.83 | 317.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 11:00:00 | 327.00 | 327.83 | 317.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-27 09:15:00 | 318.55 | 2024-05-29 11:15:00 | 319.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-05-28 12:15:00 | 318.55 | 2024-05-29 11:15:00 | 319.30 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2024-05-28 15:00:00 | 318.00 | 2024-05-29 11:15:00 | 319.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-05-29 09:15:00 | 317.70 | 2024-05-29 11:15:00 | 319.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-06-18 11:30:00 | 359.00 | 2024-06-19 09:15:00 | 353.40 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-06-18 13:30:00 | 360.00 | 2024-06-19 09:15:00 | 353.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-06-20 15:15:00 | 351.45 | 2024-06-24 11:15:00 | 353.65 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-06-21 09:45:00 | 352.10 | 2024-06-24 11:15:00 | 353.65 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-06-24 13:00:00 | 352.05 | 2024-06-25 10:15:00 | 353.40 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-06-24 13:45:00 | 351.65 | 2024-06-25 10:15:00 | 353.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-07-05 09:15:00 | 355.45 | 2024-07-05 14:15:00 | 373.10 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2024-07-11 12:30:00 | 356.70 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-07-12 09:30:00 | 357.00 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-07-12 11:00:00 | 356.95 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-07-12 12:30:00 | 357.10 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-07-12 15:15:00 | 355.50 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-07-16 10:45:00 | 356.25 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-07-16 12:30:00 | 355.80 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-07-16 13:00:00 | 355.00 | 2024-07-16 15:15:00 | 357.15 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-07-23 09:30:00 | 340.50 | 2024-07-24 10:15:00 | 345.25 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-07-23 11:15:00 | 340.70 | 2024-07-24 10:15:00 | 345.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-23 15:15:00 | 341.00 | 2024-07-24 10:15:00 | 345.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-07-24 09:30:00 | 341.85 | 2024-07-24 10:15:00 | 345.25 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-08-06 12:15:00 | 332.80 | 2024-08-09 11:15:00 | 333.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-08-07 14:15:00 | 333.45 | 2024-08-09 11:15:00 | 333.40 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-08-08 09:15:00 | 332.85 | 2024-08-09 11:15:00 | 333.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-08-09 10:15:00 | 332.20 | 2024-08-09 11:15:00 | 333.40 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-08-21 09:15:00 | 341.25 | 2024-08-26 09:15:00 | 338.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-08-22 09:15:00 | 343.85 | 2024-08-26 09:15:00 | 338.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-09-02 12:45:00 | 337.00 | 2024-09-03 09:15:00 | 340.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2024-09-18 09:15:00 | 372.15 | 2024-09-19 09:15:00 | 360.65 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2024-10-01 13:30:00 | 360.60 | 2024-10-03 09:15:00 | 356.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-04 12:30:00 | 354.55 | 2024-10-09 10:15:00 | 355.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-10-22 14:15:00 | 340.00 | 2024-10-24 09:15:00 | 323.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 14:15:00 | 340.00 | 2024-10-24 14:15:00 | 341.75 | STOP_HIT | 0.50 | -0.51% |
| SELL | retest2 | 2024-10-23 09:15:00 | 336.40 | 2024-10-29 10:15:00 | 340.40 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-10-25 09:15:00 | 335.95 | 2024-10-29 10:15:00 | 340.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-01 18:00:00 | 360.95 | 2024-11-07 12:15:00 | 352.50 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-11-04 14:45:00 | 358.95 | 2024-11-07 12:15:00 | 352.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-11-11 09:15:00 | 346.65 | 2024-11-13 14:15:00 | 329.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 11:00:00 | 346.95 | 2024-11-13 14:15:00 | 329.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 346.65 | 2024-11-14 09:15:00 | 335.60 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2024-11-11 11:00:00 | 346.95 | 2024-11-14 09:15:00 | 335.60 | STOP_HIT | 0.50 | 3.27% |
| BUY | retest2 | 2024-11-25 09:15:00 | 340.25 | 2024-12-10 09:15:00 | 373.12 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2024-11-27 09:15:00 | 339.20 | 2024-12-11 09:15:00 | 374.28 | TARGET_HIT | 1.00 | 10.34% |
| SELL | retest2 | 2024-12-16 11:45:00 | 365.55 | 2024-12-30 14:15:00 | 347.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:45:00 | 365.55 | 2024-12-31 14:15:00 | 351.00 | STOP_HIT | 0.50 | 3.98% |
| BUY | retest2 | 2025-01-16 15:00:00 | 348.00 | 2025-01-22 13:15:00 | 340.45 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-02-07 11:15:00 | 360.50 | 2025-02-10 09:15:00 | 351.20 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-02-18 09:15:00 | 314.55 | 2025-02-19 13:15:00 | 322.50 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-19 10:45:00 | 318.20 | 2025-02-19 13:15:00 | 322.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2025-03-20 09:15:00 | 305.15 | 2025-03-21 09:15:00 | 320.41 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 10:30:00 | 303.85 | 2025-03-21 09:15:00 | 319.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-20 09:15:00 | 305.15 | 2025-03-21 10:15:00 | 335.67 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-03-20 10:30:00 | 303.85 | 2025-03-21 10:15:00 | 334.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-04 14:15:00 | 317.05 | 2025-04-07 09:15:00 | 298.85 | STOP_HIT | 1.00 | -5.74% |
| BUY | retest2 | 2025-04-04 14:45:00 | 317.85 | 2025-04-07 09:15:00 | 298.85 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest2 | 2025-04-17 11:30:00 | 324.00 | 2025-04-29 09:15:00 | 331.15 | STOP_HIT | 1.00 | 2.21% |
| SELL | retest2 | 2025-05-12 11:45:00 | 339.70 | 2025-05-12 15:15:00 | 341.75 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-05-13 15:00:00 | 344.45 | 2025-05-26 12:15:00 | 357.10 | STOP_HIT | 1.00 | 3.67% |
| SELL | retest2 | 2025-05-28 10:15:00 | 354.00 | 2025-05-30 11:15:00 | 357.35 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-05 12:30:00 | 359.95 | 2025-06-09 12:15:00 | 353.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-06-05 15:00:00 | 358.95 | 2025-06-09 12:15:00 | 353.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-06-18 11:15:00 | 347.15 | 2025-06-25 09:15:00 | 347.40 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-06-18 13:45:00 | 347.00 | 2025-06-25 09:15:00 | 347.40 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-06-19 09:45:00 | 346.75 | 2025-06-25 09:15:00 | 347.40 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-07-01 09:15:00 | 357.30 | 2025-07-07 10:15:00 | 355.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-01 11:45:00 | 356.15 | 2025-07-07 10:15:00 | 355.15 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-07-07 10:15:00 | 356.35 | 2025-07-07 10:15:00 | 355.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-14 09:30:00 | 361.90 | 2025-07-14 11:15:00 | 359.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-14 15:15:00 | 364.85 | 2025-07-18 09:15:00 | 401.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 11:30:00 | 438.70 | 2025-08-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-05 09:15:00 | 438.00 | 2025-08-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-08-05 10:00:00 | 438.60 | 2025-08-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-08-06 09:30:00 | 439.50 | 2025-08-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest1 | 2025-08-13 09:15:00 | 460.25 | 2025-08-14 10:15:00 | 449.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-08-14 12:45:00 | 450.50 | 2025-08-22 15:15:00 | 455.25 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-08-19 09:15:00 | 453.00 | 2025-08-22 15:15:00 | 455.25 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-22 09:15:00 | 452.00 | 2025-09-23 14:15:00 | 445.05 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-09-22 11:30:00 | 448.65 | 2025-09-23 14:15:00 | 445.05 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-22 12:00:00 | 449.70 | 2025-09-23 14:15:00 | 445.05 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-22 12:30:00 | 449.25 | 2025-09-23 14:15:00 | 445.05 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-29 14:30:00 | 416.65 | 2025-10-06 09:15:00 | 427.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-10-08 10:30:00 | 431.35 | 2025-10-13 14:15:00 | 429.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-10-08 13:45:00 | 431.20 | 2025-10-13 14:15:00 | 429.60 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-10-08 14:15:00 | 430.80 | 2025-10-13 14:15:00 | 429.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-08 15:15:00 | 430.85 | 2025-10-13 14:15:00 | 429.60 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-10-09 14:30:00 | 432.00 | 2025-10-13 14:15:00 | 429.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-13 10:45:00 | 430.40 | 2025-10-13 14:15:00 | 429.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-10-15 13:45:00 | 413.30 | 2025-10-20 15:15:00 | 418.65 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2025-11-10 09:15:00 | 375.80 | 2025-11-17 15:15:00 | 370.00 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-11-14 10:00:00 | 368.15 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-11-14 11:30:00 | 368.40 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-11-14 15:15:00 | 367.70 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-11-17 11:15:00 | 368.15 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-11-18 12:30:00 | 362.30 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-11-18 13:15:00 | 362.20 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-11-18 15:15:00 | 362.25 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-11-20 09:15:00 | 361.85 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-11-20 14:45:00 | 360.25 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2025-11-20 15:15:00 | 360.00 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2025-11-21 13:15:00 | 360.50 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2025-11-24 10:15:00 | 360.00 | 2025-11-25 09:15:00 | 378.45 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2025-12-08 09:30:00 | 348.00 | 2025-12-11 09:15:00 | 330.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:30:00 | 348.00 | 2025-12-11 11:15:00 | 336.15 | STOP_HIT | 0.50 | 3.41% |
| BUY | retest2 | 2025-12-24 10:15:00 | 366.70 | 2025-12-29 14:15:00 | 361.55 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-26 09:15:00 | 371.00 | 2025-12-29 14:15:00 | 361.55 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-12-29 11:30:00 | 366.00 | 2025-12-29 14:15:00 | 361.55 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-01 10:15:00 | 353.35 | 2026-01-05 15:15:00 | 358.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-01 12:00:00 | 352.70 | 2026-01-05 15:15:00 | 358.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-08 09:15:00 | 350.40 | 2026-01-13 10:15:00 | 353.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-01-20 13:30:00 | 345.55 | 2026-01-21 15:15:00 | 350.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-01-20 14:00:00 | 346.45 | 2026-01-21 15:15:00 | 350.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-20 15:15:00 | 345.00 | 2026-01-21 15:15:00 | 350.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-01-29 14:30:00 | 350.50 | 2026-01-30 09:15:00 | 348.25 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-11 11:15:00 | 350.45 | 2026-02-12 15:15:00 | 347.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-12 13:15:00 | 348.85 | 2026-02-12 15:15:00 | 347.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-02-12 14:15:00 | 349.50 | 2026-02-12 15:15:00 | 347.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-02-24 09:30:00 | 330.45 | 2026-03-04 09:15:00 | 313.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 10:00:00 | 331.20 | 2026-03-04 09:15:00 | 314.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 331.75 | 2026-03-04 09:15:00 | 315.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 331.00 | 2026-03-04 09:15:00 | 314.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:30:00 | 331.55 | 2026-03-04 09:15:00 | 314.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 331.35 | 2026-03-04 09:15:00 | 314.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:30:00 | 330.45 | 2026-03-05 11:15:00 | 308.80 | STOP_HIT | 0.50 | 6.55% |
| SELL | retest2 | 2026-02-24 10:00:00 | 331.20 | 2026-03-05 11:15:00 | 308.80 | STOP_HIT | 0.50 | 6.76% |
| SELL | retest2 | 2026-02-25 12:45:00 | 331.75 | 2026-03-05 11:15:00 | 308.80 | STOP_HIT | 0.50 | 6.92% |
| SELL | retest2 | 2026-02-25 15:15:00 | 331.00 | 2026-03-05 11:15:00 | 308.80 | STOP_HIT | 0.50 | 6.71% |
| SELL | retest2 | 2026-02-26 12:30:00 | 331.55 | 2026-03-05 11:15:00 | 308.80 | STOP_HIT | 0.50 | 6.86% |
| SELL | retest2 | 2026-02-26 15:15:00 | 331.35 | 2026-03-05 11:15:00 | 308.80 | STOP_HIT | 0.50 | 6.81% |
| SELL | retest2 | 2026-03-16 10:15:00 | 282.35 | 2026-03-17 15:15:00 | 294.00 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2026-04-02 09:15:00 | 282.75 | 2026-04-02 14:15:00 | 294.55 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2026-04-02 14:30:00 | 287.25 | 2026-04-06 12:15:00 | 292.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-04-06 11:00:00 | 287.80 | 2026-04-06 12:15:00 | 292.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-04-07 15:15:00 | 293.90 | 2026-04-15 10:15:00 | 300.60 | STOP_HIT | 1.00 | 2.28% |
| SELL | retest2 | 2026-04-28 09:30:00 | 292.45 | 2026-04-30 14:15:00 | 297.30 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-04-28 10:30:00 | 293.75 | 2026-04-30 14:15:00 | 297.30 | STOP_HIT | 1.00 | -1.21% |
