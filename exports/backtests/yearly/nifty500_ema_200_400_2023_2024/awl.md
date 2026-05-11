# AWL Agri Business Ltd. (AWL)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 206.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 86 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 95 |
| PARTIAL | 31 |
| TARGET_HIT | 12 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 126 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 69
- **Target hits / Stop hits / Partials:** 11 / 84 / 31
- **Avg / median % per leg:** 0.96% / -0.32%
- **Sum % (uncompounded):** 121.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.89% | -26.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.89% | -26.5% |
| SELL (all) | 112 | 57 | 50.9% | 11 | 70 | 31 | 1.32% | 147.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 112 | 57 | 50.9% | 11 | 70 | 31 | 1.32% | 147.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 126 | 57 | 45.2% | 11 | 84 | 31 | 0.96% | 121.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 372.85 | 343.19 | 343.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 394.30 | 355.98 | 351.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 09:15:00 | 361.60 | 363.45 | 357.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 09:30:00 | 362.00 | 363.45 | 357.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 356.25 | 363.22 | 357.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 09:15:00 | 360.70 | 362.77 | 357.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 15:15:00 | 359.00 | 362.39 | 357.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:45:00 | 358.75 | 362.31 | 357.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 09:15:00 | 352.15 | 361.79 | 356.96 | SL hit (close<static) qty=1.00 sl=353.20 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 339.95 | 354.58 | 354.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 14:15:00 | 338.30 | 354.41 | 354.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 12:15:00 | 350.80 | 350.60 | 352.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 12:15:00 | 350.80 | 350.60 | 352.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 350.80 | 350.60 | 352.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:00:00 | 350.80 | 350.60 | 352.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 366.50 | 350.65 | 352.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:30:00 | 370.15 | 350.65 | 352.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 365.95 | 350.80 | 352.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:30:00 | 366.90 | 350.80 | 352.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 11:15:00 | 361.80 | 354.07 | 354.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 384.35 | 354.62 | 354.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 365.40 | 367.35 | 361.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 15:00:00 | 365.40 | 367.35 | 361.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 364.35 | 367.31 | 361.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 364.35 | 367.31 | 361.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 361.20 | 367.25 | 361.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 361.20 | 367.25 | 361.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 11:15:00 | 358.30 | 367.16 | 361.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 12:00:00 | 358.30 | 367.16 | 361.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 12:15:00 | 359.50 | 367.09 | 361.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 13:45:00 | 361.50 | 367.04 | 361.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 10:15:00 | 361.95 | 366.62 | 361.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-11 11:15:00 | 356.55 | 366.43 | 361.82 | SL hit (close<static) qty=1.00 sl=356.90 alert=retest2 |

### Cycle 4 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 337.10 | 358.18 | 358.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 335.25 | 357.74 | 357.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 11:15:00 | 352.55 | 347.59 | 352.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 11:15:00 | 352.55 | 347.59 | 352.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 352.55 | 347.59 | 352.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 12:00:00 | 352.55 | 347.59 | 352.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 12:15:00 | 352.85 | 347.64 | 352.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:30:00 | 351.05 | 348.70 | 352.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 351.40 | 349.74 | 352.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 10:45:00 | 350.55 | 349.76 | 352.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 14:00:00 | 351.30 | 349.78 | 352.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 333.50 | 349.53 | 352.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 333.83 | 349.53 | 352.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 333.02 | 349.53 | 352.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 333.74 | 349.53 | 352.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-29 10:15:00 | 345.95 | 345.39 | 349.03 | SL hit (close>ema200) qty=0.50 sl=345.39 alert=retest2 |

### Cycle 5 — BUY (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 12:15:00 | 377.95 | 340.49 | 340.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-05 13:15:00 | 387.85 | 340.97 | 340.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 10:15:00 | 353.95 | 354.40 | 348.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 11:00:00 | 353.95 | 354.40 | 348.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 358.60 | 365.29 | 358.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 360.15 | 365.29 | 358.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 359.85 | 365.24 | 358.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 366.00 | 364.94 | 358.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 15:15:00 | 361.80 | 364.81 | 358.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 15:15:00 | 357.00 | 364.47 | 358.28 | SL hit (close<static) qty=1.00 sl=357.50 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 339.50 | 355.28 | 355.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 338.35 | 354.29 | 354.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 339.40 | 338.49 | 345.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 339.40 | 338.49 | 345.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 345.95 | 336.78 | 343.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 347.00 | 336.78 | 343.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 346.45 | 336.88 | 343.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 346.45 | 336.88 | 343.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 346.00 | 336.97 | 343.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 12:15:00 | 345.20 | 336.97 | 343.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 13:30:00 | 345.15 | 337.13 | 343.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-01 17:15:00 | 347.65 | 337.41 | 343.36 | SL hit (close>static) qty=1.00 sl=347.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 12:15:00 | 283.45 | 269.75 | 269.71 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 14:15:00 | 262.25 | 269.81 | 269.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 257.45 | 269.40 | 269.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 267.40 | 267.31 | 268.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 267.40 | 267.31 | 268.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 267.40 | 267.31 | 268.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:30:00 | 268.45 | 267.31 | 268.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 269.55 | 267.34 | 268.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 268.55 | 267.34 | 268.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 267.35 | 267.34 | 268.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 14:30:00 | 267.00 | 267.33 | 268.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 15:00:00 | 267.05 | 267.37 | 268.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 11:15:00 | 271.75 | 267.47 | 268.48 | SL hit (close>static) qty=1.00 sl=270.90 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 279.70 | 265.36 | 265.34 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 253.45 | 265.97 | 265.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 252.95 | 265.84 | 265.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 260.20 | 259.72 | 262.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 258.45 | 259.72 | 262.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 262.10 | 259.75 | 262.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 262.45 | 259.75 | 262.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 261.70 | 259.77 | 262.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 262.35 | 259.77 | 262.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 262.00 | 259.82 | 262.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 260.70 | 259.82 | 262.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 260.85 | 259.83 | 262.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 258.20 | 259.88 | 262.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 258.85 | 259.83 | 262.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 258.30 | 259.82 | 262.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 258.80 | 259.67 | 261.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 260.10 | 257.39 | 260.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 260.20 | 257.39 | 260.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 263.70 | 257.45 | 260.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 263.70 | 257.45 | 260.33 | SL hit (close>static) qty=1.00 sl=263.40 alert=retest2 |

### Cycle 11 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 265.80 | 261.06 | 261.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 266.60 | 261.20 | 261.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 261.70 | 263.21 | 262.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 260.50 | 263.18 | 262.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 262.25 | 263.18 | 262.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 259.85 | 263.07 | 262.22 | SL hit (close<static) qty=1.00 sl=260.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 246.40 | 265.16 | 265.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 245.35 | 261.09 | 263.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 11:15:00 | 196.28 | 196.17 | 210.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 196.28 | 196.17 | 210.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 196.18 | 184.24 | 193.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 196.18 | 184.24 | 193.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 196.25 | 184.36 | 193.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 196.66 | 184.36 | 193.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-31 13:15:00 | 438.30 | 2023-06-15 09:15:00 | 436.15 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2023-05-31 14:00:00 | 438.75 | 2023-06-15 09:15:00 | 436.15 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2023-05-31 14:45:00 | 437.15 | 2023-06-15 09:15:00 | 436.15 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2023-06-02 09:45:00 | 439.25 | 2023-06-19 13:15:00 | 416.38 | PARTIAL | 0.50 | 5.21% |
| SELL | retest2 | 2023-06-12 12:15:00 | 422.00 | 2023-06-19 13:15:00 | 416.81 | PARTIAL | 0.50 | 1.23% |
| SELL | retest2 | 2023-06-12 15:15:00 | 421.95 | 2023-06-19 13:15:00 | 417.29 | PARTIAL | 0.50 | 1.10% |
| SELL | retest2 | 2023-06-13 09:30:00 | 422.25 | 2023-06-20 09:15:00 | 415.29 | PARTIAL | 0.50 | 1.65% |
| SELL | retest2 | 2023-06-02 09:45:00 | 439.25 | 2023-06-26 09:15:00 | 394.47 | TARGET_HIT | 0.50 | 10.19% |
| SELL | retest2 | 2023-06-12 12:15:00 | 422.00 | 2023-06-26 09:15:00 | 394.88 | TARGET_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2023-06-12 15:15:00 | 421.95 | 2023-06-26 09:15:00 | 395.32 | TARGET_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2023-06-16 13:00:00 | 421.65 | 2023-06-26 09:15:00 | 400.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-13 09:30:00 | 422.25 | 2023-07-25 14:15:00 | 417.55 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest2 | 2023-06-16 13:00:00 | 421.65 | 2023-07-25 14:15:00 | 417.55 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2023-07-27 11:15:00 | 410.70 | 2023-08-03 09:15:00 | 390.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-28 12:00:00 | 409.20 | 2023-08-03 09:15:00 | 388.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-31 09:45:00 | 410.75 | 2023-08-03 09:15:00 | 390.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 10:30:00 | 409.40 | 2023-08-03 09:15:00 | 388.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 14:00:00 | 411.90 | 2023-08-03 09:15:00 | 391.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-02 10:30:00 | 410.55 | 2023-08-03 09:15:00 | 390.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-27 11:15:00 | 410.70 | 2023-08-16 10:15:00 | 370.71 | TARGET_HIT | 0.50 | 9.74% |
| SELL | retest2 | 2023-07-28 12:00:00 | 409.20 | 2023-08-16 11:15:00 | 369.63 | TARGET_HIT | 0.50 | 9.67% |
| SELL | retest2 | 2023-07-31 09:45:00 | 410.75 | 2023-08-16 11:15:00 | 369.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-08-01 10:30:00 | 409.40 | 2023-08-16 11:15:00 | 369.50 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2023-08-01 14:00:00 | 411.90 | 2023-08-17 15:15:00 | 368.28 | TARGET_HIT | 0.50 | 10.59% |
| SELL | retest2 | 2023-08-02 10:30:00 | 410.55 | 2023-08-17 15:15:00 | 368.46 | TARGET_HIT | 0.50 | 10.25% |
| BUY | retest2 | 2024-01-18 09:15:00 | 360.70 | 2024-01-20 09:15:00 | 352.15 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-01-18 15:15:00 | 359.00 | 2024-01-20 09:15:00 | 352.15 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-01-19 09:45:00 | 358.75 | 2024-01-20 09:15:00 | 352.15 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-01-20 14:30:00 | 362.95 | 2024-01-23 14:15:00 | 352.45 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-03-06 13:45:00 | 361.50 | 2024-03-11 11:15:00 | 356.55 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-03-11 10:15:00 | 361.95 | 2024-03-11 11:15:00 | 356.55 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-04-05 09:30:00 | 351.05 | 2024-04-15 09:15:00 | 333.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 09:15:00 | 351.40 | 2024-04-15 09:15:00 | 333.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 10:45:00 | 350.55 | 2024-04-15 09:15:00 | 333.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 14:00:00 | 351.30 | 2024-04-15 09:15:00 | 333.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 09:30:00 | 351.05 | 2024-04-29 10:15:00 | 345.95 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2024-04-10 09:15:00 | 351.40 | 2024-04-29 10:15:00 | 345.95 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2024-04-10 10:45:00 | 350.55 | 2024-04-29 10:15:00 | 345.95 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2024-04-10 14:00:00 | 351.30 | 2024-04-29 10:15:00 | 345.95 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2024-04-30 12:30:00 | 354.80 | 2024-05-06 09:15:00 | 337.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 13:00:00 | 354.95 | 2024-05-06 09:15:00 | 337.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 09:45:00 | 354.50 | 2024-05-06 09:15:00 | 336.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 12:30:00 | 354.80 | 2024-05-21 11:15:00 | 349.10 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2024-04-30 13:00:00 | 354.95 | 2024-05-21 11:15:00 | 349.10 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2024-05-02 09:45:00 | 354.50 | 2024-05-21 11:15:00 | 349.10 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2024-05-31 15:00:00 | 354.60 | 2024-06-03 09:15:00 | 367.95 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2024-06-04 12:00:00 | 331.45 | 2024-06-10 09:15:00 | 347.40 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-06-04 14:45:00 | 331.45 | 2024-06-10 09:15:00 | 347.40 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-06-05 12:15:00 | 333.35 | 2024-06-10 09:15:00 | 347.40 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2024-06-05 14:00:00 | 333.70 | 2024-06-10 09:15:00 | 347.40 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2024-06-06 11:30:00 | 343.85 | 2024-06-13 09:15:00 | 346.85 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-07 10:45:00 | 344.00 | 2024-06-13 09:15:00 | 346.85 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-06-07 12:30:00 | 344.35 | 2024-07-18 09:15:00 | 327.99 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2024-06-07 14:00:00 | 344.20 | 2024-07-18 09:15:00 | 328.03 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2024-06-10 14:15:00 | 345.25 | 2024-07-18 09:15:00 | 328.08 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2024-06-10 15:00:00 | 345.30 | 2024-07-18 09:15:00 | 328.18 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-06-11 09:30:00 | 345.35 | 2024-07-18 09:15:00 | 327.18 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2024-06-11 10:15:00 | 345.45 | 2024-07-18 09:15:00 | 327.56 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2024-06-11 15:00:00 | 344.25 | 2024-07-18 09:15:00 | 327.23 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-06-12 14:15:00 | 344.45 | 2024-07-18 10:15:00 | 326.89 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-06-13 11:15:00 | 344.10 | 2024-07-22 09:15:00 | 316.68 | PARTIAL | 0.50 | 7.97% |
| SELL | retest2 | 2024-06-13 14:00:00 | 344.40 | 2024-07-22 09:15:00 | 317.01 | PARTIAL | 0.50 | 7.95% |
| SELL | retest2 | 2024-06-07 12:30:00 | 344.35 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.58% |
| SELL | retest2 | 2024-06-07 14:00:00 | 344.20 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.62% |
| SELL | retest2 | 2024-06-10 14:15:00 | 345.25 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2024-06-10 15:00:00 | 345.30 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.30% |
| SELL | retest2 | 2024-06-11 09:30:00 | 345.35 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.29% |
| SELL | retest2 | 2024-06-11 10:15:00 | 345.45 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.26% |
| SELL | retest2 | 2024-06-11 15:00:00 | 344.25 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2024-06-12 14:15:00 | 344.45 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.55% |
| SELL | retest2 | 2024-06-13 11:15:00 | 344.10 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.65% |
| SELL | retest2 | 2024-06-13 14:00:00 | 344.40 | 2024-07-29 11:15:00 | 346.35 | STOP_HIT | 0.50 | -0.57% |
| SELL | retest2 | 2024-06-14 12:30:00 | 344.80 | 2024-07-30 09:15:00 | 349.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-06-14 13:30:00 | 344.45 | 2024-07-30 09:15:00 | 349.20 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-07-29 12:45:00 | 344.85 | 2024-08-02 09:15:00 | 362.95 | STOP_HIT | 1.00 | -5.25% |
| SELL | retest2 | 2024-07-29 13:45:00 | 343.40 | 2024-08-02 09:15:00 | 362.95 | STOP_HIT | 1.00 | -5.69% |
| BUY | retest2 | 2024-09-10 09:15:00 | 366.00 | 2024-09-11 15:15:00 | 357.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-09-10 15:15:00 | 361.80 | 2024-09-11 15:15:00 | 357.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-12 14:30:00 | 361.50 | 2024-09-18 11:15:00 | 355.70 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-09-16 09:15:00 | 369.05 | 2024-09-18 11:15:00 | 355.70 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-10-31 12:15:00 | 345.20 | 2024-11-01 17:15:00 | 347.65 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-10-31 13:30:00 | 345.15 | 2024-11-01 17:15:00 | 347.65 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-11-01 18:15:00 | 344.85 | 2024-11-05 09:15:00 | 327.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-01 18:15:00 | 344.85 | 2024-11-05 09:15:00 | 337.20 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-11-06 13:30:00 | 345.25 | 2024-11-11 10:15:00 | 327.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 13:30:00 | 345.25 | 2024-11-21 09:15:00 | 310.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-27 09:15:00 | 317.50 | 2024-12-27 11:15:00 | 322.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-12-31 09:15:00 | 305.50 | 2025-01-01 10:15:00 | 326.80 | STOP_HIT | 1.00 | -6.97% |
| SELL | retest2 | 2025-01-10 09:15:00 | 296.20 | 2025-01-13 09:15:00 | 281.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 296.20 | 2025-01-13 14:15:00 | 266.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-14 14:30:00 | 267.00 | 2025-05-16 11:15:00 | 271.75 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-15 15:00:00 | 267.05 | 2025-05-16 11:15:00 | 271.75 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-05-20 09:15:00 | 266.65 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-20 11:30:00 | 267.05 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-05-27 13:45:00 | 267.85 | 2025-05-30 14:15:00 | 278.45 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-05-30 15:15:00 | 267.90 | 2025-06-10 09:15:00 | 268.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-02 10:15:00 | 270.45 | 2025-06-10 09:15:00 | 268.60 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-06-02 11:45:00 | 270.30 | 2025-06-10 09:15:00 | 268.60 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-06-05 09:45:00 | 266.85 | 2025-06-10 12:15:00 | 275.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-06-05 11:00:00 | 266.85 | 2025-06-10 12:15:00 | 275.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-06-09 10:00:00 | 266.70 | 2025-06-10 12:15:00 | 275.10 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-06-13 09:15:00 | 264.35 | 2025-07-10 09:15:00 | 264.35 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-07-09 14:15:00 | 262.00 | 2025-07-10 10:15:00 | 269.60 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-16 09:15:00 | 261.70 | 2025-07-17 09:15:00 | 277.20 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2025-07-16 12:30:00 | 261.50 | 2025-07-17 09:15:00 | 277.20 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 258.20 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-22 12:00:00 | 258.85 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-08-22 15:00:00 | 258.30 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-25 15:15:00 | 258.80 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-08 11:45:00 | 260.35 | 2025-09-12 09:15:00 | 261.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-09 10:00:00 | 260.50 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-11 11:15:00 | 260.30 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-11 12:15:00 | 260.55 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-11 15:00:00 | 258.45 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-15 10:00:00 | 258.75 | 2025-09-19 13:15:00 | 262.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-15 14:00:00 | 258.65 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-09-15 15:00:00 | 258.00 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-09-16 10:45:00 | 258.45 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-09-19 10:45:00 | 258.50 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-09-19 15:00:00 | 257.00 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2025-09-24 09:15:00 | 256.90 | 2025-09-25 14:15:00 | 262.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-24 10:30:00 | 256.40 | 2025-09-25 14:15:00 | 262.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-10-20 09:15:00 | 262.25 | 2025-10-20 12:15:00 | 259.85 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-24 12:30:00 | 263.10 | 2025-11-28 12:15:00 | 259.05 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-10-27 10:30:00 | 262.90 | 2025-11-28 12:15:00 | 259.05 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-28 09:30:00 | 263.40 | 2025-11-28 12:15:00 | 259.05 | STOP_HIT | 1.00 | -1.65% |
