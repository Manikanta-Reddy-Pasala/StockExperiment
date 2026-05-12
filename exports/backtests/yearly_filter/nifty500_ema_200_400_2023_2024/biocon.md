# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 378.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 14 |
| ALERT2 | 15 |
| ALERT2_SKIP | 4 |
| ALERT3 | 115 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 96 |
| PARTIAL | 3 |
| TARGET_HIT | 26 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 99 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 70
- **Target hits / Stop hits / Partials:** 26 / 70 / 3
- **Avg / median % per leg:** 1.52% / -1.08%
- **Sum % (uncompounded):** 150.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 22 | 48.9% | 22 | 23 | 0 | 3.88% | 174.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 45 | 22 | 48.9% | 22 | 23 | 0 | 3.88% | 174.4% |
| SELL (all) | 54 | 7 | 13.0% | 4 | 47 | 3 | -0.45% | -24.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 7 | 13.0% | 4 | 47 | 3 | -0.45% | -24.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 99 | 29 | 29.3% | 26 | 70 | 3 | 1.52% | 150.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 13:15:00 | 235.60 | 259.83 | 259.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 234.45 | 259.10 | 259.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 237.85 | 235.16 | 243.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-20 10:00:00 | 237.85 | 235.16 | 243.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 241.35 | 235.44 | 241.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 15:15:00 | 240.30 | 235.72 | 241.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 09:45:00 | 239.80 | 235.80 | 241.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 12:15:00 | 240.25 | 235.90 | 241.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-04 13:30:00 | 240.30 | 235.99 | 241.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 240.90 | 236.08 | 241.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-05 09:15:00 | 242.90 | 236.08 | 241.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 242.25 | 236.14 | 241.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 10:00:00 | 241.05 | 236.57 | 241.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 10:45:00 | 241.30 | 236.62 | 241.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 12:00:00 | 241.45 | 236.67 | 241.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 12:45:00 | 241.00 | 236.71 | 241.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 240.35 | 236.79 | 241.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 12:15:00 | 239.75 | 237.35 | 241.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 12:30:00 | 239.05 | 237.49 | 241.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 14:30:00 | 240.00 | 237.54 | 241.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 09:15:00 | 241.65 | 237.61 | 241.33 | SL hit (close>static) qty=1.00 sl=241.40 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 09:15:00 | 248.15 | 243.75 | 243.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 12:15:00 | 258.35 | 244.69 | 244.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 11:15:00 | 267.65 | 267.99 | 259.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 12:00:00 | 267.65 | 267.99 | 259.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 259.15 | 267.39 | 259.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:45:00 | 259.75 | 267.39 | 259.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 259.10 | 267.31 | 259.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 11:15:00 | 260.70 | 266.89 | 259.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 09:15:00 | 262.75 | 266.54 | 259.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 11:00:00 | 261.45 | 266.45 | 259.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:15:00 | 261.30 | 266.20 | 259.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-05 12:15:00 | 286.77 | 267.57 | 261.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 11:15:00 | 249.10 | 267.07 | 267.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-21 15:15:00 | 248.30 | 266.36 | 266.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 11:15:00 | 264.60 | 264.49 | 265.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 11:15:00 | 264.60 | 264.49 | 265.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 264.60 | 264.49 | 265.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:00:00 | 264.60 | 264.49 | 265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 265.95 | 264.50 | 265.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:45:00 | 265.55 | 264.50 | 265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 265.50 | 264.51 | 265.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 13:30:00 | 265.55 | 264.51 | 265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 267.45 | 264.53 | 265.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:30:00 | 267.75 | 264.53 | 265.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 268.80 | 264.58 | 265.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:00:00 | 268.80 | 264.58 | 265.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 14:15:00 | 274.25 | 266.68 | 266.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 10:15:00 | 278.05 | 267.32 | 267.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 267.50 | 268.41 | 267.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 267.50 | 268.41 | 267.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 267.50 | 268.41 | 267.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 264.65 | 268.41 | 267.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 267.20 | 268.40 | 267.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:30:00 | 268.45 | 268.40 | 267.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 268.35 | 268.40 | 267.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 12:15:00 | 268.95 | 268.40 | 267.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 12:15:00 | 266.60 | 268.38 | 267.59 | SL hit (close<static) qty=1.00 sl=266.65 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 332.70 | 353.54 | 353.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 324.65 | 352.57 | 353.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 335.95 | 333.90 | 341.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 10:15:00 | 338.70 | 333.90 | 341.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 340.10 | 334.00 | 341.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:45:00 | 340.65 | 334.00 | 341.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 347.35 | 334.13 | 341.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 347.35 | 334.13 | 341.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 348.10 | 334.27 | 341.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:45:00 | 349.55 | 334.27 | 341.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 338.60 | 333.68 | 339.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:30:00 | 339.60 | 333.68 | 339.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 342.05 | 333.76 | 339.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:45:00 | 343.30 | 333.76 | 339.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 342.30 | 333.84 | 339.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:00:00 | 342.30 | 333.84 | 339.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 373.50 | 344.27 | 344.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 380.15 | 345.97 | 344.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 350.30 | 353.97 | 349.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 10:00:00 | 350.30 | 353.97 | 349.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 349.00 | 353.92 | 349.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 348.35 | 353.92 | 349.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 352.35 | 353.91 | 349.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:30:00 | 350.00 | 353.91 | 349.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 350.25 | 353.74 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 351.00 | 353.74 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 349.90 | 353.70 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 349.90 | 353.70 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 351.20 | 353.67 | 349.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 352.70 | 353.59 | 349.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 10:00:00 | 352.70 | 353.58 | 349.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 14:15:00 | 347.45 | 353.51 | 349.98 | SL hit (close<static) qty=1.00 sl=349.55 alert=retest2 |

### Cycle 7 — SELL (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 13:15:00 | 334.75 | 363.00 | 363.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 328.00 | 362.14 | 362.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 337.00 | 336.69 | 345.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 14:00:00 | 337.00 | 336.69 | 345.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 343.15 | 337.29 | 345.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 345.25 | 337.29 | 345.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 345.10 | 337.43 | 345.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:45:00 | 345.15 | 337.43 | 345.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 344.90 | 337.51 | 345.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:15:00 | 344.45 | 337.51 | 345.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 14:45:00 | 344.00 | 337.64 | 345.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 09:15:00 | 347.95 | 337.82 | 345.45 | SL hit (close>static) qty=1.00 sl=346.40 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 14:15:00 | 352.75 | 336.63 | 336.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 15:15:00 | 357.55 | 337.90 | 337.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 379.30 | 380.66 | 367.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:45:00 | 379.15 | 380.66 | 367.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 366.05 | 380.11 | 368.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 366.05 | 380.11 | 368.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 364.35 | 379.96 | 368.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 364.35 | 379.96 | 368.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 359.35 | 371.10 | 365.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 359.35 | 371.10 | 365.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 365.10 | 370.22 | 365.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 367.20 | 362.82 | 362.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:15:00 | 365.50 | 362.87 | 362.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:30:00 | 365.85 | 362.92 | 362.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 365.75 | 362.92 | 362.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 15:15:00 | 363.25 | 362.94 | 362.69 | SL hit (close<static) qty=1.00 sl=363.65 alert=retest2 |

### Cycle 9 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 361.10 | 362.59 | 362.60 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 366.50 | 362.63 | 362.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 368.35 | 362.69 | 362.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 362.85 | 363.30 | 362.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 12:15:00 | 362.85 | 363.30 | 362.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 362.85 | 363.30 | 362.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 362.60 | 363.30 | 362.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 361.30 | 363.28 | 362.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 361.30 | 363.28 | 362.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 360.95 | 363.23 | 362.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 360.10 | 363.23 | 362.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 358.60 | 363.18 | 362.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 358.60 | 363.18 | 362.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 356.45 | 362.64 | 362.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 355.10 | 362.45 | 362.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 355.75 | 355.03 | 358.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 355.75 | 355.03 | 358.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 358.00 | 354.28 | 357.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 358.25 | 354.28 | 357.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 358.65 | 354.32 | 357.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 358.80 | 354.32 | 357.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 358.70 | 354.44 | 357.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 358.70 | 354.44 | 357.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 358.40 | 354.61 | 357.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 358.40 | 354.61 | 357.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 356.10 | 354.63 | 357.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 358.35 | 354.63 | 357.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 361.40 | 354.73 | 357.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 361.40 | 354.73 | 357.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 361.50 | 354.80 | 357.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 361.60 | 354.80 | 357.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 359.20 | 356.28 | 357.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 362.70 | 356.28 | 357.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 375.40 | 359.27 | 359.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 14:15:00 | 380.40 | 362.28 | 360.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 388.75 | 391.04 | 380.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 388.75 | 391.04 | 380.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 379.60 | 390.49 | 381.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 379.60 | 390.49 | 381.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 380.80 | 390.40 | 381.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 377.05 | 390.40 | 381.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 383.75 | 390.16 | 381.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:30:00 | 381.80 | 390.16 | 381.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 380.60 | 390.00 | 381.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 380.40 | 390.00 | 381.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 378.65 | 389.89 | 381.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 378.65 | 389.89 | 381.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 381.60 | 389.80 | 381.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 13:15:00 | 381.85 | 389.72 | 381.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 381.85 | 389.23 | 381.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:15:00 | 382.45 | 388.90 | 381.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 11:30:00 | 382.25 | 391.10 | 386.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 377.50 | 390.63 | 386.37 | SL hit (close<static) qty=1.00 sl=377.80 alert=retest2 |

### Cycle 13 — SELL (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 10:15:00 | 371.00 | 383.29 | 383.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 11:15:00 | 369.95 | 383.16 | 383.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 376.55 | 376.42 | 379.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:30:00 | 377.35 | 376.42 | 379.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 385.55 | 374.18 | 377.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 385.55 | 374.18 | 377.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 381.60 | 374.25 | 377.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 374.95 | 374.26 | 377.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 378.10 | 374.31 | 377.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:30:00 | 378.80 | 374.49 | 377.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 378.20 | 374.49 | 377.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 378.10 | 374.53 | 377.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 378.25 | 374.53 | 377.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 382.25 | 374.61 | 377.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 382.85 | 374.61 | 377.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 382.90 | 374.69 | 377.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 382.45 | 374.69 | 377.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 378.85 | 375.07 | 377.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 377.80 | 375.24 | 377.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 377.90 | 375.31 | 377.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 376.60 | 375.36 | 377.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 377.95 | 375.51 | 377.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 383.65 | 375.59 | 377.71 | SL hit (close>static) qty=1.00 sl=380.55 alert=retest2 |

### Cycle 14 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 396.20 | 379.49 | 379.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 400.90 | 383.45 | 381.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 384.05 | 384.84 | 382.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:00:00 | 384.05 | 384.84 | 382.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 385.00 | 384.90 | 382.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 385.95 | 384.87 | 382.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 372.90 | 384.75 | 382.56 | SL hit (close<static) qty=1.00 sl=382.60 alert=retest2 |

### Cycle 15 — SELL (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 15:15:00 | 369.10 | 380.88 | 380.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 362.85 | 380.07 | 380.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 364.50 | 363.29 | 370.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 10:00:00 | 364.50 | 363.29 | 370.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 362.95 | 361.66 | 367.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 362.90 | 361.78 | 367.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 362.25 | 361.78 | 367.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 358.55 | 361.80 | 367.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 362.55 | 361.72 | 367.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 367.60 | 361.86 | 367.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 367.20 | 361.86 | 367.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 367.55 | 361.92 | 367.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 367.50 | 361.92 | 367.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 368.00 | 361.98 | 367.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:15:00 | 370.00 | 361.98 | 367.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 370.00 | 362.06 | 367.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 370.00 | 362.06 | 367.30 | SL hit (close>static) qty=1.00 sl=368.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-05 11:00:00 | 242.45 | 2023-06-30 14:15:00 | 266.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-05 12:45:00 | 242.10 | 2023-06-30 14:15:00 | 266.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-05 13:15:00 | 242.25 | 2023-06-30 14:15:00 | 266.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-06 14:15:00 | 242.75 | 2023-06-30 14:15:00 | 267.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-26 14:15:00 | 239.90 | 2023-06-30 14:15:00 | 263.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-26 14:45:00 | 240.40 | 2023-06-30 14:15:00 | 264.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-27 09:15:00 | 243.05 | 2023-06-30 14:15:00 | 267.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-01 15:15:00 | 240.30 | 2023-12-12 09:15:00 | 241.65 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-12-04 09:45:00 | 239.80 | 2023-12-12 09:15:00 | 241.65 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-12-04 12:15:00 | 240.25 | 2023-12-12 09:15:00 | 241.65 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-12-04 13:30:00 | 240.30 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2023-12-06 10:00:00 | 241.05 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2023-12-06 10:45:00 | 241.30 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2023-12-06 12:00:00 | 241.45 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2023-12-06 12:45:00 | 241.00 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2023-12-08 12:15:00 | 239.75 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2023-12-11 12:30:00 | 239.05 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2023-12-11 14:30:00 | 240.00 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2023-12-12 14:30:00 | 238.80 | 2023-12-13 13:15:00 | 246.50 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2024-01-29 11:15:00 | 260.70 | 2024-02-05 12:15:00 | 286.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-30 09:15:00 | 262.75 | 2024-02-05 12:15:00 | 287.60 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2024-01-30 11:00:00 | 261.45 | 2024-02-05 12:15:00 | 287.43 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2024-01-31 09:15:00 | 261.30 | 2024-02-05 14:15:00 | 289.03 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2024-02-14 14:30:00 | 271.70 | 2024-02-19 09:15:00 | 298.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-14 15:15:00 | 273.00 | 2024-02-19 09:15:00 | 300.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-22 09:30:00 | 272.45 | 2024-03-13 09:15:00 | 264.60 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-02-22 13:45:00 | 271.70 | 2024-03-13 09:15:00 | 264.60 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-02-29 14:30:00 | 275.50 | 2024-03-13 09:15:00 | 264.60 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-03-06 13:30:00 | 273.10 | 2024-03-13 10:15:00 | 261.45 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2024-03-12 11:45:00 | 272.90 | 2024-03-13 10:15:00 | 261.45 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2024-04-15 12:15:00 | 268.95 | 2024-04-15 12:15:00 | 266.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-04-16 11:00:00 | 268.70 | 2024-04-16 11:15:00 | 265.75 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-04-18 09:15:00 | 269.80 | 2024-04-19 09:15:00 | 263.45 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-04-22 11:00:00 | 268.80 | 2024-04-26 09:15:00 | 295.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 12:15:00 | 301.20 | 2024-06-06 14:15:00 | 331.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 12:45:00 | 300.95 | 2024-06-06 14:15:00 | 331.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-18 09:15:00 | 352.70 | 2024-12-18 14:15:00 | 347.45 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-18 10:00:00 | 352.70 | 2024-12-18 14:15:00 | 347.45 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-27 09:30:00 | 353.05 | 2025-01-07 11:15:00 | 388.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-27 13:00:00 | 352.95 | 2025-01-07 11:15:00 | 388.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-29 14:30:00 | 358.00 | 2025-02-05 11:15:00 | 392.59 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2025-01-30 09:30:00 | 358.00 | 2025-02-06 09:15:00 | 393.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-30 11:00:00 | 356.90 | 2025-02-06 09:15:00 | 393.80 | TARGET_HIT | 1.00 | 10.34% |
| BUY | retest2 | 2025-01-30 11:30:00 | 358.00 | 2025-02-06 09:15:00 | 393.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 13:30:00 | 370.30 | 2025-02-11 12:15:00 | 361.15 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-02-04 09:15:00 | 370.50 | 2025-02-11 12:15:00 | 361.15 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-02-11 10:15:00 | 368.55 | 2025-02-11 12:15:00 | 361.15 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-02-11 10:45:00 | 368.75 | 2025-02-11 12:15:00 | 361.15 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-03-21 13:15:00 | 344.45 | 2025-03-24 09:15:00 | 347.95 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-21 14:45:00 | 344.00 | 2025-03-24 09:15:00 | 347.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-03-27 12:45:00 | 344.50 | 2025-04-03 09:15:00 | 348.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-27 13:30:00 | 344.50 | 2025-04-03 09:15:00 | 348.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-28 11:15:00 | 339.65 | 2025-04-04 10:15:00 | 322.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:00:00 | 339.95 | 2025-04-04 10:15:00 | 322.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 14:00:00 | 339.50 | 2025-04-04 10:15:00 | 322.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 11:15:00 | 339.65 | 2025-04-07 09:15:00 | 305.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 13:00:00 | 339.95 | 2025-04-07 09:15:00 | 305.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 14:00:00 | 339.50 | 2025-04-07 09:15:00 | 305.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 337.00 | 2025-04-07 09:15:00 | 303.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-08 15:15:00 | 333.95 | 2025-05-14 09:15:00 | 337.10 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-09 09:45:00 | 330.80 | 2025-05-14 09:15:00 | 337.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-05-13 09:45:00 | 334.50 | 2025-05-14 09:15:00 | 337.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-15 10:15:00 | 334.45 | 2025-05-15 13:15:00 | 337.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-20 14:15:00 | 333.45 | 2025-05-21 09:15:00 | 339.10 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-05-22 11:00:00 | 334.00 | 2025-05-28 09:15:00 | 337.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-05-22 11:30:00 | 333.65 | 2025-05-28 09:15:00 | 337.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-05-22 13:30:00 | 333.00 | 2025-05-28 09:15:00 | 337.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-05-28 13:00:00 | 334.95 | 2025-06-03 09:15:00 | 336.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-05-29 09:30:00 | 334.85 | 2025-06-03 09:15:00 | 336.85 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-05-29 14:00:00 | 334.35 | 2025-06-03 09:15:00 | 336.85 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-05-30 09:30:00 | 334.50 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-06-02 09:15:00 | 332.75 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-06-02 10:45:00 | 333.30 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-06-02 13:45:00 | 333.80 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-04 09:30:00 | 333.60 | 2025-06-04 14:15:00 | 336.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-04 12:45:00 | 334.35 | 2025-06-04 14:15:00 | 336.45 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-06-05 09:15:00 | 334.00 | 2025-06-09 12:15:00 | 338.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-09 12:00:00 | 333.85 | 2025-06-09 12:15:00 | 338.65 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-08 09:30:00 | 367.20 | 2025-09-08 15:15:00 | 363.25 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-08 12:15:00 | 365.50 | 2025-09-08 15:15:00 | 363.25 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-08 13:30:00 | 365.85 | 2025-09-08 15:15:00 | 363.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-09-08 14:15:00 | 365.75 | 2025-09-08 15:15:00 | 363.25 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-10 13:15:00 | 381.85 | 2026-01-08 15:15:00 | 377.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-11 11:15:00 | 381.85 | 2026-01-08 15:15:00 | 377.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-11 15:15:00 | 382.45 | 2026-01-08 15:15:00 | 377.50 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-08 11:30:00 | 382.25 | 2026-01-08 15:15:00 | 377.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-12 12:00:00 | 374.95 | 2026-02-19 10:15:00 | 383.65 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-02-13 09:30:00 | 378.10 | 2026-02-19 10:15:00 | 383.65 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-13 14:30:00 | 378.80 | 2026-02-19 10:15:00 | 383.65 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-13 15:00:00 | 378.20 | 2026-02-19 10:15:00 | 383.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-02-17 14:30:00 | 377.80 | 2026-02-24 13:15:00 | 389.45 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-02-18 10:15:00 | 377.90 | 2026-02-24 13:15:00 | 389.45 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-02-18 12:15:00 | 376.60 | 2026-02-24 13:15:00 | 389.45 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-02-19 10:15:00 | 377.95 | 2026-02-24 13:15:00 | 389.45 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-16 09:30:00 | 385.95 | 2026-03-16 10:15:00 | 372.90 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2026-04-29 13:45:00 | 362.90 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-04-29 14:45:00 | 362.25 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-30 09:15:00 | 358.55 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2026-05-04 13:15:00 | 362.55 | 2026-05-05 15:15:00 | 370.00 | STOP_HIT | 1.00 | -2.05% |
