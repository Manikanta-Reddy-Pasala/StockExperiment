# Nuvoco Vistas Corporation Ltd. (NUVOCO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 328.90
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
| ALERT2_SKIP | 8 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 52 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 46
- **Target hits / Stop hits / Partials:** 4 / 53 / 6
- **Avg / median % per leg:** -0.89% / -2.05%
- **Sum % (uncompounded):** -55.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 3 | 9.4% | 3 | 29 | 0 | -1.08% | -34.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 3 | 9.4% | 3 | 29 | 0 | -1.08% | -34.6% |
| SELL (all) | 31 | 14 | 45.2% | 1 | 24 | 6 | -0.69% | -21.3% |
| SELL @ 2nd Alert (retest1) | 9 | 8 | 88.9% | 0 | 5 | 4 | 3.18% | 28.6% |
| SELL @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 1 | 19 | 2 | -2.27% | -49.9% |
| retest1 (combined) | 9 | 8 | 88.9% | 0 | 5 | 4 | 3.18% | 28.6% |
| retest2 (combined) | 54 | 9 | 16.7% | 4 | 48 | 2 | -1.56% | -84.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 340.75 | 348.02 | 348.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 10:15:00 | 338.70 | 347.14 | 347.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 09:15:00 | 350.35 | 346.32 | 347.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 350.35 | 346.32 | 347.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 350.35 | 346.32 | 347.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 11:30:00 | 346.10 | 346.36 | 347.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 13:15:00 | 357.20 | 346.51 | 347.20 | SL hit (close>static) qty=1.00 sl=353.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 13:15:00 | 373.55 | 347.91 | 347.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 10:15:00 | 377.55 | 348.97 | 348.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 09:15:00 | 367.00 | 367.24 | 360.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-05 09:45:00 | 367.25 | 367.24 | 360.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 361.75 | 367.33 | 361.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 365.20 | 366.78 | 361.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 14:15:00 | 354.90 | 366.32 | 360.96 | SL hit (close<static) qty=1.00 sl=357.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 342.00 | 358.80 | 358.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 15:15:00 | 340.00 | 358.29 | 358.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 13:15:00 | 354.60 | 350.56 | 353.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 13:15:00 | 354.60 | 350.56 | 353.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 13:15:00 | 354.60 | 350.56 | 353.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:00:00 | 354.60 | 350.56 | 353.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 352.45 | 350.58 | 353.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 14:45:00 | 354.10 | 350.58 | 353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 353.10 | 350.60 | 353.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:15:00 | 355.20 | 350.60 | 353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 354.15 | 350.64 | 353.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:30:00 | 356.60 | 350.64 | 353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 353.70 | 350.67 | 353.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 11:15:00 | 352.55 | 350.67 | 353.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 10:15:00 | 352.45 | 350.70 | 353.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 363.60 | 350.88 | 353.41 | SL hit (close>static) qty=1.00 sl=354.90 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 13:15:00 | 371.10 | 355.60 | 355.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 374.65 | 356.09 | 355.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 365.00 | 365.40 | 361.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 14:15:00 | 365.00 | 365.40 | 361.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 365.00 | 365.40 | 361.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:30:00 | 362.90 | 365.40 | 361.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 368.40 | 365.43 | 361.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 13:30:00 | 373.30 | 365.48 | 361.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:30:00 | 369.60 | 371.10 | 365.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 11:15:00 | 370.00 | 371.08 | 365.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 12:15:00 | 370.25 | 371.05 | 365.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 12:15:00 | 365.95 | 370.88 | 366.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 12:45:00 | 365.00 | 370.88 | 366.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 363.80 | 370.81 | 366.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 363.80 | 370.81 | 366.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 367.10 | 370.77 | 366.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 368.45 | 370.74 | 366.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 10:15:00 | 362.40 | 370.59 | 366.03 | SL hit (close<static) qty=1.00 sl=363.05 alert=retest2 |

### Cycle 5 — SELL (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 15:15:00 | 342.00 | 362.64 | 362.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 340.45 | 362.42 | 362.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 09:15:00 | 355.30 | 354.70 | 358.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 14:15:00 | 351.70 | 354.66 | 358.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 357.30 | 354.67 | 358.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-31 09:15:00 | 361.70 | 354.77 | 358.10 | SL hit (close>ema400) qty=1.00 sl=358.10 alert=retest1 |

### Cycle 6 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 11:15:00 | 350.00 | 326.57 | 326.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 363.70 | 327.79 | 327.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 349.75 | 351.49 | 343.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 349.75 | 351.49 | 343.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 341.30 | 351.34 | 343.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 341.30 | 351.34 | 343.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 338.00 | 351.21 | 343.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:45:00 | 338.25 | 351.21 | 343.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 341.55 | 349.77 | 343.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 340.50 | 349.77 | 343.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 342.50 | 349.70 | 343.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 342.50 | 349.70 | 343.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 341.00 | 349.31 | 343.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 342.60 | 349.31 | 343.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 345.25 | 349.20 | 343.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:45:00 | 347.55 | 348.55 | 343.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 12:15:00 | 348.25 | 348.55 | 343.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:30:00 | 347.30 | 349.48 | 344.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:00:00 | 348.15 | 349.48 | 344.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 345.00 | 349.40 | 344.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:00:00 | 345.00 | 349.40 | 344.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 346.05 | 349.37 | 344.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 347.15 | 349.37 | 344.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 340.05 | 349.25 | 344.74 | SL hit (close<static) qty=1.00 sl=341.35 alert=retest2 |

### Cycle 7 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 330.95 | 341.37 | 341.41 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 14:15:00 | 350.15 | 341.16 | 341.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 354.35 | 342.32 | 341.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 350.75 | 353.01 | 348.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 350.75 | 353.01 | 348.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 350.75 | 353.01 | 348.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:45:00 | 349.05 | 353.01 | 348.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 351.35 | 353.00 | 348.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:45:00 | 350.00 | 353.00 | 348.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 353.15 | 352.93 | 349.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 10:00:00 | 358.70 | 352.99 | 349.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 356.85 | 353.12 | 349.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 357.50 | 353.15 | 349.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 10:00:00 | 358.85 | 353.21 | 349.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 350.10 | 353.71 | 350.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 350.10 | 353.71 | 350.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 351.00 | 353.68 | 350.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 350.75 | 353.68 | 350.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 351.50 | 353.66 | 350.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 351.25 | 353.66 | 350.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 349.80 | 353.62 | 350.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 349.80 | 353.62 | 350.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 350.00 | 353.59 | 350.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:30:00 | 350.05 | 353.59 | 350.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 349.10 | 353.54 | 350.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 348.70 | 353.54 | 350.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 349.75 | 353.51 | 350.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 348.30 | 353.51 | 350.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 347.50 | 353.45 | 350.09 | SL hit (close<static) qty=1.00 sl=348.30 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 344.05 | 348.26 | 348.27 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 12:15:00 | 354.00 | 348.28 | 348.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 15:15:00 | 354.75 | 348.42 | 348.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 349.15 | 350.20 | 349.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 11:15:00 | 349.15 | 350.20 | 349.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 349.15 | 350.20 | 349.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 349.15 | 350.20 | 349.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 348.15 | 350.18 | 349.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:15:00 | 351.30 | 350.17 | 349.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 347.00 | 350.15 | 349.33 | SL hit (close<static) qty=1.00 sl=347.20 alert=retest2 |

### Cycle 11 — SELL (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 13:15:00 | 332.95 | 348.51 | 348.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 328.20 | 348.31 | 348.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 346.25 | 344.49 | 346.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 346.25 | 344.49 | 346.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 346.25 | 344.49 | 346.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 346.25 | 344.49 | 346.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 345.40 | 344.50 | 346.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 342.60 | 344.50 | 346.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 343.90 | 344.13 | 345.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 345.20 | 344.15 | 345.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 14:30:00 | 345.35 | 344.17 | 345.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 344.00 | 344.17 | 345.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 346.00 | 344.17 | 345.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 345.95 | 344.19 | 345.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 348.70 | 344.23 | 345.94 | SL hit (close>static) qty=1.00 sl=347.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 367.95 | 347.44 | 347.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 10:15:00 | 369.45 | 348.78 | 348.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 356.50 | 356.97 | 353.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 12:00:00 | 356.50 | 356.97 | 353.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 355.45 | 356.97 | 353.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 357.90 | 356.91 | 353.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 358.35 | 356.93 | 353.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 352.70 | 356.98 | 353.44 | SL hit (close<static) qty=1.00 sl=352.95 alert=retest2 |

### Cycle 13 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 350.70 | 352.16 | 352.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 340.45 | 351.93 | 352.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 352.95 | 351.94 | 352.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 352.95 | 351.94 | 352.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 352.95 | 351.94 | 352.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 352.95 | 351.94 | 352.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 352.95 | 351.95 | 352.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 335.30 | 351.95 | 352.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 347.60 | 351.51 | 351.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 12:30:00 | 348.05 | 351.31 | 351.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 09:45:00 | 348.30 | 349.62 | 350.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 350.45 | 349.45 | 350.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 344.00 | 349.40 | 350.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 353.00 | 349.21 | 350.44 | SL hit (close>static) qty=1.00 sl=351.20 alert=retest2 |

### Cycle 14 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 341.55 | 325.56 | 325.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 345.20 | 326.79 | 326.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 13:15:00 | 327.40 | 327.87 | 326.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 13:15:00 | 327.40 | 327.87 | 326.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 327.40 | 327.87 | 326.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 327.40 | 327.87 | 326.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 344.30 | 350.49 | 344.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 344.30 | 350.49 | 344.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 341.05 | 350.40 | 344.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 341.05 | 350.40 | 344.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 340.80 | 350.31 | 344.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:30:00 | 340.90 | 350.31 | 344.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 338.00 | 349.63 | 343.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 339.95 | 349.41 | 343.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 339.90 | 349.11 | 343.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 341.05 | 348.62 | 343.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 09:15:00 | 373.94 | 355.20 | 349.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 398.50 | 423.47 | 423.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 391.05 | 422.89 | 423.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 363.90 | 363.74 | 382.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:30:00 | 361.00 | 363.69 | 382.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 15:00:00 | 359.95 | 363.64 | 381.70 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 14:30:00 | 361.75 | 362.56 | 376.10 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 15:00:00 | 361.55 | 362.56 | 376.10 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 342.95 | 358.90 | 370.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 343.66 | 358.90 | 370.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 343.47 | 358.90 | 370.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 341.95 | 358.02 | 369.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 350.75 | 349.19 | 357.76 | SL hit (close>ema200) qty=0.50 sl=349.19 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-04 11:30:00 | 346.10 | 2023-09-04 13:15:00 | 357.20 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2023-10-10 09:15:00 | 365.20 | 2023-10-10 14:15:00 | 354.90 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2023-10-11 15:00:00 | 366.10 | 2023-10-19 09:15:00 | 360.55 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2023-10-12 09:15:00 | 365.60 | 2023-10-19 09:15:00 | 360.55 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-10-12 12:15:00 | 365.05 | 2023-10-19 09:15:00 | 360.55 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-10-13 11:00:00 | 370.20 | 2023-10-19 09:15:00 | 360.55 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2023-10-16 09:15:00 | 369.85 | 2023-10-20 12:15:00 | 356.80 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2023-10-16 12:45:00 | 367.00 | 2023-10-20 12:15:00 | 356.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2023-10-17 12:45:00 | 366.70 | 2023-10-20 12:15:00 | 356.80 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2023-10-19 11:30:00 | 362.35 | 2023-10-20 12:15:00 | 356.80 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2023-10-19 12:00:00 | 362.40 | 2023-10-20 12:15:00 | 356.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-11-17 11:15:00 | 352.55 | 2023-11-24 09:15:00 | 363.60 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2023-11-20 10:15:00 | 352.45 | 2023-11-24 09:15:00 | 363.60 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2023-12-21 13:30:00 | 373.30 | 2024-01-08 10:15:00 | 362.40 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-01-04 09:30:00 | 369.60 | 2024-01-09 11:15:00 | 360.00 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-01-04 11:15:00 | 370.00 | 2024-01-09 11:15:00 | 360.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-01-04 12:15:00 | 370.25 | 2024-01-09 11:15:00 | 360.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-01-08 09:15:00 | 368.45 | 2024-01-09 11:15:00 | 360.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest1 | 2024-01-29 14:15:00 | 351.70 | 2024-01-31 09:15:00 | 361.70 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-02-09 09:30:00 | 353.70 | 2024-02-14 09:15:00 | 336.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-09 09:30:00 | 353.70 | 2024-02-20 09:15:00 | 351.65 | STOP_HIT | 0.50 | 0.58% |
| BUY | retest2 | 2024-07-26 11:45:00 | 347.55 | 2024-08-02 09:15:00 | 340.05 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-07-26 12:15:00 | 348.25 | 2024-08-02 09:15:00 | 340.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-08-01 10:30:00 | 347.30 | 2024-08-02 09:15:00 | 340.05 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-08-01 11:00:00 | 348.15 | 2024-08-02 09:15:00 | 340.05 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-08-01 15:15:00 | 347.15 | 2024-08-02 09:15:00 | 340.05 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-10-09 10:00:00 | 358.70 | 2024-10-16 09:15:00 | 347.50 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-10-09 15:00:00 | 356.85 | 2024-10-16 09:15:00 | 347.50 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-10-10 09:15:00 | 357.50 | 2024-10-16 09:15:00 | 347.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-10-10 10:00:00 | 358.85 | 2024-10-16 09:15:00 | 347.50 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-11-08 14:15:00 | 351.30 | 2024-11-11 09:15:00 | 347.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-11-26 09:15:00 | 342.60 | 2024-11-29 10:15:00 | 348.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-11-28 10:30:00 | 343.90 | 2024-11-29 10:15:00 | 348.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-11-28 12:30:00 | 345.20 | 2024-11-29 10:15:00 | 348.70 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-11-28 14:30:00 | 345.35 | 2024-11-29 10:15:00 | 348.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-12-19 15:15:00 | 357.90 | 2024-12-20 13:15:00 | 352.70 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-20 09:30:00 | 358.35 | 2024-12-20 13:15:00 | 352.70 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-01-03 09:15:00 | 357.60 | 2025-01-03 15:15:00 | 352.50 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-01-03 09:45:00 | 357.35 | 2025-01-03 15:15:00 | 352.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-01-23 09:15:00 | 335.30 | 2025-02-03 10:15:00 | 353.00 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-01-27 09:15:00 | 347.60 | 2025-02-05 12:15:00 | 357.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-01-27 12:30:00 | 348.05 | 2025-02-05 12:15:00 | 357.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-01-30 09:45:00 | 348.30 | 2025-02-05 12:15:00 | 357.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-01 11:45:00 | 344.00 | 2025-02-05 12:15:00 | 357.00 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-02-11 09:15:00 | 343.10 | 2025-02-12 09:15:00 | 325.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 343.10 | 2025-02-14 10:15:00 | 308.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-02 09:30:00 | 346.55 | 2025-05-07 09:15:00 | 341.55 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2025-05-05 12:30:00 | 346.95 | 2025-05-07 09:15:00 | 341.55 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-06-20 15:15:00 | 339.95 | 2025-07-16 09:15:00 | 373.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 10:30:00 | 339.90 | 2025-07-16 09:15:00 | 373.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 09:15:00 | 341.05 | 2025-07-16 09:15:00 | 375.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-12-15 11:30:00 | 361.00 | 2026-01-08 15:15:00 | 342.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-15 15:00:00 | 359.95 | 2026-01-08 15:15:00 | 343.66 | PARTIAL | 0.50 | 4.52% |
| SELL | retest1 | 2025-12-29 14:30:00 | 361.75 | 2026-01-08 15:15:00 | 343.47 | PARTIAL | 0.50 | 5.05% |
| SELL | retest1 | 2025-12-29 15:00:00 | 361.55 | 2026-01-12 09:15:00 | 341.95 | PARTIAL | 0.50 | 5.42% |
| SELL | retest1 | 2025-12-15 11:30:00 | 361.00 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest1 | 2025-12-15 15:00:00 | 359.95 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest1 | 2025-12-29 14:30:00 | 361.75 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest1 | 2025-12-29 15:00:00 | 361.55 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2026-04-22 10:00:00 | 297.00 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -10.20% |
| SELL | retest2 | 2026-04-24 09:15:00 | 294.20 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -11.25% |
| SELL | retest2 | 2026-04-30 13:00:00 | 297.00 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -10.20% |
| SELL | retest2 | 2026-04-30 13:45:00 | 296.75 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -10.29% |
