# Kotak Mahindra Bank Ltd. (KOTAKBANK)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 381.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 2 |
| ALERT3 | 76 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 85 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 80
- **Target hits / Stop hits / Partials:** 1 / 84 / 3
- **Avg / median % per leg:** -1.64% / -1.44%
- **Sum % (uncompounded):** -144.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 2 | 4.3% | 0 | 46 | 0 | -2.17% | -99.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 46 | 2 | 4.3% | 0 | 46 | 0 | -2.17% | -99.7% |
| SELL (all) | 42 | 6 | 14.3% | 1 | 38 | 3 | -1.07% | -45.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 6 | 14.3% | 1 | 38 | 3 | -1.07% | -45.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 88 | 8 | 9.1% | 1 | 84 | 3 | -1.64% | -144.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 360.35 | 371.89 | 371.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 15:15:00 | 359.60 | 371.77 | 371.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 13:15:00 | 359.64 | 359.40 | 363.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-08 14:00:00 | 359.64 | 359.40 | 363.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 360.81 | 359.53 | 363.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 15:15:00 | 359.10 | 360.54 | 363.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 10:00:00 | 357.35 | 360.49 | 363.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 10:15:00 | 341.14 | 352.04 | 355.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 13:15:00 | 339.48 | 351.69 | 355.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-07 13:15:00 | 350.40 | 349.11 | 353.16 | SL hit (close>ema200) qty=0.50 sl=349.11 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 10:15:00 | 366.40 | 353.75 | 353.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 368.15 | 354.23 | 353.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 367.60 | 368.65 | 363.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 10:15:00 | 367.00 | 368.65 | 363.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 365.78 | 368.46 | 363.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:45:00 | 365.40 | 368.46 | 363.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 364.02 | 368.41 | 363.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 363.99 | 368.41 | 363.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 364.61 | 368.38 | 363.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 11:30:00 | 364.43 | 368.38 | 363.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 364.42 | 368.34 | 363.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:45:00 | 364.27 | 368.34 | 363.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 13:15:00 | 364.31 | 368.30 | 363.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 13:30:00 | 364.27 | 368.30 | 363.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 363.98 | 368.25 | 363.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 363.98 | 368.25 | 363.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 363.97 | 368.11 | 363.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 363.97 | 368.11 | 363.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 364.39 | 368.07 | 363.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 12:30:00 | 365.82 | 367.94 | 363.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 362.59 | 368.15 | 364.29 | SL hit (close<static) qty=1.00 sl=363.07 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 347.48 | 362.01 | 362.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 13:15:00 | 345.21 | 361.57 | 361.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 350.55 | 349.26 | 353.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 10:00:00 | 350.55 | 349.26 | 353.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 353.02 | 349.37 | 353.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:00:00 | 353.02 | 349.37 | 353.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 345.57 | 349.43 | 353.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 13:30:00 | 345.21 | 349.29 | 353.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 10:30:00 | 343.98 | 349.15 | 353.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 13:30:00 | 344.33 | 349.01 | 353.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 11:15:00 | 344.94 | 348.88 | 353.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 348.09 | 348.53 | 352.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 10:30:00 | 346.98 | 348.54 | 352.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 12:15:00 | 352.64 | 348.61 | 352.16 | SL hit (close>static) qty=1.00 sl=352.29 alert=retest2 |

### Cycle 4 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 357.40 | 353.80 | 353.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 10:15:00 | 359.01 | 353.85 | 353.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 332.78 | 356.31 | 355.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 332.78 | 356.31 | 355.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 332.78 | 356.31 | 355.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:30:00 | 332.47 | 356.31 | 355.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 324.16 | 353.88 | 353.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 14:15:00 | 321.88 | 353.00 | 353.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 10:15:00 | 338.01 | 337.05 | 343.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 11:00:00 | 338.01 | 337.05 | 343.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 341.06 | 337.78 | 342.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:30:00 | 340.82 | 337.80 | 342.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 343.67 | 338.14 | 342.79 | SL hit (close>static) qty=1.00 sl=343.57 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 354.06 | 343.96 | 343.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 355.07 | 344.07 | 343.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 348.64 | 349.10 | 346.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 348.64 | 349.10 | 346.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 354.88 | 358.62 | 353.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:00:00 | 365.80 | 357.20 | 353.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 10:45:00 | 364.29 | 357.48 | 353.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:30:00 | 364.54 | 357.51 | 353.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 351.14 | 356.68 | 354.48 | SL hit (close<static) qty=1.00 sl=351.23 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 348.20 | 362.67 | 362.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 346.65 | 362.51 | 362.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 353.00 | 352.52 | 356.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 14:00:00 | 353.00 | 352.52 | 356.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 354.80 | 352.54 | 356.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 10:30:00 | 353.81 | 352.56 | 356.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 11:30:00 | 353.88 | 352.58 | 356.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 358.74 | 352.68 | 356.33 | SL hit (close>static) qty=1.00 sl=357.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 384.92 | 356.14 | 356.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 385.35 | 369.82 | 364.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 10:15:00 | 383.10 | 383.55 | 375.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-28 11:00:00 | 383.10 | 383.55 | 375.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 413.00 | 427.83 | 412.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 415.02 | 427.83 | 412.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 415.76 | 426.15 | 412.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 417.22 | 425.94 | 412.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:45:00 | 417.76 | 425.66 | 413.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 417.64 | 424.63 | 414.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 418.14 | 424.47 | 414.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 417.02 | 423.59 | 415.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:30:00 | 419.58 | 422.15 | 415.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 419.24 | 422.06 | 415.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:45:00 | 419.68 | 422.03 | 415.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:00:00 | 419.28 | 422.00 | 415.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 413.44 | 421.80 | 415.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 413.44 | 421.80 | 415.48 | SL hit (close<static) qty=1.00 sl=414.40 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.78 | 424.37 | 424.40 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.36 | 407.83 | 407.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.10 | 408.05 | 407.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.80 | 424.03 | 417.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 10:00:00 | 423.80 | 424.03 | 417.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 419.94 | 423.84 | 417.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 419.76 | 423.84 | 417.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 417.70 | 423.40 | 418.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 422.14 | 420.87 | 417.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 420.80 | 420.87 | 417.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 421.00 | 420.87 | 417.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 421.44 | 420.67 | 418.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 420.62 | 420.65 | 418.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 14:00:00 | 421.86 | 420.64 | 418.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 415.40 | 420.40 | 418.20 | SL hit (close<static) qty=1.00 sl=417.04 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.22 | 425.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 406.50 | 423.79 | 424.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 422.70 | 419.66 | 422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 420.10 | 419.66 | 422.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 13:15:00 | 419.00 | 419.66 | 422.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 429.65 | 419.82 | 422.13 | SL hit (close>static) qty=1.00 sl=423.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-18 15:15:00 | 359.10 | 2023-10-26 10:15:00 | 341.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-20 10:00:00 | 357.35 | 2023-10-26 13:15:00 | 339.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-18 15:15:00 | 359.10 | 2023-11-07 13:15:00 | 350.40 | STOP_HIT | 0.50 | 2.42% |
| SELL | retest2 | 2023-09-20 10:00:00 | 357.35 | 2023-11-07 13:15:00 | 350.40 | STOP_HIT | 0.50 | 1.94% |
| BUY | retest2 | 2024-01-12 12:30:00 | 365.82 | 2024-01-17 09:15:00 | 362.59 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-01-31 10:00:00 | 365.82 | 2024-02-05 10:15:00 | 361.69 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-01-31 11:15:00 | 365.46 | 2024-02-05 10:15:00 | 361.69 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-01-31 13:45:00 | 365.65 | 2024-02-05 10:15:00 | 361.69 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-03-11 13:30:00 | 345.21 | 2024-03-20 12:15:00 | 352.64 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-03-12 10:30:00 | 343.98 | 2024-03-21 09:15:00 | 357.49 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2024-03-12 13:30:00 | 344.33 | 2024-03-21 09:15:00 | 357.49 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2024-03-13 11:15:00 | 344.94 | 2024-03-21 09:15:00 | 357.49 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-03-20 10:30:00 | 346.98 | 2024-03-21 09:15:00 | 357.49 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-04-03 12:30:00 | 347.12 | 2024-04-05 10:15:00 | 352.38 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-04-04 10:00:00 | 347.32 | 2024-04-05 10:15:00 | 352.38 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-04-04 11:30:00 | 347.38 | 2024-04-05 10:15:00 | 352.38 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-05-24 10:30:00 | 340.82 | 2024-05-27 12:15:00 | 343.67 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-05-28 14:30:00 | 340.92 | 2024-06-03 11:15:00 | 343.98 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-05-29 09:30:00 | 339.83 | 2024-06-03 11:15:00 | 343.98 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-06-03 09:30:00 | 340.93 | 2024-06-03 11:15:00 | 343.98 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-06-04 09:15:00 | 332.90 | 2024-06-05 14:15:00 | 344.44 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-06-05 11:15:00 | 340.58 | 2024-06-05 14:15:00 | 344.44 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-06-05 13:00:00 | 340.00 | 2024-06-05 14:15:00 | 344.44 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-06-06 09:45:00 | 341.02 | 2024-06-06 10:15:00 | 346.88 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-07-26 13:00:00 | 365.80 | 2024-08-13 13:15:00 | 351.14 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2024-07-29 10:45:00 | 364.29 | 2024-08-13 13:15:00 | 351.14 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2024-07-29 11:30:00 | 364.54 | 2024-08-13 13:15:00 | 351.14 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2024-08-22 15:00:00 | 364.46 | 2024-09-04 09:15:00 | 353.20 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2024-08-30 11:00:00 | 356.68 | 2024-09-04 09:15:00 | 353.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-02 09:15:00 | 357.16 | 2024-09-04 09:15:00 | 353.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-09-02 14:00:00 | 356.50 | 2024-09-04 09:15:00 | 353.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-09-03 10:00:00 | 356.75 | 2024-09-04 09:15:00 | 353.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-03 13:15:00 | 356.61 | 2024-09-04 09:15:00 | 353.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-09-03 14:30:00 | 356.69 | 2024-09-06 09:15:00 | 353.01 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-09-04 13:45:00 | 356.73 | 2024-10-04 12:15:00 | 363.02 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2024-09-09 13:15:00 | 356.50 | 2024-10-07 11:15:00 | 361.72 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2024-10-04 10:30:00 | 366.40 | 2024-10-21 09:15:00 | 352.68 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2024-10-07 09:15:00 | 366.02 | 2024-10-21 09:15:00 | 352.68 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2024-10-10 10:00:00 | 366.89 | 2024-10-21 09:15:00 | 352.68 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2024-10-10 10:45:00 | 366.82 | 2024-10-21 10:15:00 | 349.23 | STOP_HIT | 1.00 | -4.80% |
| SELL | retest2 | 2024-11-25 10:30:00 | 353.81 | 2024-11-25 14:15:00 | 358.74 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-11-25 11:30:00 | 353.88 | 2024-11-25 14:15:00 | 358.74 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-11-28 10:30:00 | 354.01 | 2024-12-06 13:15:00 | 356.64 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-11-28 11:15:00 | 353.31 | 2024-12-06 13:15:00 | 356.64 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-06 09:15:00 | 354.91 | 2024-12-09 09:15:00 | 357.40 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-06 10:45:00 | 354.80 | 2024-12-09 09:15:00 | 357.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-12-06 15:15:00 | 355.05 | 2024-12-09 09:15:00 | 357.40 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-12-12 12:30:00 | 354.91 | 2024-12-13 13:15:00 | 358.13 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-12-18 14:15:00 | 354.90 | 2024-12-31 12:15:00 | 358.23 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-19 09:15:00 | 350.28 | 2024-12-31 12:15:00 | 358.23 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-12-30 12:00:00 | 354.58 | 2024-12-31 12:15:00 | 358.23 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-07 15:00:00 | 354.20 | 2025-01-09 09:15:00 | 359.91 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-10 14:00:00 | 352.76 | 2025-01-15 09:15:00 | 356.29 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-01-14 14:15:00 | 352.69 | 2025-01-15 09:15:00 | 356.29 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-01-17 10:45:00 | 352.44 | 2025-01-20 09:15:00 | 384.04 | STOP_HIT | 1.00 | -8.97% |
| SELL | retest2 | 2025-01-17 13:00:00 | 351.76 | 2025-01-20 09:15:00 | 384.04 | STOP_HIT | 1.00 | -9.18% |
| BUY | retest2 | 2025-05-07 11:30:00 | 417.22 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-05-07 14:45:00 | 417.76 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-05-15 10:30:00 | 417.64 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-15 13:00:00 | 418.14 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-23 12:30:00 | 419.58 | 2025-05-28 11:15:00 | 413.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-05-26 10:45:00 | 419.24 | 2025-06-02 09:15:00 | 412.56 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-26 11:45:00 | 419.68 | 2025-06-03 09:15:00 | 408.70 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-05-26 13:00:00 | 419.28 | 2025-06-03 09:15:00 | 408.70 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-27 11:15:00 | 419.06 | 2025-06-03 09:15:00 | 408.70 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-05-29 15:00:00 | 417.40 | 2025-06-03 09:15:00 | 408.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-09 09:15:00 | 423.28 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -5.95% |
| BUY | retest2 | 2025-11-17 09:30:00 | 422.14 | 2025-11-25 09:15:00 | 415.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-11-17 10:30:00 | 420.80 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-17 12:00:00 | 421.00 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-11-19 14:00:00 | 421.44 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-21 14:00:00 | 421.86 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-11-27 11:00:00 | 423.68 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-14 10:45:00 | 423.90 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-16 09:15:00 | 425.00 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-01-20 09:15:00 | 429.80 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-20 11:30:00 | 428.10 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest2 | 2026-01-20 12:15:00 | 429.50 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest2 | 2026-01-20 12:45:00 | 427.90 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-01-22 14:15:00 | 424.90 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-01-23 12:30:00 | 424.00 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-01-23 15:15:00 | 424.50 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-02-06 13:15:00 | 419.00 | 2026-02-09 09:15:00 | 429.65 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-02-19 12:45:00 | 419.40 | 2026-02-23 09:15:00 | 427.30 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-02-19 13:15:00 | 419.05 | 2026-02-23 09:15:00 | 427.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-02-19 14:15:00 | 418.75 | 2026-02-23 09:15:00 | 427.30 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-02-27 09:15:00 | 420.35 | 2026-03-06 14:15:00 | 399.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 420.35 | 2026-03-12 09:15:00 | 378.32 | TARGET_HIT | 0.50 | 10.00% |
