# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 336.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 44 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 42
- **Target hits / Stop hits / Partials:** 5 / 43 / 4
- **Avg / median % per leg:** -1.21% / -2.85%
- **Sum % (uncompounded):** -63.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 1 | 3.8% | 1 | 25 | 0 | -2.58% | -67.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 1 | 3.8% | 1 | 25 | 0 | -2.58% | -67.2% |
| SELL (all) | 26 | 9 | 34.6% | 4 | 18 | 4 | 0.16% | 4.1% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.38% | -21.5% |
| SELL @ 3rd Alert (retest2) | 22 | 9 | 40.9% | 4 | 14 | 4 | 1.16% | 25.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.38% | -21.5% |
| retest2 (combined) | 48 | 10 | 20.8% | 5 | 39 | 4 | -0.87% | -41.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 14:15:00 | 430.50 | 438.56 | 438.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 427.25 | 438.37 | 438.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 15:15:00 | 425.95 | 424.12 | 429.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:15:00 | 423.60 | 424.12 | 429.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 431.45 | 424.23 | 429.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:45:00 | 432.85 | 424.23 | 429.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 435.00 | 424.34 | 429.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 15:00:00 | 435.00 | 424.34 | 429.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 428.50 | 426.52 | 430.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:30:00 | 423.90 | 426.54 | 430.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 10:15:00 | 425.45 | 426.54 | 430.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 13:15:00 | 402.70 | 424.39 | 428.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 13:15:00 | 404.18 | 424.39 | 428.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-07 09:15:00 | 381.51 | 423.18 | 428.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 14:15:00 | 407.75 | 397.04 | 397.03 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 377.80 | 396.97 | 397.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 375.15 | 396.75 | 396.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 369.95 | 368.99 | 377.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 369.95 | 368.99 | 377.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 369.70 | 369.05 | 377.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 368.45 | 369.05 | 377.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 379.25 | 369.18 | 377.55 | SL hit (close>static) qty=1.00 sl=378.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 427.70 | 382.84 | 382.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 12:15:00 | 433.90 | 383.35 | 383.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 400.20 | 402.36 | 394.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 15:00:00 | 400.20 | 402.36 | 394.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 397.75 | 402.34 | 395.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:00:00 | 407.75 | 402.32 | 395.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:30:00 | 405.60 | 409.46 | 400.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 10:00:00 | 404.50 | 409.40 | 401.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 392.80 | 408.20 | 401.04 | SL hit (close<static) qty=1.00 sl=394.85 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 353.05 | 396.84 | 396.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 337.50 | 382.88 | 388.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 345.85 | 345.54 | 362.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 11:15:00 | 341.90 | 345.52 | 362.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 09:45:00 | 341.15 | 345.24 | 362.01 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 13:45:00 | 341.50 | 345.16 | 361.64 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 15:00:00 | 340.05 | 345.11 | 361.53 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 359.50 | 344.59 | 359.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-13 14:15:00 | 359.50 | 344.59 | 359.09 | SL hit (close>ema400) qty=1.00 sl=359.09 alert=retest1 |

### Cycle 6 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 387.70 | 364.35 | 364.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 389.30 | 364.60 | 364.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 368.00 | 369.26 | 367.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 368.80 | 369.26 | 367.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 368.80 | 369.26 | 367.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:45:00 | 373.85 | 366.18 | 365.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 14:15:00 | 374.10 | 366.18 | 365.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 374.00 | 366.43 | 365.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 374.00 | 368.08 | 366.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 369.45 | 369.35 | 367.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:45:00 | 368.30 | 369.35 | 367.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 370.00 | 369.36 | 367.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 363.35 | 369.12 | 367.71 | SL hit (close<static) qty=1.00 sl=363.85 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 345.25 | 367.84 | 367.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 344.90 | 367.20 | 367.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 365.30 | 363.30 | 365.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 363.45 | 363.31 | 365.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 362.60 | 363.30 | 365.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 12:45:00 | 361.60 | 363.39 | 365.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 362.65 | 363.83 | 365.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 373.65 | 364.63 | 365.65 | SL hit (close>static) qty=1.00 sl=370.90 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 380.25 | 366.57 | 366.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 15:15:00 | 382.20 | 366.72 | 366.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 373.85 | 374.57 | 371.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 373.85 | 374.57 | 371.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 377.85 | 375.24 | 371.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 379.65 | 375.32 | 372.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 380.40 | 375.32 | 372.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 381.60 | 375.42 | 372.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:00:00 | 379.65 | 375.63 | 372.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 372.95 | 375.64 | 372.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 372.95 | 375.64 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 372.10 | 375.61 | 372.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 370.90 | 375.61 | 372.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 373.05 | 375.58 | 372.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 372.10 | 375.58 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 374.60 | 375.57 | 372.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:30:00 | 379.00 | 375.58 | 372.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:15:00 | 375.10 | 375.55 | 372.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 363.40 | 375.36 | 372.48 | SL hit (close<static) qty=1.00 sl=371.20 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 377.30 | 386.51 | 386.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 375.35 | 386.31 | 386.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 380.20 | 379.68 | 382.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 380.00 | 379.69 | 382.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 386.45 | 379.69 | 382.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 388.25 | 379.77 | 382.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 389.30 | 379.77 | 382.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 385.55 | 379.83 | 382.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 384.85 | 379.83 | 382.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 384.95 | 379.88 | 382.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 14:15:00 | 365.70 | 377.85 | 380.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 15:15:00 | 365.61 | 377.73 | 380.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 14:15:00 | 346.37 | 366.77 | 372.76 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-02 09:30:00 | 423.90 | 2024-08-06 13:15:00 | 402.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 10:15:00 | 425.45 | 2024-08-06 13:15:00 | 404.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 09:30:00 | 423.90 | 2024-08-07 09:15:00 | 381.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-02 10:15:00 | 425.45 | 2024-08-07 09:15:00 | 382.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-11 10:00:00 | 425.35 | 2024-10-11 10:15:00 | 432.15 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-10-15 09:15:00 | 424.60 | 2024-10-17 14:15:00 | 407.75 | STOP_HIT | 1.00 | 3.97% |
| SELL | retest2 | 2024-11-27 12:15:00 | 368.45 | 2024-11-28 09:15:00 | 379.25 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-12-31 13:00:00 | 407.75 | 2025-01-14 14:15:00 | 392.80 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-01-10 10:30:00 | 405.60 | 2025-01-14 14:15:00 | 392.80 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-01-13 10:00:00 | 404.50 | 2025-01-14 14:15:00 | 392.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-01-16 09:15:00 | 402.00 | 2025-01-22 09:15:00 | 395.40 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-01-20 10:45:00 | 404.10 | 2025-01-22 09:15:00 | 395.40 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-01-20 11:45:00 | 404.55 | 2025-01-22 09:15:00 | 395.40 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-01-20 13:15:00 | 404.70 | 2025-01-22 09:15:00 | 395.40 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-01-20 14:15:00 | 404.25 | 2025-01-23 09:15:00 | 392.40 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest1 | 2025-03-06 11:15:00 | 341.90 | 2025-03-13 14:15:00 | 359.50 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest1 | 2025-03-07 09:45:00 | 341.15 | 2025-03-13 14:15:00 | 359.50 | STOP_HIT | 1.00 | -5.38% |
| SELL | retest1 | 2025-03-07 13:45:00 | 341.50 | 2025-03-13 14:15:00 | 359.50 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest1 | 2025-03-07 15:00:00 | 340.05 | 2025-03-13 14:15:00 | 359.50 | STOP_HIT | 1.00 | -5.72% |
| SELL | retest2 | 2025-03-17 09:15:00 | 355.50 | 2025-03-17 09:15:00 | 362.80 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-03-17 11:45:00 | 356.00 | 2025-03-17 13:15:00 | 360.85 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-03-28 12:45:00 | 354.85 | 2025-04-01 13:15:00 | 362.65 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-04-01 10:00:00 | 355.15 | 2025-04-01 13:15:00 | 362.65 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-04-09 09:15:00 | 353.30 | 2025-04-15 09:15:00 | 368.00 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-04-09 09:45:00 | 353.15 | 2025-04-15 09:15:00 | 368.00 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-04-09 13:45:00 | 353.50 | 2025-04-15 09:15:00 | 368.00 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-04-09 15:00:00 | 353.10 | 2025-04-15 09:15:00 | 368.00 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2025-05-16 13:45:00 | 373.85 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-05-16 14:15:00 | 374.10 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-05-19 10:15:00 | 374.00 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-05-21 11:45:00 | 374.00 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-05-30 13:15:00 | 372.05 | 2025-06-12 13:15:00 | 365.25 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-30 15:15:00 | 371.30 | 2025-06-12 13:15:00 | 365.25 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-26 12:00:00 | 362.60 | 2025-07-08 09:15:00 | 373.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-06-27 12:45:00 | 361.60 | 2025-07-08 09:15:00 | 373.65 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-07-02 13:15:00 | 362.65 | 2025-07-08 09:15:00 | 373.65 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-07-31 11:30:00 | 379.65 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-07-31 12:00:00 | 380.40 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-08-01 09:15:00 | 381.60 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2025-08-01 13:00:00 | 379.65 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-08-04 14:30:00 | 379.00 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-08-05 12:15:00 | 375.10 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-08-13 11:45:00 | 375.15 | 2025-08-13 13:15:00 | 412.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 09:45:00 | 376.05 | 2025-09-29 09:15:00 | 371.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-08 11:45:00 | 392.05 | 2025-10-13 11:15:00 | 382.05 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-10-10 09:15:00 | 393.00 | 2025-10-13 11:15:00 | 382.05 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-10-28 14:15:00 | 392.15 | 2025-11-12 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2025-10-30 11:00:00 | 392.50 | 2025-11-12 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2025-12-10 11:15:00 | 384.85 | 2025-12-17 14:15:00 | 365.70 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-12-10 11:45:00 | 384.95 | 2025-12-17 15:15:00 | 365.61 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-12-10 11:15:00 | 384.85 | 2026-01-09 14:15:00 | 346.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-10 11:45:00 | 384.95 | 2026-01-09 14:15:00 | 346.45 | TARGET_HIT | 0.50 | 10.00% |
