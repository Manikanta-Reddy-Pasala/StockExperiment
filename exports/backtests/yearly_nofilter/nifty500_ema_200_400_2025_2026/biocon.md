# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 378.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 32
- **Target hits / Stop hits / Partials:** 0 / 34 / 0
- **Avg / median % per leg:** -1.66% / -1.29%
- **Sum % (uncompounded):** -56.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.25% | -11.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.25% | -11.3% |
| SELL (all) | 25 | 2 | 8.0% | 0 | 25 | 0 | -1.80% | -45.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 2 | 8.0% | 0 | 25 | 0 | -1.80% | -45.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 2 | 5.9% | 0 | 34 | 0 | -1.66% | -56.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 14:15:00)

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

### Cycle 2 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 361.10 | 362.59 | 362.60 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-18 13:15:00)

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

### Cycle 4 — SELL (started 2025-09-25 11:15:00)

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

### Cycle 5 — BUY (started 2025-10-31 09:15:00)

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

### Cycle 6 — SELL (started 2026-01-22 10:15:00)

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

### Cycle 7 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 396.20 | 379.49 | 379.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 400.90 | 383.45 | 381.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 384.05 | 384.84 | 382.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:00:00 | 384.05 | 384.84 | 382.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 385.00 | 384.90 | 382.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 385.95 | 384.87 | 382.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 372.90 | 384.75 | 382.56 | SL hit (close<static) qty=1.00 sl=382.60 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-24 15:15:00)

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
| SELL | retest2 | 2025-05-16 10:00:00 | 339.90 | 2025-06-03 09:15:00 | 336.85 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-05-16 13:15:00 | 339.55 | 2025-06-03 09:15:00 | 336.85 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-05-28 13:00:00 | 334.95 | 2025-06-03 09:15:00 | 336.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-05-29 09:30:00 | 334.85 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-05-29 14:00:00 | 334.35 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-05-30 09:30:00 | 334.50 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-06-02 09:15:00 | 332.75 | 2025-06-03 11:15:00 | 338.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-06-02 10:45:00 | 333.30 | 2025-06-04 14:15:00 | 336.45 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-06-02 13:45:00 | 333.80 | 2025-06-04 14:15:00 | 336.45 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-04 09:30:00 | 333.60 | 2025-06-09 12:15:00 | 338.65 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-06-04 12:45:00 | 334.35 | 2025-06-09 12:15:00 | 338.65 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-06-05 09:15:00 | 334.00 | 2025-06-10 09:15:00 | 346.80 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-06-09 12:00:00 | 333.85 | 2025-06-10 09:15:00 | 346.80 | STOP_HIT | 1.00 | -3.88% |
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
