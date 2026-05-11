# Firstsource Solutions Ltd. (FSL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 272.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT2_SKIP | 7 |
| ALERT3 | 64 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 31
- **Target hits / Stop hits / Partials:** 8 / 35 / 13
- **Avg / median % per leg:** 1.12% / -0.42%
- **Sum % (uncompounded):** 62.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 1 | 7.1% | 1 | 13 | 0 | -1.46% | -20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 1 | 7.1% | 1 | 13 | 0 | -1.46% | -20.5% |
| SELL (all) | 42 | 24 | 57.1% | 7 | 22 | 13 | 1.99% | 83.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 24 | 57.1% | 7 | 22 | 13 | 1.99% | 83.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 56 | 25 | 44.6% | 8 | 35 | 13 | 1.12% | 63.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 193.60 | 197.64 | 197.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 192.80 | 197.28 | 197.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 199.00 | 193.81 | 195.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 199.00 | 193.81 | 195.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 199.00 | 193.81 | 195.51 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 202.14 | 196.73 | 196.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 210.00 | 196.97 | 196.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 303.20 | 304.18 | 281.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 303.20 | 304.18 | 281.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 346.80 | 363.46 | 349.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:00:00 | 346.80 | 363.46 | 349.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 352.65 | 363.36 | 349.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:15:00 | 353.80 | 363.36 | 349.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-02 09:15:00 | 389.18 | 365.89 | 353.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 14:15:00 | 335.00 | 358.19 | 358.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 330.00 | 354.59 | 355.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 10:15:00 | 355.40 | 354.29 | 355.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 10:15:00 | 355.40 | 354.29 | 355.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 355.40 | 354.29 | 355.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 355.80 | 354.29 | 355.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 357.00 | 354.32 | 355.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:00:00 | 357.00 | 354.32 | 355.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 357.00 | 354.35 | 355.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:30:00 | 357.05 | 354.35 | 355.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 355.00 | 354.35 | 355.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:45:00 | 353.25 | 354.35 | 355.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 352.80 | 354.35 | 355.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 358.70 | 354.40 | 355.74 | SL hit (close>static) qty=1.00 sl=357.10 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 374.90 | 339.54 | 339.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 378.30 | 342.84 | 341.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 373.80 | 374.32 | 363.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 373.80 | 374.32 | 363.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 361.05 | 376.83 | 367.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 361.50 | 376.83 | 367.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 359.80 | 376.66 | 367.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 359.80 | 376.66 | 367.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 367.95 | 375.92 | 367.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 367.50 | 375.92 | 367.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 366.65 | 375.83 | 367.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:15:00 | 365.75 | 375.83 | 367.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 366.75 | 375.74 | 367.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 366.45 | 375.74 | 367.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 363.35 | 375.62 | 367.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:45:00 | 364.10 | 375.62 | 367.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 368.05 | 375.10 | 367.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 368.05 | 375.10 | 367.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 368.00 | 375.03 | 367.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 371.05 | 375.03 | 367.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:15:00 | 368.40 | 374.68 | 367.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:00:00 | 368.60 | 374.62 | 367.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 365.40 | 374.31 | 367.72 | SL hit (close<static) qty=1.00 sl=367.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 346.10 | 363.68 | 363.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 345.70 | 362.87 | 363.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 353.00 | 351.98 | 356.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 11:00:00 | 353.00 | 351.98 | 356.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 356.45 | 352.08 | 356.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 356.45 | 352.08 | 356.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 355.90 | 352.12 | 356.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 359.35 | 352.12 | 356.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 354.90 | 352.14 | 356.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 352.40 | 352.15 | 356.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 364.70 | 352.44 | 356.82 | SL hit (close>static) qty=1.00 sl=361.40 alert=retest2 |

### Cycle 6 — BUY (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 15:15:00 | 369.00 | 359.86 | 359.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 375.95 | 360.02 | 359.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 13:15:00 | 358.00 | 360.82 | 360.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 13:15:00 | 358.00 | 360.82 | 360.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 358.00 | 360.82 | 360.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 358.00 | 360.82 | 360.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 354.55 | 360.76 | 360.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 354.55 | 360.76 | 360.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 348.25 | 359.77 | 359.81 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 368.00 | 359.88 | 359.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 369.05 | 360.04 | 359.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 355.00 | 360.29 | 360.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 355.00 | 360.29 | 360.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 355.00 | 360.29 | 360.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 355.00 | 360.29 | 360.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 354.00 | 360.23 | 360.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 355.00 | 360.23 | 360.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 349.65 | 359.79 | 359.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 347.30 | 359.67 | 359.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 362.85 | 358.08 | 358.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 362.85 | 358.08 | 358.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 362.85 | 358.08 | 358.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 362.55 | 358.08 | 358.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 361.35 | 358.11 | 358.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:15:00 | 362.95 | 358.11 | 358.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 363.90 | 359.63 | 359.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 364.55 | 359.68 | 359.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 362.25 | 362.55 | 361.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:00:00 | 362.25 | 362.55 | 361.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 362.25 | 362.55 | 361.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:30:00 | 362.70 | 362.57 | 361.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 359.20 | 362.53 | 361.24 | SL hit (close<static) qty=1.00 sl=360.85 alert=retest2 |

### Cycle 11 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 327.25 | 359.88 | 360.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 10:15:00 | 323.80 | 356.85 | 358.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 343.00 | 336.17 | 344.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 343.00 | 336.17 | 344.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 343.00 | 336.17 | 344.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 343.00 | 336.17 | 344.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 348.75 | 336.29 | 344.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 348.40 | 336.29 | 344.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 351.85 | 336.45 | 344.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 347.55 | 337.17 | 345.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:00:00 | 346.85 | 337.27 | 345.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 346.40 | 337.37 | 345.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 12:15:00 | 353.95 | 339.71 | 345.47 | SL hit (close>static) qty=1.00 sl=353.90 alert=retest2 |

### Cycle 12 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 358.85 | 349.02 | 349.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 359.95 | 349.13 | 349.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 350.15 | 351.05 | 350.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 350.15 | 351.05 | 350.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 353.85 | 351.08 | 350.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 353.80 | 351.08 | 350.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 350.85 | 351.52 | 350.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 350.85 | 351.52 | 350.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 341.05 | 351.42 | 350.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 341.05 | 351.42 | 350.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 345.60 | 351.36 | 350.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 347.15 | 351.33 | 350.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 346.15 | 350.99 | 350.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 345.75 | 350.64 | 350.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 350.00 | 350.35 | 349.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 346.90 | 350.27 | 349.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 344.75 | 350.27 | 349.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 348.00 | 350.20 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 348.00 | 350.20 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 351.15 | 350.18 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 351.10 | 350.18 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 350.25 | 350.18 | 349.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:45:00 | 350.10 | 350.18 | 349.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 350.95 | 350.19 | 349.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 351.65 | 350.19 | 349.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 349.70 | 350.18 | 349.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 350.10 | 350.18 | 349.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 351.15 | 350.19 | 349.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 339.85 | 349.62 | 349.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 339.85 | 349.62 | 349.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 14:15:00 | 338.60 | 349.51 | 349.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 346.80 | 345.51 | 347.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 346.80 | 345.51 | 347.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 346.80 | 345.51 | 347.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 346.80 | 345.51 | 347.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 347.20 | 345.53 | 347.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 352.10 | 345.53 | 347.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 353.45 | 345.61 | 347.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 354.30 | 345.61 | 347.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 354.50 | 345.70 | 347.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 355.40 | 345.70 | 347.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 251.10 | 226.03 | 247.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 252.65 | 226.03 | 247.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 249.29 | 226.26 | 247.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:30:00 | 252.28 | 226.26 | 247.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 245.57 | 226.68 | 247.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:15:00 | 249.14 | 226.68 | 247.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 246.99 | 226.88 | 247.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 241.90 | 227.06 | 247.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 242.77 | 227.37 | 247.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:15:00 | 242.64 | 227.37 | 247.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 14:15:00 | 230.63 | 228.75 | 246.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-21 14:15:00 | 230.21 | 228.75 | 246.17 | SL hit (close>static) qty=0.50 sl=228.75 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-23 13:15:00 | 353.80 | 2025-01-02 09:15:00 | 389.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-23 09:30:00 | 354.35 | 2025-01-28 09:15:00 | 334.70 | STOP_HIT | 1.00 | -5.55% |
| BUY | retest2 | 2025-01-23 10:15:00 | 353.80 | 2025-01-28 09:15:00 | 334.70 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest2 | 2025-01-23 10:45:00 | 355.20 | 2025-01-28 09:15:00 | 334.70 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2025-03-04 14:45:00 | 353.25 | 2025-03-05 10:15:00 | 358.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-03-04 15:15:00 | 352.80 | 2025-03-05 10:15:00 | 358.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-03-07 09:30:00 | 352.85 | 2025-03-11 09:15:00 | 335.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 12:00:00 | 351.90 | 2025-03-11 09:15:00 | 334.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 09:30:00 | 352.85 | 2025-03-12 10:15:00 | 317.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-07 12:00:00 | 351.90 | 2025-03-12 10:15:00 | 316.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 09:15:00 | 345.00 | 2025-04-02 09:15:00 | 329.84 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2025-03-28 09:15:00 | 345.00 | 2025-04-02 10:15:00 | 334.20 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-03-28 10:15:00 | 344.85 | 2025-04-03 14:15:00 | 345.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-03-28 11:00:00 | 347.20 | 2025-04-03 14:15:00 | 345.95 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-03-28 14:00:00 | 344.95 | 2025-04-03 14:15:00 | 345.95 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-04-01 10:15:00 | 334.00 | 2025-04-03 14:15:00 | 345.95 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-04-01 12:45:00 | 334.00 | 2025-04-03 14:15:00 | 345.95 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-04-02 11:00:00 | 334.20 | 2025-04-03 14:15:00 | 345.95 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-04-02 13:30:00 | 334.65 | 2025-04-04 10:15:00 | 327.75 | PARTIAL | 0.50 | 2.06% |
| SELL | retest2 | 2025-04-03 11:15:00 | 339.30 | 2025-04-04 10:15:00 | 327.61 | PARTIAL | 0.50 | 3.45% |
| SELL | retest2 | 2025-04-03 13:00:00 | 337.30 | 2025-04-04 10:15:00 | 327.70 | PARTIAL | 0.50 | 2.85% |
| SELL | retest2 | 2025-04-04 09:30:00 | 339.10 | 2025-04-04 13:15:00 | 322.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 13:30:00 | 334.65 | 2025-04-07 09:15:00 | 310.50 | TARGET_HIT | 0.50 | 7.22% |
| SELL | retest2 | 2025-04-03 11:15:00 | 339.30 | 2025-04-07 09:15:00 | 310.37 | TARGET_HIT | 0.50 | 8.53% |
| SELL | retest2 | 2025-04-03 13:00:00 | 337.30 | 2025-04-07 09:15:00 | 310.45 | TARGET_HIT | 0.50 | 7.96% |
| SELL | retest2 | 2025-04-04 09:30:00 | 339.10 | 2025-04-07 09:15:00 | 305.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-17 11:15:00 | 336.65 | 2025-04-21 10:15:00 | 343.75 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-02 12:15:00 | 329.85 | 2025-05-07 09:15:00 | 313.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 11:15:00 | 333.20 | 2025-05-07 09:15:00 | 316.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 332.00 | 2025-05-07 09:15:00 | 315.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 12:15:00 | 329.85 | 2025-05-12 09:15:00 | 338.40 | STOP_HIT | 0.50 | -2.59% |
| SELL | retest2 | 2025-05-05 11:15:00 | 333.20 | 2025-05-12 09:15:00 | 338.40 | STOP_HIT | 0.50 | -1.56% |
| SELL | retest2 | 2025-05-06 09:45:00 | 332.00 | 2025-05-12 09:15:00 | 338.40 | STOP_HIT | 0.50 | -1.93% |
| BUY | retest2 | 2025-07-04 09:15:00 | 371.05 | 2025-07-08 09:15:00 | 365.40 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-07 10:15:00 | 368.40 | 2025-07-08 09:15:00 | 365.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-07 11:00:00 | 368.60 | 2025-07-08 09:15:00 | 365.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-09 09:30:00 | 368.65 | 2025-07-09 13:15:00 | 366.35 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-10 09:15:00 | 367.30 | 2025-07-10 09:15:00 | 365.75 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-08-06 10:30:00 | 352.40 | 2025-08-07 13:15:00 | 364.70 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-09-23 12:30:00 | 362.70 | 2025-09-24 09:15:00 | 359.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-28 10:00:00 | 347.55 | 2025-10-31 12:15:00 | 353.95 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-28 11:00:00 | 346.85 | 2025-10-31 12:15:00 | 353.95 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-10-28 12:15:00 | 346.40 | 2025-10-31 12:15:00 | 353.95 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-07 09:15:00 | 343.00 | 2025-11-12 09:15:00 | 358.70 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2025-11-25 09:45:00 | 347.15 | 2025-12-08 13:15:00 | 339.85 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-11-28 15:00:00 | 346.15 | 2025-12-08 13:15:00 | 339.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-02 10:30:00 | 345.75 | 2025-12-08 13:15:00 | 339.85 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-12-03 09:15:00 | 350.00 | 2025-12-08 13:15:00 | 339.85 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2026-04-16 15:15:00 | 241.90 | 2026-04-21 14:15:00 | 230.63 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-04-16 15:15:00 | 241.90 | 2026-04-21 14:15:00 | 230.21 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2026-04-17 09:30:00 | 242.77 | 2026-04-21 14:15:00 | 230.51 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2026-04-17 09:30:00 | 242.77 | 2026-04-21 14:15:00 | 230.21 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2026-04-17 10:15:00 | 242.64 | 2026-04-22 09:15:00 | 229.81 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2026-04-17 10:15:00 | 242.64 | 2026-04-24 09:15:00 | 217.71 | TARGET_HIT | 0.50 | 10.27% |
| SELL | retest2 | 2026-05-07 09:15:00 | 241.50 | 2026-05-08 09:15:00 | 256.54 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest2 | 2026-05-07 14:45:00 | 235.91 | 2026-05-08 09:15:00 | 256.54 | STOP_HIT | 1.00 | -8.74% |
