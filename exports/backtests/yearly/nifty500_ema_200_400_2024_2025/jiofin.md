# Jio Financial Services Ltd. (JIOFIN)

## Backtest Summary

- **Window:** 2023-08-21 09:15:00 → 2026-05-08 15:15:00 (4687 bars)
- **Last close:** 249.01
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 3
- **Avg / median % per leg:** 1.18% / -1.16%
- **Sum % (uncompounded):** 24.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.42% | -11.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.42% | -11.4% |
| SELL (all) | 13 | 6 | 46.2% | 3 | 7 | 3 | 2.78% | 36.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 3 | 7 | 3 | 2.78% | 36.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 6 | 28.6% | 3 | 15 | 3 | 1.18% | 24.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 13:15:00 | 335.00 | 349.63 | 349.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 331.85 | 348.26 | 348.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 333.70 | 333.05 | 339.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 10:00:00 | 333.70 | 333.05 | 339.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 345.40 | 330.36 | 335.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 345.40 | 330.36 | 335.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 347.80 | 330.54 | 335.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 345.80 | 330.54 | 335.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 338.15 | 333.70 | 336.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:45:00 | 338.95 | 333.70 | 336.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 338.40 | 333.74 | 336.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:45:00 | 338.55 | 333.74 | 336.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 337.55 | 333.81 | 336.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:30:00 | 337.65 | 333.81 | 336.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 336.95 | 333.85 | 336.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 15:15:00 | 337.75 | 333.85 | 336.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 337.75 | 333.88 | 336.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 333.35 | 333.88 | 336.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 334.20 | 333.89 | 336.87 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 348.95 | 339.22 | 339.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 355.00 | 340.12 | 339.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 344.70 | 345.25 | 342.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 342.15 | 345.21 | 342.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 342.15 | 345.21 | 342.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 342.15 | 345.21 | 342.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 342.20 | 345.18 | 342.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 342.20 | 345.18 | 342.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 340.30 | 345.13 | 342.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 340.30 | 345.13 | 342.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 342.40 | 345.13 | 342.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 341.00 | 345.13 | 342.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 343.25 | 345.11 | 342.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:45:00 | 344.60 | 345.10 | 342.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 338.95 | 344.97 | 342.71 | SL hit (close<static) qty=1.00 sl=341.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 14:15:00 | 330.00 | 341.27 | 341.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 328.55 | 340.81 | 341.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 323.45 | 322.14 | 328.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 323.45 | 322.14 | 328.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 328.75 | 322.38 | 328.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 328.75 | 322.38 | 328.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 329.70 | 322.45 | 328.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:30:00 | 329.45 | 322.45 | 328.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 327.90 | 322.86 | 328.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 328.60 | 322.86 | 328.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 327.25 | 322.90 | 328.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:45:00 | 326.85 | 322.90 | 328.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 329.50 | 323.09 | 328.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 331.80 | 323.09 | 328.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 329.40 | 323.15 | 328.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 330.20 | 323.15 | 328.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 329.25 | 323.27 | 328.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 329.25 | 323.27 | 328.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 328.35 | 323.43 | 328.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 326.60 | 323.43 | 328.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 328.85 | 323.48 | 328.08 | SL hit (close>static) qty=1.00 sl=328.60 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 340.90 | 331.30 | 331.28 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 11:15:00 | 314.25 | 331.17 | 331.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 13:15:00 | 312.45 | 330.81 | 331.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 230.05 | 228.67 | 247.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 230.05 | 228.67 | 247.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 238.21 | 227.11 | 238.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 15:00:00 | 238.21 | 227.11 | 238.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 15:15:00 | 239.20 | 227.23 | 238.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 09:15:00 | 241.90 | 227.23 | 238.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 241.88 | 227.38 | 238.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:00:00 | 241.88 | 227.38 | 238.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 241.54 | 227.52 | 238.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 09:15:00 | 238.97 | 228.23 | 238.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 10:00:00 | 238.99 | 228.34 | 238.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 11:15:00 | 243.42 | 228.62 | 238.54 | SL hit (close>static) qty=1.00 sl=242.15 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 255.40 | 244.65 | 244.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 260.40 | 245.41 | 245.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 283.30 | 283.58 | 272.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 284.90 | 283.58 | 272.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 313.30 | 322.32 | 313.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 312.00 | 322.32 | 313.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 315.80 | 322.26 | 313.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 317.45 | 316.00 | 312.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 316.70 | 315.99 | 312.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 316.65 | 316.05 | 312.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 316.50 | 316.08 | 312.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 313.90 | 316.20 | 313.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 313.65 | 316.20 | 313.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 311.80 | 316.14 | 313.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 311.80 | 316.14 | 313.28 | SL hit (close<static) qty=1.00 sl=312.85 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 294.50 | 311.01 | 311.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 291.90 | 305.13 | 306.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 302.30 | 300.48 | 303.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:45:00 | 301.75 | 300.48 | 303.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 301.55 | 298.92 | 301.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 301.60 | 298.92 | 301.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 302.10 | 298.95 | 301.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 302.10 | 298.95 | 301.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 302.00 | 298.98 | 301.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 301.30 | 299.01 | 301.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 300.50 | 299.04 | 301.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 301.00 | 299.11 | 301.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 300.80 | 299.15 | 301.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 304.30 | 299.10 | 301.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 304.30 | 299.10 | 301.79 | SL hit (close>static) qty=1.00 sl=302.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-04 11:45:00 | 344.60 | 2024-10-04 14:15:00 | 338.95 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-07 09:15:00 | 344.90 | 2024-10-07 09:15:00 | 340.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-10-08 15:00:00 | 345.30 | 2024-10-11 15:15:00 | 340.95 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-10-10 09:15:00 | 344.25 | 2024-10-11 15:15:00 | 340.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-12-02 09:15:00 | 326.60 | 2024-12-02 09:15:00 | 328.85 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-17 09:15:00 | 238.97 | 2025-04-17 11:15:00 | 243.42 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-04-17 10:00:00 | 238.99 | 2025-04-17 11:15:00 | 243.42 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-09-16 11:30:00 | 317.45 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-09-17 09:15:00 | 316.70 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-09-18 09:15:00 | 316.65 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-09-18 13:00:00 | 316.50 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-02 14:30:00 | 301.30 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 300.50 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-05 13:00:00 | 301.00 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-06 10:00:00 | 300.80 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-07 13:15:00 | 302.90 | 2026-01-09 14:15:00 | 287.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:15:00 | 303.20 | 2026-01-09 14:15:00 | 288.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 301.70 | 2026-01-09 14:15:00 | 286.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 302.90 | 2026-01-20 09:15:00 | 272.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 14:15:00 | 303.20 | 2026-01-20 09:15:00 | 272.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 301.70 | 2026-01-20 09:15:00 | 271.53 | TARGET_HIT | 0.50 | 10.00% |
