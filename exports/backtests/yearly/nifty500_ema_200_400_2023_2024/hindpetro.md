# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 387.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 91 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 67 |
| PARTIAL | 17 |
| TARGET_HIT | 21 |
| STOP_HIT | 46 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 34
- **Target hits / Stop hits / Partials:** 21 / 46 / 17
- **Avg / median % per leg:** 2.95% / 2.45%
- **Sum % (uncompounded):** 247.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 19 | 51.4% | 17 | 20 | 0 | 3.69% | 136.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 37 | 19 | 51.4% | 17 | 20 | 0 | 3.69% | 136.4% |
| SELL (all) | 47 | 31 | 66.0% | 4 | 26 | 17 | 2.37% | 111.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 47 | 31 | 66.0% | 4 | 26 | 17 | 2.37% | 111.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 84 | 50 | 59.5% | 21 | 46 | 17 | 2.95% | 247.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 15:15:00 | 173.83 | 180.72 | 180.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 09:15:00 | 172.43 | 180.09 | 180.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 10:15:00 | 176.90 | 175.57 | 177.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 10:15:00 | 176.90 | 175.57 | 177.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 176.90 | 175.57 | 177.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:30:00 | 178.37 | 175.57 | 177.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 171.83 | 175.57 | 177.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 11:45:00 | 170.30 | 175.48 | 177.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 10:00:00 | 170.43 | 174.78 | 177.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:15:00 | 169.93 | 174.61 | 176.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-20 09:15:00 | 169.97 | 173.98 | 176.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 176.37 | 173.83 | 176.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:30:00 | 176.43 | 173.83 | 176.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 11:15:00 | 173.87 | 173.83 | 176.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:30:00 | 176.83 | 173.83 | 176.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 176.13 | 173.88 | 176.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:30:00 | 177.37 | 173.88 | 176.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 174.53 | 173.88 | 176.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 15:00:00 | 174.00 | 173.93 | 176.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 15:00:00 | 174.13 | 173.84 | 176.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 14:45:00 | 173.77 | 173.88 | 176.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 11:15:00 | 165.42 | 172.72 | 175.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 12:15:00 | 165.30 | 172.65 | 175.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 12:15:00 | 165.08 | 172.65 | 175.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-06 13:15:00 | 172.70 | 172.18 | 174.61 | SL hit (close>ema200) qty=0.50 sl=172.18 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 13:15:00 | 199.57 | 173.77 | 173.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 12:15:00 | 200.93 | 175.28 | 174.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 338.70 | 339.36 | 312.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-06 10:00:00 | 338.70 | 339.36 | 312.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 315.20 | 338.52 | 317.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:45:00 | 309.33 | 338.52 | 317.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 307.17 | 338.21 | 317.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 11:00:00 | 307.17 | 338.21 | 317.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 314.50 | 329.00 | 315.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 09:15:00 | 314.53 | 329.00 | 315.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 315.50 | 328.37 | 315.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 15:00:00 | 315.50 | 328.37 | 315.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 315.57 | 328.24 | 315.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 313.10 | 328.24 | 315.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 314.30 | 328.10 | 315.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 10:30:00 | 314.80 | 327.99 | 315.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 320.77 | 327.39 | 315.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 315.33 | 326.81 | 315.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 12:00:00 | 314.70 | 325.74 | 315.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 319.47 | 324.76 | 315.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 308.13 | 323.95 | 315.82 | SL hit (close<static) qty=1.00 sl=309.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 392.45 | 397.29 | 397.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 386.05 | 397.05 | 397.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 385.55 | 384.38 | 389.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:30:00 | 387.20 | 384.38 | 389.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 389.35 | 383.56 | 388.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 389.35 | 383.56 | 388.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 388.00 | 383.60 | 388.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:15:00 | 389.50 | 383.60 | 388.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 387.05 | 383.64 | 388.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:30:00 | 388.95 | 383.64 | 388.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 387.95 | 383.68 | 388.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 387.70 | 383.68 | 388.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 389.70 | 383.74 | 388.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:45:00 | 389.90 | 383.74 | 388.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 387.90 | 383.78 | 388.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:45:00 | 385.75 | 383.84 | 388.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 11:45:00 | 386.70 | 383.91 | 388.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 12:15:00 | 391.50 | 383.98 | 388.32 | SL hit (close>static) qty=1.00 sl=389.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 414.55 | 391.85 | 391.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 10:15:00 | 417.60 | 396.82 | 394.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 399.55 | 402.63 | 398.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 399.55 | 402.63 | 398.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 399.55 | 402.63 | 398.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 396.35 | 402.63 | 398.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 393.10 | 402.53 | 398.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 393.10 | 402.53 | 398.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 392.70 | 402.43 | 398.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 391.95 | 402.43 | 398.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 371.00 | 395.16 | 395.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 12:15:00 | 366.80 | 393.79 | 394.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 333.00 | 329.56 | 348.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:00:00 | 333.00 | 329.56 | 348.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 342.50 | 329.58 | 342.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:45:00 | 344.05 | 329.58 | 342.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 348.35 | 329.77 | 342.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:45:00 | 349.00 | 329.77 | 342.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 349.65 | 329.97 | 342.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 349.75 | 329.97 | 342.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 361.00 | 344.37 | 347.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 11:00:00 | 361.00 | 344.37 | 347.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 354.10 | 345.13 | 347.57 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 379.80 | 350.02 | 349.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 383.60 | 353.43 | 351.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 396.85 | 401.81 | 389.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 396.85 | 401.81 | 389.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 381.55 | 401.14 | 389.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:45:00 | 386.15 | 400.82 | 389.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 385.50 | 400.09 | 389.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-26 10:15:00 | 424.05 | 398.98 | 391.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 394.70 | 410.55 | 410.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 393.80 | 410.22 | 410.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 398.30 | 396.35 | 401.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:45:00 | 397.80 | 396.35 | 401.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 403.80 | 396.42 | 401.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 403.80 | 396.42 | 401.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 402.35 | 396.48 | 401.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 401.85 | 396.48 | 401.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 399.15 | 396.51 | 401.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 398.60 | 396.52 | 401.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 398.10 | 396.61 | 401.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:30:00 | 398.50 | 396.67 | 401.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 398.70 | 396.72 | 401.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 402.20 | 396.88 | 401.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 406.80 | 397.28 | 401.48 | SL hit (close>static) qty=1.00 sl=402.90 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 14:15:00 | 423.05 | 404.64 | 404.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 435.70 | 406.34 | 405.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 465.00 | 467.20 | 450.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 14:00:00 | 465.00 | 467.20 | 450.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 452.35 | 465.03 | 452.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 450.20 | 465.03 | 452.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 452.60 | 464.91 | 452.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 456.80 | 464.91 | 452.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 450.00 | 464.65 | 452.39 | SL hit (close<static) qty=1.00 sl=451.45 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 421.20 | 457.24 | 457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 418.25 | 456.85 | 457.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 447.70 | 447.14 | 451.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 10:00:00 | 447.70 | 447.14 | 451.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 452.45 | 447.15 | 451.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 452.45 | 447.15 | 451.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 453.65 | 447.22 | 451.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 453.50 | 447.22 | 451.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 450.90 | 447.27 | 451.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 450.90 | 447.27 | 451.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 463.25 | 447.47 | 451.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 463.25 | 447.47 | 451.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 460.20 | 447.59 | 451.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 11:30:00 | 459.05 | 447.71 | 451.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:30:00 | 459.10 | 448.02 | 451.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 464.60 | 448.31 | 451.93 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-12 11:45:00 | 170.30 | 2023-10-04 11:15:00 | 165.42 | PARTIAL | 0.50 | 2.86% |
| SELL | retest2 | 2023-09-14 10:00:00 | 170.43 | 2023-10-04 12:15:00 | 165.30 | PARTIAL | 0.50 | 3.01% |
| SELL | retest2 | 2023-09-15 09:15:00 | 169.93 | 2023-10-04 12:15:00 | 165.08 | PARTIAL | 0.50 | 2.85% |
| SELL | retest2 | 2023-09-12 11:45:00 | 170.30 | 2023-10-06 13:15:00 | 172.70 | STOP_HIT | 0.50 | -1.41% |
| SELL | retest2 | 2023-09-14 10:00:00 | 170.43 | 2023-10-06 13:15:00 | 172.70 | STOP_HIT | 0.50 | -1.33% |
| SELL | retest2 | 2023-09-15 09:15:00 | 169.93 | 2023-10-06 13:15:00 | 172.70 | STOP_HIT | 0.50 | -1.63% |
| SELL | retest2 | 2023-09-20 09:15:00 | 169.97 | 2023-10-25 11:15:00 | 164.95 | PARTIAL | 0.50 | 2.95% |
| SELL | retest2 | 2023-09-22 15:00:00 | 174.00 | 2023-10-25 12:15:00 | 163.66 | PARTIAL | 0.50 | 5.94% |
| SELL | retest2 | 2023-09-25 15:00:00 | 174.13 | 2023-10-25 12:15:00 | 163.69 | PARTIAL | 0.50 | 6.00% |
| SELL | retest2 | 2023-09-26 14:45:00 | 173.77 | 2023-10-25 12:15:00 | 163.37 | PARTIAL | 0.50 | 5.98% |
| SELL | retest2 | 2023-10-12 12:30:00 | 173.63 | 2023-10-25 12:15:00 | 163.97 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2023-10-18 12:00:00 | 172.27 | 2023-10-25 12:15:00 | 163.97 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2023-10-19 09:30:00 | 172.30 | 2023-10-26 09:15:00 | 161.78 | PARTIAL | 0.50 | 6.10% |
| SELL | retest2 | 2023-10-19 10:00:00 | 171.97 | 2023-10-26 09:15:00 | 161.91 | PARTIAL | 0.50 | 5.85% |
| SELL | retest2 | 2023-10-19 12:30:00 | 172.60 | 2023-10-26 09:15:00 | 161.43 | PARTIAL | 0.50 | 6.47% |
| SELL | retest2 | 2023-10-20 09:15:00 | 172.60 | 2023-10-26 09:15:00 | 161.47 | PARTIAL | 0.50 | 6.45% |
| SELL | retest2 | 2023-09-20 09:15:00 | 169.97 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 0.35% |
| SELL | retest2 | 2023-09-22 15:00:00 | 174.00 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2023-09-25 15:00:00 | 174.13 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2023-09-26 14:45:00 | 173.77 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2023-10-12 12:30:00 | 173.63 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2023-10-18 12:00:00 | 172.27 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2023-10-19 09:30:00 | 172.30 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2023-10-19 10:00:00 | 171.97 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2023-10-19 12:30:00 | 172.60 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2023-10-20 09:15:00 | 172.60 | 2023-11-01 11:15:00 | 169.37 | STOP_HIT | 0.50 | 1.87% |
| SELL | retest2 | 2023-11-06 10:30:00 | 174.10 | 2023-11-07 10:15:00 | 180.73 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2023-11-06 11:45:00 | 174.07 | 2023-11-07 10:15:00 | 180.73 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2023-11-06 12:45:00 | 174.17 | 2023-11-07 10:15:00 | 180.73 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-03-26 10:30:00 | 314.80 | 2024-04-04 11:15:00 | 308.13 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-03-27 09:15:00 | 320.77 | 2024-04-04 11:15:00 | 308.13 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-03-27 15:15:00 | 315.33 | 2024-04-04 11:15:00 | 308.13 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-04-01 12:00:00 | 314.70 | 2024-04-04 11:15:00 | 308.13 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-04-10 14:15:00 | 323.30 | 2024-04-15 09:15:00 | 312.50 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2024-04-12 09:45:00 | 322.53 | 2024-04-15 09:15:00 | 312.50 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-04-18 09:15:00 | 326.93 | 2024-04-19 09:15:00 | 310.00 | STOP_HIT | 1.00 | -5.18% |
| BUY | retest2 | 2024-04-22 09:15:00 | 330.07 | 2024-05-23 10:15:00 | 363.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 14:00:00 | 329.13 | 2024-05-23 10:15:00 | 362.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 10:30:00 | 329.00 | 2024-05-23 10:15:00 | 361.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 12:30:00 | 329.00 | 2024-05-23 10:15:00 | 361.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-16 14:15:00 | 331.03 | 2024-05-24 11:15:00 | 364.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:30:00 | 339.87 | 2024-06-25 09:15:00 | 335.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-06-06 09:15:00 | 344.43 | 2024-06-25 09:15:00 | 335.80 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-06-07 11:45:00 | 339.67 | 2024-06-25 09:15:00 | 335.80 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-06-07 12:45:00 | 340.60 | 2024-07-25 11:15:00 | 373.64 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2024-06-21 14:45:00 | 342.05 | 2024-07-25 14:15:00 | 373.86 | TARGET_HIT | 1.00 | 9.30% |
| BUY | retest2 | 2024-06-24 10:15:00 | 341.80 | 2024-07-25 14:15:00 | 374.66 | TARGET_HIT | 1.00 | 9.61% |
| BUY | retest2 | 2024-06-24 13:00:00 | 341.65 | 2024-07-26 09:15:00 | 378.87 | TARGET_HIT | 1.00 | 10.90% |
| BUY | retest2 | 2024-07-11 09:15:00 | 343.25 | 2024-07-26 09:15:00 | 377.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-22 10:30:00 | 354.75 | 2024-07-29 09:15:00 | 386.81 | TARGET_HIT | 1.00 | 9.04% |
| BUY | retest2 | 2024-07-24 10:00:00 | 351.65 | 2024-07-29 09:15:00 | 386.21 | TARGET_HIT | 1.00 | 9.83% |
| BUY | retest2 | 2024-07-24 11:30:00 | 351.10 | 2024-07-29 09:15:00 | 386.16 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2024-07-24 13:00:00 | 351.05 | 2024-07-30 09:15:00 | 390.23 | TARGET_HIT | 1.00 | 11.16% |
| BUY | retest2 | 2024-10-24 15:00:00 | 405.05 | 2024-10-25 09:15:00 | 389.00 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-12-05 09:45:00 | 385.75 | 2024-12-05 12:15:00 | 391.50 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-05 11:45:00 | 386.70 | 2024-12-05 12:15:00 | 391.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-13 11:45:00 | 386.15 | 2025-06-26 10:15:00 | 424.05 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2025-06-16 10:15:00 | 385.50 | 2025-06-27 09:15:00 | 424.76 | TARGET_HIT | 1.00 | 10.19% |
| BUY | retest2 | 2025-08-18 12:30:00 | 385.30 | 2025-08-21 10:15:00 | 394.70 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-08-18 13:45:00 | 386.25 | 2025-08-21 10:15:00 | 394.70 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-09-11 13:30:00 | 398.60 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-12 11:45:00 | 398.10 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-09-12 14:30:00 | 398.50 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-15 09:30:00 | 398.70 | 2025-09-17 09:15:00 | 406.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-12-02 09:15:00 | 456.80 | 2025-12-02 10:15:00 | 450.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-02 11:30:00 | 453.15 | 2025-12-02 12:15:00 | 449.65 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-03 13:30:00 | 453.10 | 2025-12-03 14:15:00 | 451.05 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-09 12:00:00 | 452.70 | 2025-12-09 14:15:00 | 448.60 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-10 09:15:00 | 450.50 | 2025-12-11 09:15:00 | 445.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-10 15:00:00 | 450.00 | 2025-12-11 09:15:00 | 445.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-11 09:15:00 | 450.75 | 2025-12-11 09:15:00 | 445.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-12 09:15:00 | 451.20 | 2025-12-31 13:15:00 | 496.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-04 11:30:00 | 459.05 | 2026-02-05 09:15:00 | 464.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-04 14:30:00 | 459.10 | 2026-02-05 09:15:00 | 464.60 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-02-05 12:30:00 | 457.50 | 2026-02-06 09:15:00 | 466.45 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-05 15:00:00 | 458.90 | 2026-02-06 09:15:00 | 466.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-25 11:15:00 | 435.40 | 2026-03-02 09:15:00 | 416.00 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2026-02-26 11:30:00 | 437.90 | 2026-03-02 09:15:00 | 416.15 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-02-26 13:15:00 | 438.05 | 2026-03-04 09:15:00 | 413.63 | PARTIAL | 0.50 | 5.57% |
| SELL | retest2 | 2026-02-27 10:45:00 | 436.50 | 2026-03-04 09:15:00 | 414.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 435.40 | 2026-03-09 09:15:00 | 391.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 437.90 | 2026-03-09 09:15:00 | 394.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 13:15:00 | 438.05 | 2026-03-09 09:15:00 | 394.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 436.50 | 2026-03-09 09:15:00 | 392.85 | TARGET_HIT | 0.50 | 10.00% |
