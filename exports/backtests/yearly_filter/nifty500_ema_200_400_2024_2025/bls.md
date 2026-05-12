# BLS International Services Ltd. (BLS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 290.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 22 |
| TARGET_HIT | 18 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 34
- **Target hits / Stop hits / Partials:** 18 / 43 / 22
- **Avg / median % per leg:** 2.49% / 2.62%
- **Sum % (uncompounded):** 206.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 11 | 57.9% | 10 | 9 | 0 | 4.51% | 85.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 11 | 57.9% | 10 | 9 | 0 | 4.51% | 85.7% |
| SELL (all) | 64 | 38 | 59.4% | 8 | 34 | 22 | 1.89% | 120.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 64 | 38 | 59.4% | 8 | 34 | 22 | 1.89% | 120.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 83 | 49 | 59.0% | 18 | 43 | 22 | 2.49% | 206.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 359.60 | 333.13 | 333.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 363.05 | 337.96 | 335.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 351.75 | 358.88 | 349.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 15:00:00 | 351.75 | 358.88 | 349.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 345.20 | 358.68 | 349.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 345.20 | 358.68 | 349.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 346.20 | 358.55 | 349.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:15:00 | 344.45 | 358.55 | 349.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 348.00 | 352.43 | 347.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 355.30 | 352.43 | 347.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 09:30:00 | 356.50 | 355.03 | 350.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 12:00:00 | 353.45 | 354.98 | 350.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 13:15:00 | 353.75 | 354.94 | 350.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 347.50 | 354.82 | 350.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 364.90 | 354.82 | 350.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-06 12:15:00 | 388.80 | 355.76 | 350.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 364.70 | 386.40 | 386.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 363.30 | 386.17 | 386.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 383.05 | 382.90 | 384.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:30:00 | 383.80 | 382.90 | 384.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 384.35 | 382.91 | 384.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 386.65 | 382.91 | 384.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 383.75 | 382.92 | 384.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 384.35 | 382.92 | 384.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 380.30 | 382.89 | 384.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 372.80 | 382.73 | 384.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:45:00 | 377.50 | 382.53 | 384.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 15:00:00 | 377.30 | 382.41 | 384.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 376.50 | 382.32 | 384.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 358.62 | 381.44 | 383.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 358.44 | 381.44 | 383.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 357.68 | 381.44 | 383.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 354.16 | 381.19 | 383.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 385.05 | 379.40 | 382.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 385.05 | 379.40 | 382.47 | SL hit (close>ema200) qty=0.50 sl=379.40 alert=retest2 |

### Cycle 3 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 418.00 | 384.35 | 384.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 420.90 | 386.29 | 385.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 386.00 | 393.10 | 389.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 386.00 | 393.10 | 389.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 386.00 | 393.10 | 389.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 386.00 | 393.10 | 389.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 386.30 | 393.03 | 389.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 12:00:00 | 394.30 | 393.04 | 389.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 09:45:00 | 394.35 | 392.90 | 389.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 12:00:00 | 393.75 | 392.89 | 389.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 382.85 | 392.72 | 389.12 | SL hit (close<static) qty=1.00 sl=382.90 alert=retest2 |

### Cycle 4 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 395.40 | 436.75 | 436.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 388.90 | 435.11 | 436.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 366.50 | 365.23 | 388.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:45:00 | 365.40 | 365.23 | 388.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 399.75 | 365.97 | 388.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 399.75 | 365.97 | 388.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 399.60 | 366.31 | 388.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 399.05 | 366.31 | 388.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 394.00 | 374.89 | 390.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 393.70 | 374.89 | 390.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 396.85 | 377.60 | 390.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 395.85 | 377.76 | 390.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:45:00 | 395.15 | 378.53 | 391.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 10:30:00 | 393.85 | 378.67 | 391.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:00:00 | 392.40 | 378.67 | 391.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 402.60 | 378.91 | 391.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 402.60 | 378.91 | 391.08 | SL hit (close>static) qty=1.00 sl=398.90 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 419.40 | 383.69 | 383.54 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 364.90 | 386.27 | 386.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 362.10 | 385.81 | 386.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 378.75 | 377.41 | 381.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 381.35 | 377.41 | 381.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 370.55 | 377.36 | 381.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 366.30 | 376.94 | 380.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 388.50 | 374.35 | 378.42 | SL hit (close>static) qty=1.00 sl=382.40 alert=retest2 |

### Cycle 7 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 397.50 | 379.60 | 379.52 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 373.00 | 379.92 | 379.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 368.85 | 379.67 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 15:15:00 | 355.25 | 353.95 | 363.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 356.80 | 353.95 | 363.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 337.50 | 323.97 | 337.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 337.50 | 323.97 | 337.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 337.40 | 324.10 | 337.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 335.05 | 324.10 | 337.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 336.00 | 324.22 | 337.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:15:00 | 319.20 | 324.44 | 337.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 329.00 | 324.48 | 336.97 | SL hit (close>ema200) qty=0.50 sl=324.48 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 13:15:00 | 293.15 | 278.16 | 278.14 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-29 09:15:00 | 355.30 | 2024-08-06 12:15:00 | 388.80 | TARGET_HIT | 1.00 | 9.43% |
| BUY | retest2 | 2024-08-05 09:30:00 | 356.50 | 2024-08-06 12:15:00 | 389.13 | TARGET_HIT | 1.00 | 9.15% |
| BUY | retest2 | 2024-08-05 12:00:00 | 353.45 | 2024-08-06 13:15:00 | 390.83 | TARGET_HIT | 1.00 | 10.58% |
| BUY | retest2 | 2024-08-05 13:15:00 | 353.75 | 2024-08-06 13:15:00 | 392.15 | TARGET_HIT | 1.00 | 10.86% |
| BUY | retest2 | 2024-08-06 09:15:00 | 364.90 | 2024-08-08 11:15:00 | 401.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-04 10:30:00 | 356.20 | 2024-10-07 12:15:00 | 344.30 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2024-10-08 13:00:00 | 352.75 | 2024-10-10 13:15:00 | 364.70 | STOP_HIT | 1.00 | 3.39% |
| SELL | retest2 | 2024-10-18 09:15:00 | 372.80 | 2024-10-22 09:15:00 | 358.62 | PARTIAL | 0.50 | 3.80% |
| SELL | retest2 | 2024-10-18 11:45:00 | 377.50 | 2024-10-22 09:15:00 | 358.44 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2024-10-18 15:00:00 | 377.30 | 2024-10-22 09:15:00 | 357.68 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2024-10-21 09:30:00 | 376.50 | 2024-10-22 10:15:00 | 354.16 | PARTIAL | 0.50 | 5.93% |
| SELL | retest2 | 2024-10-18 09:15:00 | 372.80 | 2024-10-23 11:15:00 | 385.05 | STOP_HIT | 0.50 | -3.29% |
| SELL | retest2 | 2024-10-18 11:45:00 | 377.50 | 2024-10-23 11:15:00 | 385.05 | STOP_HIT | 0.50 | -2.00% |
| SELL | retest2 | 2024-10-18 15:00:00 | 377.30 | 2024-10-23 11:15:00 | 385.05 | STOP_HIT | 0.50 | -2.05% |
| SELL | retest2 | 2024-10-21 09:30:00 | 376.50 | 2024-10-23 11:15:00 | 385.05 | STOP_HIT | 0.50 | -2.27% |
| SELL | retest2 | 2024-10-25 09:45:00 | 374.50 | 2024-10-31 11:15:00 | 395.10 | STOP_HIT | 1.00 | -5.50% |
| SELL | retest2 | 2024-10-25 13:00:00 | 377.40 | 2024-10-31 11:15:00 | 395.10 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2024-10-28 09:30:00 | 369.90 | 2024-10-31 11:15:00 | 395.10 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest2 | 2024-10-28 13:00:00 | 377.55 | 2024-10-31 11:15:00 | 395.10 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2024-11-13 12:00:00 | 394.30 | 2024-11-18 09:15:00 | 382.85 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-11-14 09:45:00 | 394.35 | 2024-11-18 09:15:00 | 382.85 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-11-14 12:00:00 | 393.75 | 2024-11-18 09:15:00 | 382.85 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-11-19 09:15:00 | 394.50 | 2024-11-21 13:15:00 | 381.75 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-11-25 09:15:00 | 391.20 | 2024-12-03 12:15:00 | 424.71 | TARGET_HIT | 1.00 | 8.57% |
| BUY | retest2 | 2024-11-29 10:45:00 | 386.10 | 2024-12-03 12:15:00 | 424.66 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2024-11-29 11:15:00 | 386.05 | 2024-12-03 12:15:00 | 425.15 | TARGET_HIT | 1.00 | 10.13% |
| BUY | retest2 | 2024-11-29 12:00:00 | 386.50 | 2024-12-06 11:15:00 | 430.32 | TARGET_HIT | 1.00 | 11.34% |
| BUY | retest2 | 2024-12-03 09:15:00 | 412.50 | 2024-12-10 09:15:00 | 453.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-12 10:30:00 | 400.45 | 2025-02-13 14:15:00 | 395.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-02-12 15:00:00 | 397.05 | 2025-02-13 14:15:00 | 395.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-02-13 09:30:00 | 398.70 | 2025-02-13 14:15:00 | 395.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-04-01 11:30:00 | 395.85 | 2025-04-02 11:15:00 | 402.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-04-02 09:45:00 | 395.15 | 2025-04-02 11:15:00 | 402.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-04-02 10:30:00 | 393.85 | 2025-04-02 11:15:00 | 402.60 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-04-02 11:00:00 | 392.40 | 2025-04-02 11:15:00 | 402.60 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-16 09:30:00 | 383.75 | 2025-04-16 13:15:00 | 388.45 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-04-16 13:15:00 | 383.70 | 2025-04-16 13:15:00 | 388.45 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-04-16 14:30:00 | 383.45 | 2025-04-21 09:15:00 | 392.75 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-04-16 15:00:00 | 383.00 | 2025-04-21 09:15:00 | 392.75 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-04-25 09:15:00 | 381.15 | 2025-04-25 11:15:00 | 362.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:15:00 | 381.15 | 2025-05-06 13:15:00 | 343.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-15 09:45:00 | 381.10 | 2025-05-15 12:15:00 | 394.30 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-05-15 11:15:00 | 380.45 | 2025-05-15 12:15:00 | 394.30 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-07-01 09:15:00 | 366.30 | 2025-07-10 09:15:00 | 388.50 | STOP_HIT | 1.00 | -6.06% |
| SELL | retest2 | 2025-07-23 13:30:00 | 368.00 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-07-23 14:30:00 | 367.70 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-07-23 15:00:00 | 367.15 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2025-07-31 09:15:00 | 373.10 | 2025-08-01 09:15:00 | 383.10 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-08-04 10:45:00 | 374.40 | 2025-08-04 13:15:00 | 382.95 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-12 14:15:00 | 335.05 | 2025-11-14 09:15:00 | 319.20 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-11-12 14:15:00 | 335.05 | 2025-11-14 10:15:00 | 329.00 | STOP_HIT | 0.50 | 1.81% |
| SELL | retest2 | 2025-11-12 15:00:00 | 336.00 | 2025-11-21 11:15:00 | 318.30 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-11-12 15:00:00 | 336.00 | 2025-11-28 09:15:00 | 331.65 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2025-12-04 10:30:00 | 335.35 | 2025-12-08 13:15:00 | 318.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 12:15:00 | 335.50 | 2025-12-08 13:15:00 | 318.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 328.70 | 2025-12-09 09:15:00 | 312.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 10:30:00 | 335.35 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 0.50 | 0.25% |
| SELL | retest2 | 2025-12-04 12:15:00 | 335.50 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 0.50 | 0.30% |
| SELL | retest2 | 2025-12-08 09:15:00 | 328.70 | 2025-12-19 09:15:00 | 334.50 | STOP_HIT | 0.50 | -1.76% |
| SELL | retest2 | 2025-12-19 10:30:00 | 330.35 | 2025-12-30 11:15:00 | 313.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 12:30:00 | 329.65 | 2025-12-30 12:15:00 | 313.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 13:00:00 | 329.75 | 2025-12-30 12:15:00 | 313.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-19 10:30:00 | 330.35 | 2025-12-31 15:15:00 | 321.70 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-12-19 12:30:00 | 329.65 | 2025-12-31 15:15:00 | 321.70 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-12-19 13:00:00 | 329.75 | 2025-12-31 15:15:00 | 321.70 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-12-23 11:15:00 | 322.80 | 2026-01-08 14:15:00 | 306.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 322.55 | 2026-01-08 14:15:00 | 306.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:30:00 | 323.10 | 2026-01-08 14:15:00 | 306.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 13:15:00 | 323.00 | 2026-01-08 14:15:00 | 306.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 11:15:00 | 322.80 | 2026-01-20 14:15:00 | 290.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 322.55 | 2026-01-20 14:15:00 | 290.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 11:30:00 | 323.10 | 2026-01-20 14:15:00 | 290.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 13:15:00 | 323.00 | 2026-01-20 14:15:00 | 290.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 287.80 | 2026-02-13 09:15:00 | 273.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:30:00 | 287.80 | 2026-02-23 10:15:00 | 293.85 | STOP_HIT | 0.50 | -2.10% |
| SELL | retest2 | 2026-02-23 11:15:00 | 289.45 | 2026-02-24 09:15:00 | 274.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 289.45 | 2026-02-24 14:15:00 | 285.25 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2026-02-23 12:00:00 | 288.70 | 2026-03-02 09:15:00 | 274.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 289.15 | 2026-03-02 09:15:00 | 274.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:00:00 | 288.70 | 2026-03-04 09:15:00 | 259.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 14:00:00 | 289.15 | 2026-03-04 09:15:00 | 260.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 269.55 | 2026-03-27 09:15:00 | 256.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 269.55 | 2026-03-30 09:15:00 | 242.60 | TARGET_HIT | 0.50 | 10.00% |
