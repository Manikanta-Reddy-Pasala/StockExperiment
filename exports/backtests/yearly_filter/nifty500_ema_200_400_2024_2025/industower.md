# Indus Towers Ltd. (INDUSTOWER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 402.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 61 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 57 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 44
- **Target hits / Stop hits / Partials:** 7 / 51 / 7
- **Avg / median % per leg:** -0.29% / -2.19%
- **Sum % (uncompounded):** -18.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 7 | 31.8% | 7 | 15 | 0 | 1.28% | 28.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 7 | 31.8% | 7 | 15 | 0 | 1.28% | 28.1% |
| SELL (all) | 43 | 14 | 32.6% | 0 | 36 | 7 | -1.09% | -46.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.74% | -3.7% |
| SELL @ 3rd Alert (retest2) | 42 | 14 | 33.3% | 0 | 35 | 7 | -1.03% | -43.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.74% | -3.7% |
| retest2 (combined) | 64 | 21 | 32.8% | 7 | 50 | 7 | -0.23% | -15.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 361.05 | 403.71 | 403.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 357.05 | 389.13 | 395.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 348.65 | 345.76 | 363.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:30:00 | 352.10 | 345.76 | 363.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 359.30 | 347.19 | 360.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 361.05 | 347.19 | 360.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 362.60 | 347.34 | 360.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 359.85 | 350.13 | 360.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 357.05 | 350.35 | 360.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:00:00 | 358.95 | 350.57 | 360.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:30:00 | 359.30 | 350.74 | 360.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 360.75 | 351.03 | 360.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:45:00 | 361.45 | 351.03 | 360.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 360.75 | 351.13 | 360.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 11:45:00 | 359.85 | 351.21 | 360.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:45:00 | 360.25 | 351.48 | 360.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 341.86 | 351.17 | 360.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 339.20 | 351.17 | 360.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 341.00 | 351.17 | 360.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 341.33 | 351.17 | 360.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 341.86 | 351.17 | 360.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:15:00 | 342.24 | 351.17 | 360.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 352.90 | 350.83 | 359.64 | SL hit (close>ema200) qty=0.50 sl=350.83 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 363.65 | 343.92 | 343.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 366.00 | 344.14 | 343.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 385.30 | 385.55 | 373.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 10:00:00 | 385.30 | 385.55 | 373.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 379.55 | 386.41 | 379.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 380.35 | 386.11 | 379.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 380.65 | 386.06 | 379.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 381.00 | 385.95 | 379.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 10:15:00 | 418.39 | 391.16 | 383.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 346.15 | 391.90 | 391.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 345.10 | 390.54 | 391.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 351.65 | 348.74 | 361.77 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 14:15:00 | 348.50 | 348.80 | 361.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 360.50 | 349.27 | 361.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 361.05 | 349.27 | 361.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 359.35 | 349.37 | 361.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 360.95 | 349.37 | 361.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 361.55 | 349.50 | 361.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 361.55 | 349.50 | 361.16 | SL hit (close>ema400) qty=1.00 sl=361.16 alert=retest1 |

### Cycle 4 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 382.75 | 359.33 | 359.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 390.90 | 359.65 | 359.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 422.65 | 422.87 | 409.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 10:30:00 | 422.50 | 422.87 | 409.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 410.15 | 422.45 | 410.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 410.40 | 422.45 | 410.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 407.60 | 422.30 | 410.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 404.25 | 422.30 | 410.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 413.55 | 422.22 | 410.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 12:30:00 | 415.05 | 422.15 | 410.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 14:30:00 | 414.20 | 421.98 | 410.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 417.65 | 421.89 | 410.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:00:00 | 414.65 | 421.71 | 410.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 412.00 | 421.24 | 410.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 411.05 | 421.24 | 410.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2026-02-09 14:15:00 | 456.56 | 430.09 | 419.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 418.00 | 434.21 | 434.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 413.75 | 433.70 | 434.00 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-09 14:00:00 | 359.85 | 2024-12-13 09:15:00 | 341.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 09:15:00 | 357.05 | 2024-12-13 09:15:00 | 339.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 12:00:00 | 358.95 | 2024-12-13 09:15:00 | 341.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 13:30:00 | 359.30 | 2024-12-13 09:15:00 | 341.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 11:45:00 | 359.85 | 2024-12-13 09:15:00 | 341.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 14:45:00 | 360.25 | 2024-12-13 09:15:00 | 342.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 14:00:00 | 359.85 | 2024-12-16 09:15:00 | 352.90 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2024-12-10 09:15:00 | 357.05 | 2024-12-16 09:15:00 | 352.90 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2024-12-10 12:00:00 | 358.95 | 2024-12-16 09:15:00 | 352.90 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2024-12-10 13:30:00 | 359.30 | 2024-12-16 09:15:00 | 352.90 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2024-12-11 11:45:00 | 359.85 | 2024-12-16 09:15:00 | 352.90 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2024-12-11 14:45:00 | 360.25 | 2024-12-16 09:15:00 | 352.90 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2025-01-22 09:15:00 | 353.85 | 2025-01-23 09:15:00 | 364.55 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-01-27 09:15:00 | 357.45 | 2025-02-01 11:15:00 | 339.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 357.45 | 2025-02-01 13:15:00 | 350.50 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2025-01-27 13:15:00 | 350.80 | 2025-02-04 09:15:00 | 354.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-01-27 13:45:00 | 350.75 | 2025-02-04 09:15:00 | 354.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-01-29 11:00:00 | 350.30 | 2025-02-04 09:15:00 | 354.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-02-01 15:15:00 | 347.80 | 2025-02-04 09:15:00 | 354.50 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-02-03 09:15:00 | 343.90 | 2025-02-04 12:15:00 | 357.65 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-02-03 11:45:00 | 344.85 | 2025-02-04 12:15:00 | 357.65 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-02-03 13:45:00 | 346.15 | 2025-02-04 12:15:00 | 357.65 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-02-03 14:30:00 | 347.20 | 2025-02-04 12:15:00 | 357.65 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-02-11 10:15:00 | 346.65 | 2025-02-13 10:15:00 | 355.40 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-02-11 13:00:00 | 345.95 | 2025-02-13 10:15:00 | 355.40 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-02-11 14:00:00 | 346.20 | 2025-02-13 10:15:00 | 355.40 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-02-12 09:15:00 | 340.70 | 2025-02-13 10:15:00 | 355.40 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2025-03-12 10:15:00 | 327.65 | 2025-03-21 11:15:00 | 342.25 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-03-13 11:15:00 | 328.85 | 2025-03-21 12:15:00 | 345.20 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2025-03-17 10:00:00 | 329.75 | 2025-03-21 12:15:00 | 345.20 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2025-03-17 11:00:00 | 328.10 | 2025-03-21 12:15:00 | 345.20 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2025-03-20 10:45:00 | 334.90 | 2025-03-21 12:15:00 | 345.20 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-03-28 12:45:00 | 333.75 | 2025-04-01 09:15:00 | 359.30 | STOP_HIT | 1.00 | -7.66% |
| SELL | retest2 | 2025-03-28 15:00:00 | 334.70 | 2025-04-01 09:15:00 | 359.30 | STOP_HIT | 1.00 | -7.35% |
| BUY | retest2 | 2025-06-13 14:15:00 | 380.35 | 2025-06-25 10:15:00 | 418.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 15:00:00 | 380.65 | 2025-06-25 10:15:00 | 418.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 09:30:00 | 381.00 | 2025-06-25 10:15:00 | 419.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-09-10 14:15:00 | 348.50 | 2025-09-12 11:15:00 | 361.55 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-09-23 11:00:00 | 353.80 | 2025-10-23 09:15:00 | 361.65 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-09-26 10:00:00 | 351.80 | 2025-10-23 09:15:00 | 361.65 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-10-06 09:15:00 | 351.20 | 2025-10-23 09:15:00 | 361.65 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-10-06 11:45:00 | 352.60 | 2025-10-23 09:15:00 | 361.65 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-08 10:30:00 | 355.30 | 2025-10-24 14:15:00 | 362.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-08 14:45:00 | 355.30 | 2025-10-24 14:15:00 | 362.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-09 09:30:00 | 354.85 | 2025-10-24 14:15:00 | 362.20 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-10-09 10:00:00 | 354.75 | 2025-10-24 14:15:00 | 362.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-01-21 12:30:00 | 415.05 | 2026-02-09 14:15:00 | 456.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 14:30:00 | 414.20 | 2026-02-09 14:15:00 | 455.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-22 09:15:00 | 417.65 | 2026-02-09 14:15:00 | 456.12 | TARGET_HIT | 1.00 | 9.21% |
| BUY | retest2 | 2026-01-22 12:00:00 | 414.65 | 2026-02-10 09:15:00 | 459.42 | TARGET_HIT | 1.00 | 10.80% |
| BUY | retest2 | 2026-03-10 11:00:00 | 445.25 | 2026-03-13 10:15:00 | 430.80 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-03-10 11:30:00 | 445.20 | 2026-03-13 10:15:00 | 430.80 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-03-10 13:30:00 | 445.90 | 2026-03-13 10:15:00 | 430.80 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2026-03-11 09:15:00 | 449.45 | 2026-03-13 10:15:00 | 430.80 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2026-03-12 10:30:00 | 438.60 | 2026-03-13 11:15:00 | 429.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-03-12 11:45:00 | 440.40 | 2026-03-13 11:15:00 | 429.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2026-03-18 09:15:00 | 439.30 | 2026-03-19 10:15:00 | 430.45 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-18 10:30:00 | 438.60 | 2026-03-19 10:15:00 | 430.45 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-03-20 09:15:00 | 437.65 | 2026-03-23 09:15:00 | 417.50 | STOP_HIT | 1.00 | -4.60% |
| BUY | retest2 | 2026-04-08 10:30:00 | 436.40 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-04-09 10:45:00 | 436.00 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-04-09 12:00:00 | 435.75 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-04-13 10:15:00 | 434.20 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-04-13 10:45:00 | 435.00 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-15 09:30:00 | 436.40 | 2026-04-15 10:15:00 | 425.00 | STOP_HIT | 1.00 | -2.61% |
