# Indus Towers Ltd. (INDUSTOWER)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 402.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 30 |
| PARTIAL | 0 |
| TARGET_HIT | 7 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 24
- **Target hits / Stop hits / Partials:** 7 / 24 / 0
- **Avg / median % per leg:** 0.19% / -2.22%
- **Sum % (uncompounded):** 5.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 7 | 31.8% | 7 | 15 | 0 | 1.28% | 28.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 7 | 31.8% | 7 | 15 | 0 | 1.28% | 28.1% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.48% | -22.4% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.74% | -3.7% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.33% | -18.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.74% | -3.7% |
| retest2 (combined) | 30 | 7 | 23.3% | 7 | 23 | 0 | 0.32% | 9.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 13:15:00)

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
| ALERT3_SIDEWAYS | 2025-09-12 12:15:00 | 362.35 | 349.50 | 361.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 362.95 | 349.63 | 361.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 362.50 | 349.63 | 361.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 361.60 | 349.85 | 361.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 361.50 | 349.85 | 361.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 362.00 | 349.97 | 361.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 366.15 | 349.97 | 361.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 361.55 | 351.81 | 360.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 361.55 | 351.81 | 360.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 355.60 | 351.85 | 360.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:00:00 | 353.80 | 352.44 | 360.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 351.80 | 353.52 | 360.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 351.20 | 351.96 | 358.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:45:00 | 352.60 | 351.96 | 358.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 355.95 | 352.06 | 357.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 357.70 | 352.06 | 357.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 359.20 | 352.13 | 357.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 359.20 | 352.13 | 357.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 357.45 | 352.18 | 357.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:15:00 | 357.90 | 352.18 | 357.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 357.90 | 352.24 | 357.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 357.75 | 352.24 | 357.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 355.85 | 352.28 | 357.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:30:00 | 355.30 | 352.30 | 357.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:45:00 | 355.30 | 352.47 | 357.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 354.85 | 352.51 | 357.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 354.75 | 352.51 | 357.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 356.85 | 352.71 | 357.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 356.85 | 352.71 | 357.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 357.50 | 352.75 | 357.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 357.50 | 352.75 | 357.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 354.70 | 350.53 | 355.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 354.70 | 350.53 | 355.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 354.25 | 350.62 | 355.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 355.45 | 350.62 | 355.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 355.25 | 350.66 | 355.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 358.60 | 350.75 | 355.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 359.90 | 350.84 | 355.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 362.75 | 350.84 | 355.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 361.65 | 350.94 | 355.65 | SL hit (close>static) qty=1.00 sl=360.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 361.65 | 350.94 | 355.65 | SL hit (close>static) qty=1.00 sl=360.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 361.65 | 350.94 | 355.65 | SL hit (close>static) qty=1.00 sl=360.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 361.65 | 350.94 | 355.65 | SL hit (close>static) qty=1.00 sl=360.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 362.20 | 351.95 | 355.89 | SL hit (close>static) qty=1.00 sl=361.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 362.20 | 351.95 | 355.89 | SL hit (close>static) qty=1.00 sl=361.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 362.20 | 351.95 | 355.89 | SL hit (close>static) qty=1.00 sl=361.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 362.20 | 351.95 | 355.89 | SL hit (close>static) qty=1.00 sl=361.70 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-03 15:15:00)

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
| Target hit | 2026-02-09 14:15:00 | 455.62 | 430.09 | 419.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-09 14:15:00 | 456.12 | 430.09 | 419.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-10 09:15:00 | 459.42 | 430.63 | 419.38 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 437.30 | 452.99 | 438.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 437.30 | 452.99 | 438.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 438.50 | 452.85 | 438.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:30:00 | 432.05 | 452.85 | 438.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 435.50 | 452.67 | 438.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:00:00 | 435.50 | 452.67 | 438.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 439.30 | 452.54 | 438.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:30:00 | 435.65 | 452.54 | 438.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 439.25 | 452.15 | 439.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 445.25 | 451.16 | 439.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:30:00 | 445.20 | 451.10 | 439.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 13:30:00 | 445.90 | 450.97 | 439.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 449.45 | 450.85 | 439.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 438.00 | 450.42 | 439.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 438.00 | 450.42 | 439.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 438.20 | 450.30 | 439.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 434.45 | 450.30 | 439.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 435.15 | 450.15 | 439.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 438.60 | 450.06 | 439.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 440.40 | 449.97 | 439.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 430.80 | 449.32 | 439.57 | SL hit (close<static) qty=1.00 sl=434.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 430.80 | 449.32 | 439.57 | SL hit (close<static) qty=1.00 sl=434.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 430.80 | 449.32 | 439.57 | SL hit (close<static) qty=1.00 sl=434.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 430.80 | 449.32 | 439.57 | SL hit (close<static) qty=1.00 sl=434.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 429.00 | 449.11 | 439.52 | SL hit (close<static) qty=1.00 sl=430.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 429.00 | 449.11 | 439.52 | SL hit (close<static) qty=1.00 sl=430.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 439.30 | 445.78 | 438.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:30:00 | 438.60 | 445.62 | 438.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 438.75 | 445.56 | 438.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 438.75 | 445.56 | 438.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 440.20 | 445.50 | 438.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:30:00 | 438.30 | 445.50 | 438.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 439.00 | 445.36 | 438.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 433.50 | 445.36 | 438.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 431.00 | 445.22 | 438.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 10:15:00 | 430.45 | 445.07 | 438.55 | SL hit (close<static) qty=1.00 sl=430.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 10:15:00 | 430.45 | 445.07 | 438.55 | SL hit (close<static) qty=1.00 sl=430.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 437.65 | 444.22 | 438.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 417.50 | 443.49 | 438.14 | SL hit (close<static) qty=1.00 sl=429.25 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:30:00 | 436.40 | 434.33 | 434.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:45:00 | 436.00 | 434.57 | 434.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 435.75 | 434.59 | 434.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 438.10 | 434.95 | 434.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 435.70 | 434.95 | 434.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 432.55 | 434.97 | 434.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 434.20 | 434.97 | 434.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 435.00 | 434.97 | 434.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 436.40 | 435.05 | 434.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 434.95 | 434.63 | SL hit (close<static) qty=1.00 sl=429.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 434.95 | 434.63 | SL hit (close<static) qty=1.00 sl=429.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 434.95 | 434.63 | SL hit (close<static) qty=1.00 sl=429.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 434.95 | 434.63 | SL hit (close<static) qty=1.00 sl=425.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 434.95 | 434.63 | SL hit (close<static) qty=1.00 sl=425.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 425.00 | 434.95 | 434.63 | SL hit (close<static) qty=1.00 sl=425.60 alert=retest2 |

### Cycle 3 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 418.00 | 434.21 | 434.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 413.75 | 433.70 | 434.00 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
