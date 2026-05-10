# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 408.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 18
- **Target hits / Stop hits / Partials:** 7 / 18 / 5
- **Avg / median % per leg:** 2.35% / -0.79%
- **Sum % (uncompounded):** 70.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 2 | 10.0% | 2 | 18 | 0 | -0.22% | -4.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 2 | 18 | 0 | -0.22% | -4.4% |
| SELL (all) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 12 | 40.0% | 7 | 18 | 5 | 2.35% | 70.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 445.95 | 457.99 | 458.03 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 459.70 | 457.97 | 457.97 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 457.85 | 457.97 | 457.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 456.75 | 457.96 | 457.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 458.95 | 457.97 | 457.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 459.50 | 457.98 | 457.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 465.40 | 458.14 | 458.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 459.90 | 460.38 | 459.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 458.85 | 460.37 | 459.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 457.50 | 460.37 | 459.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 458.70 | 460.35 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 458.70 | 460.35 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 458.50 | 460.33 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:45:00 | 458.55 | 460.33 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 460.10 | 460.33 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 458.65 | 460.33 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 458.55 | 460.31 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 460.00 | 460.31 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 457.25 | 460.28 | 459.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 460.50 | 460.28 | 459.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 454.10 | 460.18 | 459.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 454.25 | 460.18 | 459.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 459.10 | 459.82 | 459.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 467.95 | 459.62 | 458.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 456.35 | 459.67 | 459.03 | SL hit (close<static) qty=1.00 sl=456.95 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 450.20 | 458.42 | 458.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 448.20 | 457.25 | 457.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 455.75 | 448.07 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 455.05 | 448.14 | 452.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:45:00 | 457.05 | 448.14 | 452.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 457.65 | 454.02 | 454.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 459.85 | 454.02 | 454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 458.50 | 454.34 | 454.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 458.50 | 454.34 | 454.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 467.75 | 455.33 | 455.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 485.00 | 455.75 | 455.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 475.30 | 476.68 | 469.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 13:00:00 | 475.30 | 476.68 | 469.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 467.50 | 476.60 | 469.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 467.50 | 476.60 | 469.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 473.20 | 476.57 | 469.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 474.75 | 476.57 | 469.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 465.60 | 476.42 | 469.36 | SL hit (close<static) qty=1.00 sl=467.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 473.90 | 476.42 | 469.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 467.10 | 476.33 | 469.35 | SL hit (close<static) qty=1.00 sl=467.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 473.40 | 475.19 | 469.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 473.30 | 475.17 | 469.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 09:15:00 | 520.74 | 487.57 | 478.83 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-28 09:15:00 | 520.63 | 487.57 | 478.83 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 506.95 | 516.80 | 506.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 507.35 | 516.80 | 506.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 506.60 | 516.69 | 506.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 505.40 | 516.69 | 506.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 506.25 | 516.59 | 506.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 506.40 | 516.59 | 506.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 506.80 | 516.49 | 506.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 506.80 | 516.49 | 506.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 506.20 | 516.39 | 506.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 506.60 | 516.39 | 506.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 506.00 | 516.29 | 506.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 506.20 | 516.29 | 506.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 507.05 | 516.20 | 506.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 510.65 | 516.20 | 506.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 504.90 | 515.69 | 506.79 | SL hit (close<static) qty=1.00 sl=505.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 509.00 | 513.36 | 506.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 507.35 | 513.30 | 506.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 507.60 | 513.24 | 506.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 506.20 | 513.15 | 506.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 506.20 | 513.15 | 506.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 506.30 | 513.08 | 506.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 504.20 | 513.08 | 506.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 509.40 | 513.04 | 506.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:45:00 | 510.40 | 512.97 | 506.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 510.90 | 512.92 | 506.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 510.00 | 512.89 | 506.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:00:00 | 514.00 | 512.87 | 506.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 507.15 | 512.66 | 506.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 507.05 | 512.66 | 506.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 507.00 | 512.61 | 506.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 507.00 | 512.61 | 506.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 507.30 | 512.55 | 506.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:15:00 | 507.05 | 512.55 | 506.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 507.35 | 512.50 | 506.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 505.00 | 512.31 | 506.75 | SL hit (close<static) qty=1.00 sl=505.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 505.00 | 512.31 | 506.75 | SL hit (close<static) qty=1.00 sl=505.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 505.00 | 512.31 | 506.75 | SL hit (close<static) qty=1.00 sl=505.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 507.85 | 512.31 | 506.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:45:00 | 507.95 | 512.23 | 506.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 505.00 | 512.15 | 506.75 | SL hit (close<static) qty=1.00 sl=506.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 505.00 | 512.15 | 506.75 | SL hit (close<static) qty=1.00 sl=506.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 500.90 | 512.04 | 506.72 | SL hit (close<static) qty=1.00 sl=503.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 500.90 | 512.04 | 506.72 | SL hit (close<static) qty=1.00 sl=503.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 500.90 | 512.04 | 506.72 | SL hit (close<static) qty=1.00 sl=503.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 500.90 | 512.04 | 506.72 | SL hit (close<static) qty=1.00 sl=503.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 508.95 | 507.83 | 505.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 509.55 | 507.85 | 505.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 507.05 | 510.10 | 506.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 507.05 | 510.10 | 506.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 504.50 | 510.05 | 506.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 504.50 | 510.05 | 506.82 | SL hit (close<static) qty=1.00 sl=506.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 504.50 | 510.05 | 506.82 | SL hit (close<static) qty=1.00 sl=506.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-09 14:45:00 | 504.85 | 510.05 | 506.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 504.90 | 510.00 | 506.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 490.75 | 510.00 | 506.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 507.90 | 510.80 | 507.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 507.10 | 510.80 | 507.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 509.65 | 510.79 | 507.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 505.95 | 510.79 | 507.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 506.75 | 510.75 | 507.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 506.75 | 510.75 | 507.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 512.35 | 510.76 | 507.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 509.60 | 510.76 | 507.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 510.65 | 510.77 | 507.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 514.50 | 509.77 | 507.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 503.60 | 509.67 | 507.39 | SL hit (close<static) qty=1.00 sl=504.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 491.15 | 505.55 | 505.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 487.10 | 504.58 | 505.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 503.00 | 502.82 | 504.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 502.60 | 502.82 | 504.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:30:00 | 503.40 | 502.82 | 504.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 504.20 | 502.84 | 504.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 504.20 | 502.84 | 504.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 505.25 | 502.86 | 504.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 515.90 | 502.86 | 504.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 514.75 | 502.98 | 504.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 517.40 | 502.98 | 504.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 512.00 | 503.07 | 504.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:15:00 | 509.90 | 503.17 | 504.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:30:00 | 511.50 | 503.58 | 504.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 10:00:00 | 511.50 | 503.58 | 504.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 510.10 | 503.90 | 504.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 504.00 | 504.22 | 504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:15:00 | 508.50 | 504.22 | 504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 506.95 | 504.25 | 504.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 502.00 | 504.67 | 504.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 484.40 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 485.92 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 485.92 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 484.60 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:15:00 | 476.90 | 502.79 | 503.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-19 15:15:00 | 460.35 | 496.92 | 500.71 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-19 15:15:00 | 460.35 | 496.92 | 500.71 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-20 09:15:00 | 458.91 | 496.56 | 500.51 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-20 09:15:00 | 459.09 | 496.56 | 500.51 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-23 09:15:00 | 451.80 | 493.68 | 498.91 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-05 15:15:00 | 464.65 | 2025-06-12 09:15:00 | 459.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-06 10:00:00 | 465.45 | 2025-06-12 09:15:00 | 459.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-17 09:30:00 | 467.95 | 2025-07-18 09:15:00 | 456.35 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-29 13:15:00 | 474.75 | 2025-09-29 14:15:00 | 465.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-29 14:30:00 | 473.90 | 2025-09-29 15:15:00 | 467.10 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-03 12:00:00 | 473.40 | 2025-10-28 09:15:00 | 520.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 13:15:00 | 473.30 | 2025-10-28 09:15:00 | 520.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-16 09:15:00 | 510.65 | 2025-12-17 09:15:00 | 504.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-19 10:30:00 | 509.00 | 2025-12-26 15:15:00 | 505.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-19 12:15:00 | 507.35 | 2025-12-26 15:15:00 | 505.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-19 13:00:00 | 507.60 | 2025-12-26 15:15:00 | 505.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-22 11:45:00 | 510.40 | 2025-12-29 11:15:00 | 505.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-22 13:45:00 | 510.90 | 2025-12-29 11:15:00 | 505.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-22 15:15:00 | 510.00 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-23 10:00:00 | 514.00 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-12-29 09:15:00 | 507.85 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-29 10:45:00 | 507.95 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-01-06 09:15:00 | 508.95 | 2026-01-09 14:15:00 | 504.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-01-06 10:00:00 | 509.55 | 2026-01-09 14:15:00 | 504.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-01-22 09:15:00 | 514.50 | 2026-01-22 11:15:00 | 503.60 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-02-04 12:15:00 | 509.90 | 2026-02-16 09:15:00 | 484.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 09:30:00 | 511.50 | 2026-02-16 09:15:00 | 485.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 10:00:00 | 511.50 | 2026-02-16 09:15:00 | 485.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 11:00:00 | 510.10 | 2026-02-16 09:15:00 | 484.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 502.00 | 2026-02-16 12:15:00 | 476.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 12:15:00 | 509.90 | 2026-02-19 15:15:00 | 460.35 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-05 09:30:00 | 511.50 | 2026-02-19 15:15:00 | 460.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-05 10:00:00 | 511.50 | 2026-02-20 09:15:00 | 458.91 | TARGET_HIT | 0.50 | 10.28% |
| SELL | retest2 | 2026-02-09 11:00:00 | 510.10 | 2026-02-20 09:15:00 | 459.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 502.00 | 2026-02-23 09:15:00 | 451.80 | TARGET_HIT | 0.50 | 10.00% |
