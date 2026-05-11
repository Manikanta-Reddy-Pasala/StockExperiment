# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3597 bars)
- **Last close:** 409.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 8 |
| TARGET_HIT | 8 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 11
- **Target hits / Stop hits / Partials:** 8 / 16 / 8
- **Avg / median % per leg:** 3.08% / 4.78%
- **Sum % (uncompounded):** 98.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 3 | 1 | 0 | 7.36% | 29.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 3 | 75.0% | 3 | 1 | 0 | 7.36% | 29.4% |
| SELL (all) | 28 | 18 | 64.3% | 5 | 15 | 8 | 2.47% | 69.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 18 | 64.3% | 5 | 15 | 8 | 2.47% | 69.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 21 | 65.6% | 8 | 16 | 8 | 3.08% | 98.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 471.40 | 460.41 | 460.38 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 456.05 | 460.34 | 460.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 445.65 | 460.13 | 460.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 456.70 | 456.67 | 458.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:00:00 | 456.70 | 456.67 | 458.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 456.60 | 456.25 | 458.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 457.20 | 456.25 | 458.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 456.90 | 456.18 | 457.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 456.90 | 456.18 | 457.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 460.80 | 456.14 | 457.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 460.80 | 456.14 | 457.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 461.45 | 456.20 | 457.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 461.00 | 456.20 | 457.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 491.95 | 456.61 | 458.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 491.95 | 456.61 | 458.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 13:15:00 | 510.00 | 459.86 | 459.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 521.80 | 461.77 | 460.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 476.00 | 476.30 | 469.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:30:00 | 475.50 | 476.30 | 469.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 468.20 | 476.19 | 469.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 468.20 | 476.19 | 469.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 467.00 | 476.10 | 469.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 465.45 | 476.10 | 469.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 466.65 | 476.01 | 469.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 466.45 | 476.01 | 469.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 466.45 | 475.50 | 469.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 466.45 | 475.50 | 469.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 469.85 | 475.45 | 469.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 472.00 | 475.09 | 469.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 471.00 | 475.39 | 470.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:00:00 | 470.85 | 474.68 | 470.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-22 10:15:00 | 518.10 | 477.88 | 473.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 470.75 | 484.50 | 484.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 467.50 | 481.60 | 482.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 493.05 | 466.94 | 474.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 493.05 | 466.94 | 474.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 501.50 | 467.28 | 474.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 480.50 | 468.00 | 474.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 15:15:00 | 456.47 | 467.97 | 474.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 460.60 | 457.40 | 465.65 | SL hit (close>ema200) qty=0.50 sl=457.40 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 483.65 | 469.78 | 469.77 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 454.35 | 469.79 | 469.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 453.00 | 466.66 | 468.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 458.40 | 453.62 | 459.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:45:00 | 455.35 | 453.62 | 459.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 454.80 | 453.68 | 459.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 452.25 | 453.68 | 459.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:45:00 | 452.50 | 453.65 | 459.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:30:00 | 452.30 | 453.53 | 459.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:45:00 | 452.40 | 453.54 | 459.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 459.60 | 452.25 | 457.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 459.60 | 452.25 | 457.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 457.35 | 452.30 | 457.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 455.25 | 452.60 | 457.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 455.00 | 452.67 | 457.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 454.95 | 452.76 | 457.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:30:00 | 454.65 | 452.80 | 457.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 466.00 | 453.02 | 457.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 466.00 | 453.02 | 457.75 | SL hit (close>static) qty=1.00 sl=460.75 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-03 09:30:00 | 472.00 | 2025-07-22 10:15:00 | 518.10 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2025-07-07 14:45:00 | 471.00 | 2025-07-22 10:15:00 | 517.94 | TARGET_HIT | 1.00 | 9.96% |
| BUY | retest2 | 2025-07-09 10:00:00 | 470.85 | 2025-07-22 11:15:00 | 519.20 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2025-08-28 09:45:00 | 473.40 | 2025-09-10 12:15:00 | 470.75 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-07 15:15:00 | 480.50 | 2025-10-08 15:15:00 | 456.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 15:15:00 | 480.50 | 2025-10-29 09:15:00 | 460.60 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-11-14 09:45:00 | 485.20 | 2025-11-19 11:15:00 | 483.65 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-14 12:15:00 | 486.10 | 2025-11-19 11:15:00 | 483.65 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-11-18 09:15:00 | 485.90 | 2025-11-19 11:15:00 | 483.65 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-12-22 10:15:00 | 452.25 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-12-22 11:45:00 | 452.50 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-12-23 09:30:00 | 452.30 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-12-23 11:45:00 | 452.40 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-01-01 10:00:00 | 455.25 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-01-01 11:30:00 | 455.00 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-01-02 09:15:00 | 454.95 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-01-02 11:30:00 | 454.65 | 2026-01-05 09:15:00 | 466.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-01-08 15:15:00 | 462.60 | 2026-01-12 09:15:00 | 439.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 462.60 | 2026-01-14 11:15:00 | 459.10 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2026-01-14 13:15:00 | 461.55 | 2026-01-20 10:15:00 | 439.33 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2026-01-14 14:00:00 | 462.45 | 2026-01-20 12:15:00 | 438.47 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2026-01-14 13:15:00 | 461.55 | 2026-01-21 14:15:00 | 415.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 14:00:00 | 462.45 | 2026-01-21 14:15:00 | 416.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 14:00:00 | 460.00 | 2026-02-13 14:15:00 | 437.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 14:00:00 | 460.00 | 2026-02-16 09:15:00 | 463.50 | STOP_HIT | 0.50 | -0.76% |
| SELL | retest2 | 2026-02-16 11:30:00 | 458.60 | 2026-02-17 09:15:00 | 479.10 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2026-02-20 09:15:00 | 458.75 | 2026-02-24 12:15:00 | 436.81 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2026-02-20 11:00:00 | 459.80 | 2026-02-24 13:15:00 | 435.81 | PARTIAL | 0.50 | 5.22% |
| SELL | retest2 | 2026-02-20 11:30:00 | 458.00 | 2026-02-24 13:15:00 | 435.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 09:15:00 | 458.75 | 2026-03-02 12:15:00 | 413.82 | TARGET_HIT | 0.50 | 9.79% |
| SELL | retest2 | 2026-02-20 11:00:00 | 459.80 | 2026-03-02 13:15:00 | 412.88 | TARGET_HIT | 0.50 | 10.21% |
| SELL | retest2 | 2026-02-20 11:30:00 | 458.00 | 2026-03-02 13:15:00 | 412.20 | TARGET_HIT | 0.50 | 10.00% |
