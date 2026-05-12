# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 508.35
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 57 |
| ALERT2 | 56 |
| ALERT2_SKIP | 30 |
| ALERT3 | 138 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 52 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 38
- **Target hits / Stop hits / Partials:** 3 / 50 / 4
- **Avg / median % per leg:** 0.56% / -0.40%
- **Sum % (uncompounded):** 31.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 5 | 20.0% | 3 | 22 | 0 | 0.46% | 11.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.32% | -1.3% |
| BUY @ 3rd Alert (retest2) | 24 | 5 | 20.8% | 3 | 21 | 0 | 0.53% | 12.7% |
| SELL (all) | 32 | 14 | 43.8% | 0 | 28 | 4 | 0.64% | 20.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 14 | 43.8% | 0 | 28 | 4 | 0.64% | 20.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.32% | -1.3% |
| retest2 (combined) | 56 | 19 | 33.9% | 3 | 49 | 4 | 0.59% | 33.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 511.10 | 504.33 | 503.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 513.50 | 506.17 | 504.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 509.95 | 509.99 | 507.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 509.95 | 509.99 | 507.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 508.40 | 509.22 | 507.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 512.70 | 509.22 | 507.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:00:00 | 509.95 | 510.13 | 508.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:30:00 | 508.50 | 509.37 | 509.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 14:15:00 | 506.45 | 508.79 | 508.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 506.45 | 508.79 | 508.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 09:15:00 | 503.50 | 507.59 | 508.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 504.65 | 503.23 | 505.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 504.65 | 503.23 | 505.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 504.65 | 503.23 | 505.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 12:30:00 | 501.40 | 502.63 | 504.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 09:15:00 | 476.33 | 486.49 | 492.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 474.55 | 473.89 | 479.76 | SL hit (close>ema200) qty=0.50 sl=473.89 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 488.00 | 482.65 | 482.40 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 482.95 | 483.05 | 483.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 481.30 | 482.70 | 482.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 482.95 | 482.29 | 482.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 482.95 | 482.29 | 482.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 482.95 | 482.29 | 482.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 483.75 | 482.29 | 482.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 485.65 | 482.96 | 482.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 486.60 | 484.39 | 483.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 484.40 | 484.86 | 484.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 484.40 | 484.86 | 484.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 484.40 | 484.86 | 484.10 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 481.05 | 483.62 | 483.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 10:15:00 | 480.20 | 482.94 | 483.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 13:15:00 | 482.10 | 481.98 | 482.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:45:00 | 481.85 | 481.98 | 482.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 483.55 | 482.30 | 482.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 483.55 | 482.30 | 482.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 483.00 | 482.44 | 482.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 480.95 | 482.44 | 482.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 478.15 | 481.58 | 482.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 476.05 | 479.55 | 481.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 13:15:00 | 482.70 | 481.24 | 481.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 482.70 | 481.24 | 481.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 483.50 | 481.69 | 481.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 15:15:00 | 479.95 | 481.34 | 481.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 15:15:00 | 479.95 | 481.34 | 481.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 479.95 | 481.34 | 481.22 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 479.70 | 481.01 | 481.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 478.25 | 480.46 | 480.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 470.20 | 469.98 | 472.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 12:00:00 | 470.20 | 469.98 | 472.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 475.85 | 470.56 | 471.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 477.00 | 470.56 | 471.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 473.90 | 471.23 | 472.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:15:00 | 476.15 | 471.23 | 472.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 476.50 | 473.06 | 472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 15:15:00 | 477.60 | 474.97 | 473.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 477.25 | 478.30 | 476.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:00:00 | 477.25 | 478.30 | 476.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 476.55 | 477.95 | 476.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 476.55 | 477.95 | 476.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 476.50 | 477.66 | 476.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 476.50 | 477.66 | 476.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 472.30 | 476.22 | 476.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 471.45 | 475.26 | 475.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 475.15 | 474.86 | 475.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 475.15 | 474.86 | 475.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 475.15 | 474.86 | 475.47 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 480.40 | 476.42 | 476.02 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 468.95 | 475.02 | 475.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 461.90 | 469.38 | 472.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 471.95 | 468.22 | 470.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 14:15:00 | 471.95 | 468.22 | 470.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 471.95 | 468.22 | 470.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 471.95 | 468.22 | 470.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 475.00 | 469.58 | 470.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 474.50 | 469.58 | 470.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 474.70 | 470.99 | 471.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:45:00 | 476.10 | 470.99 | 471.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 473.50 | 471.49 | 471.36 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 470.00 | 471.21 | 471.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 467.75 | 470.24 | 470.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 470.40 | 468.92 | 469.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 14:15:00 | 470.40 | 468.92 | 469.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 470.40 | 468.92 | 469.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 470.40 | 468.92 | 469.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 468.55 | 468.84 | 469.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 461.45 | 468.84 | 469.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 462.95 | 460.44 | 460.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 462.95 | 460.44 | 460.42 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 12:15:00 | 460.70 | 461.47 | 461.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 455.80 | 459.92 | 460.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 458.90 | 458.61 | 459.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 462.60 | 458.61 | 459.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 462.40 | 459.37 | 459.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 462.15 | 459.37 | 459.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 463.40 | 460.68 | 460.39 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 456.05 | 460.21 | 460.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 454.85 | 457.20 | 458.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 453.35 | 452.57 | 454.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 453.35 | 452.57 | 454.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 453.35 | 452.57 | 454.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:30:00 | 455.45 | 452.57 | 454.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 452.70 | 452.43 | 454.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 452.25 | 452.43 | 454.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 453.85 | 452.73 | 454.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:00:00 | 451.95 | 453.54 | 454.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 451.65 | 452.53 | 453.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 455.60 | 454.22 | 454.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 455.60 | 454.22 | 454.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 15:15:00 | 456.25 | 454.86 | 454.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 458.40 | 460.43 | 458.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 10:15:00 | 458.40 | 460.43 | 458.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 458.40 | 460.43 | 458.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 458.40 | 460.43 | 458.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 460.05 | 460.36 | 458.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:15:00 | 459.00 | 460.36 | 458.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 458.95 | 460.07 | 458.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:15:00 | 458.80 | 460.07 | 458.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 458.20 | 459.70 | 458.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 467.50 | 459.20 | 458.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 15:15:00 | 460.15 | 461.48 | 461.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 460.15 | 461.48 | 461.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 458.35 | 460.85 | 461.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 455.90 | 454.91 | 457.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 455.90 | 454.91 | 457.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 455.30 | 454.99 | 457.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 456.25 | 454.99 | 457.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 460.10 | 456.28 | 457.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 460.10 | 456.28 | 457.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 460.40 | 457.10 | 457.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:30:00 | 460.05 | 457.10 | 457.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 461.55 | 458.61 | 458.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 466.45 | 460.18 | 458.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 485.70 | 486.65 | 480.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:15:00 | 484.75 | 486.65 | 480.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 484.55 | 487.73 | 485.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 486.50 | 487.73 | 485.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 485.00 | 487.18 | 485.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 485.00 | 487.18 | 485.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 485.05 | 486.75 | 485.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:45:00 | 484.55 | 486.75 | 485.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 485.10 | 486.42 | 485.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 485.95 | 485.85 | 485.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:45:00 | 486.15 | 486.82 | 486.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 11:30:00 | 487.25 | 486.64 | 486.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 486.50 | 486.43 | 486.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 15:15:00 | 484.60 | 486.06 | 486.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 484.60 | 486.06 | 486.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 483.95 | 485.24 | 485.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 485.20 | 481.88 | 483.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 485.20 | 481.88 | 483.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 485.20 | 481.88 | 483.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 486.60 | 481.88 | 483.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 484.40 | 482.38 | 483.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 486.05 | 482.38 | 483.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 485.00 | 482.90 | 483.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 486.00 | 482.90 | 483.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 484.55 | 483.50 | 483.91 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 14:15:00 | 487.05 | 484.21 | 484.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 489.85 | 485.37 | 484.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 515.55 | 516.52 | 507.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:30:00 | 520.70 | 517.77 | 509.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 517.00 | 519.23 | 514.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 514.30 | 519.23 | 514.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 513.85 | 517.38 | 514.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 513.85 | 517.38 | 514.64 | SL hit (close<ema400) qty=1.00 sl=514.64 alert=retest1 |

### Cycle 24 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 503.60 | 513.14 | 513.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 501.65 | 506.12 | 508.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 502.40 | 499.15 | 501.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 502.40 | 499.15 | 501.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 502.40 | 499.15 | 501.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 502.40 | 499.15 | 501.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 503.00 | 499.92 | 502.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 502.10 | 499.92 | 502.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 500.85 | 500.11 | 501.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 499.55 | 500.22 | 501.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 498.60 | 500.25 | 501.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 509.35 | 502.66 | 501.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 509.35 | 502.66 | 501.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 516.60 | 507.47 | 504.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 514.00 | 514.62 | 511.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 14:00:00 | 514.00 | 514.62 | 511.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 511.10 | 513.92 | 511.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 511.10 | 513.92 | 511.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 510.50 | 513.23 | 511.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 507.55 | 513.23 | 511.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 502.35 | 511.06 | 510.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 500.30 | 511.06 | 510.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 499.95 | 508.84 | 509.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 498.35 | 503.30 | 506.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 504.90 | 500.82 | 502.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 504.90 | 500.82 | 502.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 504.90 | 500.82 | 502.86 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 506.55 | 503.77 | 503.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 511.05 | 505.75 | 504.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 516.15 | 516.30 | 512.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 516.15 | 516.30 | 512.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 506.70 | 515.01 | 513.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 506.70 | 515.01 | 513.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 512.80 | 514.57 | 513.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 513.70 | 514.57 | 513.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 514.75 | 515.94 | 514.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 511.70 | 514.77 | 514.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 511.70 | 514.77 | 514.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 508.05 | 511.79 | 513.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 492.80 | 488.20 | 493.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:00:00 | 492.80 | 488.20 | 493.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 495.45 | 489.65 | 493.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 495.45 | 489.65 | 493.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 496.00 | 490.92 | 494.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 496.65 | 490.92 | 494.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 498.20 | 493.62 | 494.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 499.40 | 493.62 | 494.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 13:15:00 | 496.45 | 495.62 | 495.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 14:15:00 | 499.50 | 496.39 | 495.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 500.00 | 502.52 | 500.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 500.00 | 502.52 | 500.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 500.00 | 502.52 | 500.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 500.00 | 502.52 | 500.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 499.70 | 501.95 | 500.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 499.70 | 501.95 | 500.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 493.60 | 500.28 | 499.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 493.60 | 500.28 | 499.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 493.55 | 498.94 | 499.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 488.50 | 496.85 | 498.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 11:15:00 | 477.90 | 476.56 | 483.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 477.90 | 476.56 | 483.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 476.35 | 475.40 | 477.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 475.25 | 476.03 | 477.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 473.95 | 476.43 | 477.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 474.85 | 475.97 | 477.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:00:00 | 473.50 | 468.65 | 469.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 471.50 | 469.93 | 469.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 471.50 | 469.93 | 469.85 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 468.35 | 469.62 | 469.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 466.10 | 468.67 | 469.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 473.30 | 469.36 | 469.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 473.30 | 469.36 | 469.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 473.30 | 469.36 | 469.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 473.30 | 469.36 | 469.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 472.80 | 470.05 | 469.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 474.85 | 472.36 | 471.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 470.60 | 472.62 | 471.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 470.60 | 472.62 | 471.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 470.60 | 472.62 | 471.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:15:00 | 470.50 | 472.62 | 471.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 470.05 | 472.11 | 471.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 470.85 | 472.11 | 471.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 470.30 | 471.74 | 471.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 469.90 | 471.74 | 471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 466.15 | 470.28 | 470.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 463.65 | 468.96 | 470.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 12:15:00 | 454.55 | 454.48 | 457.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:00:00 | 454.55 | 454.48 | 457.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 450.65 | 447.96 | 450.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 450.65 | 447.96 | 450.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 452.60 | 448.89 | 450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 452.60 | 448.89 | 450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 451.70 | 449.45 | 450.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 447.55 | 449.45 | 450.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 446.10 | 448.78 | 450.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 444.85 | 446.81 | 448.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 444.75 | 445.37 | 447.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 14:15:00 | 445.15 | 438.27 | 437.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 445.15 | 438.27 | 437.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 447.70 | 441.39 | 439.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 443.40 | 444.78 | 442.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 443.40 | 444.78 | 442.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 443.60 | 444.40 | 442.46 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 439.05 | 441.55 | 441.75 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 12:15:00 | 443.15 | 442.00 | 441.93 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 440.50 | 441.65 | 441.79 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 443.25 | 441.97 | 441.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 444.50 | 442.92 | 442.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 457.25 | 460.61 | 456.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 457.25 | 460.61 | 456.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 457.25 | 460.61 | 456.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 456.65 | 460.61 | 456.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 459.40 | 460.36 | 457.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 459.70 | 458.36 | 457.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 470.20 | 458.54 | 457.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 458.10 | 461.30 | 461.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 458.10 | 461.30 | 461.44 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 467.65 | 462.41 | 461.91 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 15:15:00 | 460.30 | 461.68 | 461.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 453.75 | 460.09 | 461.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 467.90 | 458.13 | 458.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 467.90 | 458.13 | 458.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 467.90 | 458.13 | 458.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 467.90 | 458.13 | 458.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 466.70 | 459.85 | 459.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 479.95 | 463.87 | 461.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 479.00 | 483.76 | 476.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 479.00 | 483.76 | 476.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 476.60 | 482.46 | 478.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 477.35 | 482.46 | 478.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 472.50 | 480.47 | 477.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 471.60 | 480.47 | 477.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 471.80 | 478.74 | 477.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 471.80 | 478.74 | 477.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 469.90 | 475.07 | 475.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 468.10 | 470.78 | 471.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 471.30 | 470.38 | 471.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 15:15:00 | 471.30 | 470.38 | 471.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 471.30 | 470.38 | 471.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 466.95 | 470.38 | 471.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 465.50 | 467.72 | 468.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 462.00 | 460.05 | 459.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 462.00 | 460.05 | 459.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 462.35 | 460.68 | 460.29 | Break + close above crossover candle high |

### Cycle 46 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 455.25 | 459.84 | 460.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 454.00 | 457.45 | 458.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 456.30 | 454.86 | 456.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 456.30 | 454.86 | 456.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 456.30 | 454.86 | 456.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 456.30 | 454.86 | 456.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 453.50 | 454.59 | 455.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 453.30 | 454.10 | 455.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 455.00 | 449.22 | 449.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 455.00 | 449.22 | 449.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 461.40 | 451.65 | 450.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 481.50 | 482.50 | 477.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 473.05 | 479.85 | 478.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 473.05 | 479.85 | 478.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 473.05 | 479.85 | 478.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 478.00 | 479.48 | 478.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:30:00 | 481.90 | 479.88 | 478.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 484.20 | 479.05 | 478.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 475.70 | 479.90 | 479.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 475.70 | 479.90 | 479.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 470.85 | 476.26 | 477.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 472.25 | 469.35 | 472.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 472.25 | 469.35 | 472.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 472.25 | 469.35 | 472.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 472.25 | 469.35 | 472.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 469.30 | 469.34 | 472.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 467.10 | 471.40 | 472.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 475.20 | 471.94 | 472.17 | SL hit (close>static) qty=1.00 sl=472.35 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 475.80 | 472.71 | 472.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 478.55 | 473.82 | 473.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 474.20 | 475.31 | 474.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 10:15:00 | 474.20 | 475.31 | 474.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 474.20 | 475.31 | 474.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 474.20 | 475.31 | 474.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 476.10 | 475.48 | 474.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 474.05 | 475.48 | 474.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 475.50 | 477.42 | 475.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:30:00 | 481.45 | 477.15 | 476.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 482.90 | 477.15 | 476.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:45:00 | 481.20 | 479.59 | 477.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 474.10 | 477.36 | 477.23 | SL hit (close<static) qty=1.00 sl=474.20 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 471.55 | 476.16 | 476.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 469.75 | 474.88 | 476.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 471.80 | 470.67 | 473.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 471.80 | 470.67 | 473.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 470.85 | 470.70 | 473.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 470.35 | 470.70 | 472.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 468.70 | 471.33 | 472.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 470.25 | 470.68 | 472.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:30:00 | 470.25 | 470.55 | 471.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 475.55 | 471.31 | 471.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 475.55 | 471.31 | 471.84 | SL hit (close>static) qty=1.00 sl=474.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 476.50 | 472.35 | 472.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 480.10 | 473.90 | 472.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 480.20 | 480.53 | 477.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:45:00 | 480.20 | 480.53 | 477.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 477.55 | 480.40 | 478.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 477.55 | 480.40 | 478.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 475.80 | 479.48 | 478.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 475.80 | 479.48 | 478.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 478.40 | 478.38 | 478.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 477.50 | 478.38 | 478.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 477.50 | 478.20 | 478.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 477.70 | 478.20 | 478.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 477.60 | 478.08 | 478.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 478.25 | 478.08 | 478.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 479.50 | 478.36 | 478.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 480.65 | 478.67 | 478.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:15:00 | 481.05 | 478.67 | 478.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 488.50 | 489.33 | 489.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 488.50 | 489.33 | 489.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 486.15 | 488.54 | 489.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 10:15:00 | 489.00 | 488.63 | 489.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 489.00 | 488.63 | 489.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 489.00 | 488.63 | 489.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 488.50 | 488.63 | 489.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 494.45 | 489.79 | 489.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 498.15 | 492.81 | 491.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 502.05 | 503.84 | 499.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 502.05 | 503.84 | 499.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 502.20 | 503.51 | 499.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 502.20 | 503.51 | 499.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 501.25 | 503.06 | 499.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 500.45 | 503.06 | 499.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 501.90 | 502.83 | 500.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 498.40 | 502.83 | 500.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 500.50 | 502.36 | 500.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 500.50 | 502.36 | 500.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 501.40 | 501.87 | 500.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 497.15 | 501.87 | 500.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 500.15 | 501.53 | 500.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 502.50 | 501.53 | 500.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 494.90 | 500.20 | 499.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 493.45 | 500.20 | 499.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 492.60 | 498.68 | 499.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 490.20 | 496.99 | 498.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 492.55 | 491.86 | 494.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 493.30 | 491.86 | 494.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 494.25 | 492.05 | 494.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 494.25 | 492.05 | 494.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 492.00 | 492.04 | 493.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 492.15 | 492.04 | 493.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 494.25 | 492.48 | 494.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 496.55 | 492.48 | 494.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 497.90 | 493.56 | 494.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 497.90 | 493.56 | 494.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 496.50 | 494.15 | 494.55 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 12:15:00 | 497.55 | 494.83 | 494.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 500.90 | 496.44 | 495.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 15:15:00 | 500.50 | 501.73 | 499.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 09:15:00 | 497.70 | 501.73 | 499.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 504.40 | 502.26 | 500.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 499.60 | 502.26 | 500.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 500.50 | 501.91 | 500.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 500.50 | 501.91 | 500.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 501.65 | 501.79 | 500.34 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 493.50 | 498.69 | 499.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 490.75 | 494.81 | 496.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 484.00 | 481.73 | 486.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 484.00 | 481.73 | 486.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 484.00 | 481.73 | 486.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 484.60 | 481.73 | 486.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 485.00 | 482.38 | 486.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 485.00 | 482.38 | 486.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 486.10 | 483.22 | 486.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:00:00 | 486.10 | 483.22 | 486.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 486.20 | 483.82 | 486.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:00:00 | 485.35 | 485.41 | 486.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 484.85 | 485.30 | 486.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 484.90 | 485.50 | 486.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:15:00 | 461.08 | 469.13 | 474.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 468.00 | 467.77 | 472.42 | SL hit (close>ema200) qty=0.50 sl=467.77 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 469.70 | 468.50 | 468.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 09:15:00 | 474.85 | 470.49 | 469.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 464.40 | 469.27 | 469.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 464.40 | 469.27 | 469.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 464.40 | 469.27 | 469.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 464.40 | 469.27 | 469.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 467.00 | 468.82 | 468.86 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 475.90 | 468.88 | 468.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 487.85 | 475.16 | 471.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 12:15:00 | 450.00 | 470.12 | 469.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 12:15:00 | 450.00 | 470.12 | 469.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 450.00 | 470.12 | 469.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 13:00:00 | 450.00 | 470.12 | 469.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 13:15:00 | 449.50 | 466.00 | 467.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 439.90 | 445.09 | 452.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 439.05 | 438.73 | 444.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 439.05 | 438.73 | 444.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 439.35 | 439.60 | 444.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 13:15:00 | 438.05 | 439.60 | 444.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 14:45:00 | 438.40 | 439.22 | 443.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 446.40 | 441.02 | 443.14 | SL hit (close>static) qty=1.00 sl=444.80 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 453.65 | 444.95 | 444.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 456.70 | 447.30 | 445.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 456.00 | 456.56 | 452.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 455.60 | 456.56 | 452.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 455.00 | 455.91 | 454.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 455.10 | 455.91 | 454.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 454.15 | 455.56 | 454.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 451.70 | 455.56 | 454.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 452.45 | 454.94 | 454.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 454.40 | 454.67 | 454.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 454.40 | 454.67 | 454.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 447.00 | 453.49 | 453.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 447.00 | 453.49 | 453.85 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 457.35 | 453.30 | 452.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 463.00 | 458.26 | 456.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 459.00 | 459.89 | 458.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 460.00 | 459.89 | 458.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 456.25 | 459.16 | 457.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 456.25 | 459.16 | 457.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 457.50 | 458.83 | 457.87 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 453.55 | 457.26 | 457.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 451.80 | 456.17 | 456.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 455.45 | 454.96 | 455.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 13:15:00 | 455.45 | 454.96 | 455.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 455.45 | 454.96 | 455.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 455.55 | 454.96 | 455.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 453.65 | 454.70 | 455.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 453.25 | 454.70 | 455.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 456.55 | 454.96 | 455.49 | SL hit (close>static) qty=1.00 sl=455.75 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 456.50 | 455.81 | 455.73 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 454.00 | 455.45 | 455.58 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 462.00 | 456.59 | 455.96 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 13:15:00 | 456.10 | 456.66 | 456.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 451.00 | 455.53 | 456.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 437.30 | 434.08 | 440.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 437.30 | 434.08 | 440.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 440.80 | 436.31 | 440.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:15:00 | 441.80 | 436.31 | 440.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 442.70 | 437.59 | 440.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 442.70 | 437.59 | 440.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 441.55 | 438.38 | 440.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 442.10 | 438.38 | 440.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 450.85 | 442.92 | 442.34 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 437.15 | 443.54 | 443.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 432.95 | 436.59 | 438.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 409.35 | 407.27 | 414.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 409.35 | 407.27 | 414.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 409.35 | 407.27 | 414.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 413.95 | 407.27 | 414.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 407.35 | 407.15 | 410.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 405.20 | 406.73 | 410.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 415.00 | 408.78 | 409.54 | SL hit (close>static) qty=1.00 sl=411.10 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 416.30 | 411.29 | 410.61 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 407.80 | 410.52 | 410.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 406.15 | 409.13 | 410.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 387.75 | 387.15 | 393.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 387.75 | 387.15 | 393.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 392.50 | 388.22 | 393.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 393.30 | 388.22 | 393.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 391.70 | 388.91 | 393.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 393.80 | 388.91 | 393.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 403.15 | 391.53 | 393.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 403.15 | 391.53 | 393.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 403.90 | 394.00 | 394.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 403.65 | 394.00 | 394.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 405.15 | 396.23 | 395.23 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 391.90 | 396.09 | 396.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 389.45 | 394.19 | 395.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 395.55 | 388.73 | 390.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 395.55 | 388.73 | 390.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 395.55 | 388.73 | 390.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 396.90 | 388.73 | 390.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 393.90 | 389.77 | 391.07 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 403.60 | 393.73 | 392.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 404.80 | 400.09 | 397.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 400.65 | 401.40 | 398.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 400.65 | 401.40 | 398.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 397.20 | 400.41 | 399.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 400.50 | 399.80 | 399.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 400.80 | 399.80 | 399.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 401.65 | 399.93 | 399.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 440.55 | 432.69 | 428.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 511.05 | 514.28 | 514.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 509.00 | 513.22 | 513.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 14:15:00 | 512.30 | 508.18 | 510.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 512.30 | 508.18 | 510.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 512.30 | 508.18 | 510.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:30:00 | 511.90 | 508.18 | 510.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 512.70 | 509.08 | 510.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 506.90 | 509.08 | 510.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 507.95 | 506.97 | 508.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:45:00 | 507.70 | 506.97 | 508.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 508.70 | 507.31 | 508.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:15:00 | 509.05 | 507.31 | 508.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 509.05 | 507.66 | 508.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 510.75 | 507.66 | 508.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 508.25 | 507.78 | 508.59 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 514.80 | 509.67 | 509.33 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 507.00 | 509.32 | 509.52 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 512.70 | 2025-05-15 14:15:00 | 506.45 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-14 12:00:00 | 509.95 | 2025-05-15 14:15:00 | 506.45 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-05-15 13:30:00 | 508.50 | 2025-05-15 14:15:00 | 506.45 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-05-19 12:30:00 | 501.40 | 2025-05-21 09:15:00 | 476.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 12:30:00 | 501.40 | 2025-05-22 15:15:00 | 474.55 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2025-05-30 14:15:00 | 476.05 | 2025-06-02 13:15:00 | 482.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-06-19 09:15:00 | 461.45 | 2025-06-23 11:15:00 | 462.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-03 14:00:00 | 451.95 | 2025-07-04 13:15:00 | 455.60 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-07-04 09:30:00 | 451.65 | 2025-07-04 13:15:00 | 455.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-09 09:15:00 | 467.50 | 2025-07-10 15:15:00 | 460.15 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-23 09:15:00 | 485.95 | 2025-07-24 15:15:00 | 484.60 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-07-24 10:45:00 | 486.15 | 2025-07-24 15:15:00 | 484.60 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-24 11:30:00 | 487.25 | 2025-07-24 15:15:00 | 484.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-24 14:30:00 | 486.50 | 2025-07-24 15:15:00 | 484.60 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-31 11:30:00 | 520.70 | 2025-08-01 13:15:00 | 513.85 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-08 10:30:00 | 499.55 | 2025-08-11 13:15:00 | 509.35 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-08-08 14:15:00 | 498.60 | 2025-08-11 13:15:00 | 509.35 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-22 12:15:00 | 513.70 | 2025-08-26 10:15:00 | 511.70 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-08-25 09:30:00 | 514.75 | 2025-08-26 10:15:00 | 511.70 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-10 10:30:00 | 475.25 | 2025-09-17 14:15:00 | 471.50 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-09-11 09:15:00 | 473.95 | 2025-09-17 14:15:00 | 471.50 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-09-11 11:15:00 | 474.85 | 2025-09-17 14:15:00 | 471.50 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-09-17 13:00:00 | 473.50 | 2025-09-17 14:15:00 | 471.50 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-09-30 12:45:00 | 444.85 | 2025-10-09 14:15:00 | 445.15 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-01 10:15:00 | 444.75 | 2025-10-09 14:15:00 | 445.15 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-21 13:45:00 | 459.70 | 2025-10-27 09:15:00 | 458.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-10-23 09:15:00 | 470.20 | 2025-10-27 09:15:00 | 458.10 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-11-10 09:15:00 | 466.95 | 2025-11-17 11:15:00 | 462.00 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2025-11-12 09:15:00 | 465.50 | 2025-11-17 11:15:00 | 462.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-11-20 12:30:00 | 453.30 | 2025-11-26 09:15:00 | 455.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-12-03 11:30:00 | 481.90 | 2025-12-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-04 09:15:00 | 484.20 | 2025-12-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-11 09:45:00 | 467.10 | 2025-12-11 11:15:00 | 475.20 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-16 09:30:00 | 481.45 | 2025-12-17 09:15:00 | 474.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-12-16 10:15:00 | 482.90 | 2025-12-17 09:15:00 | 474.10 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-16 12:45:00 | 481.20 | 2025-12-17 09:15:00 | 474.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-18 13:30:00 | 470.35 | 2025-12-22 09:15:00 | 475.55 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-19 11:00:00 | 468.70 | 2025-12-22 09:15:00 | 475.55 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-12-19 12:45:00 | 470.25 | 2025-12-22 09:15:00 | 475.55 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-12-19 13:30:00 | 470.25 | 2025-12-22 09:15:00 | 475.55 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-26 11:45:00 | 480.65 | 2026-01-05 14:15:00 | 488.50 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2025-12-26 12:15:00 | 481.05 | 2026-01-05 14:15:00 | 488.50 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2026-01-23 11:00:00 | 485.35 | 2026-01-28 11:15:00 | 461.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:00:00 | 485.35 | 2026-01-28 14:15:00 | 468.00 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-01-23 12:00:00 | 484.85 | 2026-01-29 09:15:00 | 460.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 13:15:00 | 484.90 | 2026-01-29 09:15:00 | 460.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 12:00:00 | 484.85 | 2026-01-29 13:15:00 | 463.35 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2026-01-23 13:15:00 | 484.90 | 2026-01-29 13:15:00 | 463.35 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-06 13:15:00 | 438.05 | 2026-02-09 10:15:00 | 446.40 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-02-06 14:45:00 | 438.40 | 2026-02-09 10:15:00 | 446.40 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-02-12 12:45:00 | 454.40 | 2026-02-13 09:15:00 | 447.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-02-12 13:15:00 | 454.40 | 2026-02-13 09:15:00 | 447.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-20 15:15:00 | 453.25 | 2026-02-23 10:15:00 | 456.55 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-17 11:30:00 | 405.20 | 2026-03-18 10:15:00 | 415.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-04-07 13:30:00 | 400.50 | 2026-04-15 09:15:00 | 440.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:15:00 | 400.80 | 2026-04-15 09:15:00 | 440.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 15:15:00 | 401.65 | 2026-04-15 09:15:00 | 441.81 | TARGET_HIT | 1.00 | 10.00% |
