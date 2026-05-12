# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1975 bars)
- **Last close:** 409.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 24 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 57 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 46
- **Target hits / Stop hits / Partials:** 3 / 54 / 7
- **Avg / median % per leg:** 0.08% / -1.15%
- **Sum % (uncompounded):** 5.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 2 | 10.5% | 2 | 17 | 0 | -0.05% | -0.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 2 | 10.5% | 2 | 17 | 0 | -0.05% | -0.9% |
| SELL (all) | 45 | 16 | 35.6% | 1 | 37 | 7 | 0.13% | 5.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 16 | 35.6% | 1 | 37 | 7 | 0.13% | 5.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 64 | 18 | 28.1% | 3 | 54 | 7 | 0.08% | 5.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 449.00 | 443.10 | 443.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 457.15 | 447.20 | 445.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 449.30 | 450.37 | 448.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 449.30 | 450.37 | 448.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 449.70 | 450.10 | 448.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 449.70 | 450.10 | 448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 448.55 | 449.79 | 448.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 463.00 | 449.79 | 448.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 460.60 | 466.03 | 466.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 460.60 | 466.03 | 466.76 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 12:15:00 | 469.75 | 466.11 | 465.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 471.45 | 467.79 | 466.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 10:15:00 | 464.80 | 467.77 | 467.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 464.80 | 467.77 | 467.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 464.80 | 467.77 | 467.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 464.80 | 467.77 | 467.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 466.65 | 467.55 | 467.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 466.25 | 467.55 | 467.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 461.80 | 466.40 | 466.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 460.00 | 465.12 | 465.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 465.25 | 464.47 | 465.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 465.25 | 464.47 | 465.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 465.25 | 464.47 | 465.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:15:00 | 467.60 | 464.47 | 465.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 467.60 | 465.10 | 465.60 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 466.85 | 466.04 | 465.96 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 463.70 | 465.77 | 465.86 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 471.40 | 466.25 | 465.84 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 456.95 | 465.24 | 465.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 445.65 | 455.38 | 458.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 445.65 | 441.33 | 444.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 445.65 | 441.33 | 444.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 445.65 | 441.33 | 444.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 445.65 | 441.33 | 444.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 454.00 | 443.86 | 445.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 460.75 | 443.86 | 445.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 457.00 | 446.49 | 446.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:45:00 | 456.65 | 446.49 | 446.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 454.00 | 447.99 | 447.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 460.80 | 455.48 | 453.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 10:15:00 | 512.90 | 513.47 | 502.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 10:30:00 | 514.85 | 513.47 | 502.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 507.80 | 510.62 | 505.50 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 495.55 | 504.17 | 504.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 492.40 | 501.81 | 503.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 493.00 | 492.88 | 497.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:45:00 | 489.40 | 492.88 | 497.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 494.10 | 493.12 | 497.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 497.20 | 493.12 | 497.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 491.20 | 490.17 | 493.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 493.95 | 490.17 | 493.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 490.40 | 490.59 | 492.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:30:00 | 491.85 | 490.59 | 492.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 491.35 | 490.74 | 492.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 485.00 | 490.49 | 491.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 486.20 | 487.04 | 489.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 13:15:00 | 461.89 | 468.06 | 471.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 474.95 | 468.98 | 470.81 | SL hit (close>ema200) qty=0.50 sl=468.98 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 477.50 | 471.92 | 471.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 482.50 | 475.11 | 473.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 478.25 | 478.45 | 475.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 478.25 | 478.45 | 475.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 475.85 | 477.93 | 475.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 476.20 | 477.93 | 475.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 476.35 | 477.61 | 475.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 478.25 | 477.51 | 476.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 475.15 | 477.16 | 476.22 | SL hit (close<static) qty=1.00 sl=475.65 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 472.55 | 475.22 | 475.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 470.65 | 473.84 | 474.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 469.00 | 467.91 | 470.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 469.80 | 467.91 | 470.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 470.85 | 468.50 | 470.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 469.45 | 468.50 | 470.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 471.10 | 469.02 | 470.53 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 476.20 | 471.65 | 471.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 479.70 | 474.44 | 472.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 15:15:00 | 481.00 | 481.28 | 478.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 480.40 | 481.28 | 478.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 481.95 | 481.41 | 478.41 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 475.25 | 477.51 | 477.65 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 480.00 | 477.60 | 477.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 485.80 | 481.11 | 479.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 482.00 | 482.19 | 480.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 482.00 | 482.19 | 480.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 482.00 | 482.19 | 480.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 481.35 | 482.09 | 481.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 479.70 | 481.61 | 480.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 480.05 | 481.61 | 480.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 478.70 | 481.03 | 480.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 478.70 | 481.03 | 480.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 478.60 | 480.36 | 480.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 475.00 | 478.79 | 479.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 482.70 | 477.58 | 478.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 482.70 | 477.58 | 478.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 482.70 | 477.58 | 478.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 482.90 | 477.58 | 478.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 481.05 | 478.28 | 478.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 479.95 | 478.28 | 478.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 486.60 | 479.94 | 479.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 486.60 | 479.94 | 479.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 513.65 | 488.78 | 484.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 513.00 | 519.81 | 511.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 513.00 | 519.81 | 511.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 513.00 | 519.81 | 511.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 512.60 | 519.81 | 511.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 507.00 | 517.25 | 511.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 507.85 | 517.25 | 511.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 506.00 | 515.00 | 510.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 509.80 | 510.83 | 509.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 508.00 | 510.83 | 509.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 501.60 | 508.53 | 508.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 501.60 | 508.53 | 508.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 497.40 | 506.31 | 507.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 512.20 | 501.79 | 504.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 512.20 | 501.79 | 504.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 512.20 | 501.79 | 504.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 506.85 | 501.79 | 504.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 509.00 | 503.23 | 504.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 505.75 | 503.23 | 504.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 15:15:00 | 500.00 | 497.57 | 497.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 500.00 | 497.57 | 497.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 517.70 | 501.59 | 499.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 514.60 | 516.96 | 509.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 514.60 | 516.96 | 509.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 509.45 | 515.46 | 509.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:30:00 | 521.55 | 517.44 | 511.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 504.95 | 519.02 | 518.79 | SL hit (close<static) qty=1.00 sl=505.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 506.00 | 516.42 | 517.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 500.20 | 508.80 | 513.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 510.35 | 509.11 | 512.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:45:00 | 508.75 | 509.11 | 512.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 510.20 | 503.75 | 507.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:30:00 | 512.20 | 503.75 | 507.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 504.50 | 503.90 | 507.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:30:00 | 499.55 | 503.22 | 506.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 495.00 | 502.88 | 505.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 474.57 | 496.55 | 502.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 490.10 | 489.63 | 496.48 | SL hit (close>ema200) qty=0.50 sl=489.63 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 501.55 | 496.36 | 495.96 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 493.30 | 495.28 | 495.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 489.75 | 494.17 | 494.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 493.70 | 491.81 | 493.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 493.70 | 491.81 | 493.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 493.70 | 491.81 | 493.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:30:00 | 489.00 | 490.74 | 492.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 489.60 | 490.51 | 492.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 489.00 | 490.46 | 492.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 500.00 | 493.24 | 492.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 500.00 | 493.24 | 492.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 501.75 | 494.95 | 493.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 501.95 | 503.02 | 499.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 501.95 | 503.02 | 499.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 499.50 | 502.32 | 499.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 499.50 | 502.32 | 499.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 499.25 | 501.70 | 499.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:45:00 | 497.30 | 501.70 | 499.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 499.15 | 501.19 | 499.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 499.15 | 501.19 | 499.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 497.60 | 500.47 | 499.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 505.05 | 501.95 | 499.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 499.65 | 501.62 | 500.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 499.95 | 501.15 | 500.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 492.35 | 499.00 | 499.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 492.35 | 499.00 | 499.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 489.90 | 497.18 | 498.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 476.50 | 476.25 | 481.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:30:00 | 477.10 | 476.25 | 481.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 474.20 | 473.77 | 476.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 477.00 | 473.77 | 476.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 478.50 | 474.72 | 476.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 481.85 | 474.72 | 476.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 478.25 | 475.42 | 476.54 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 480.45 | 477.55 | 477.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 483.40 | 478.89 | 478.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 477.60 | 480.96 | 479.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 477.60 | 480.96 | 479.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 477.60 | 480.96 | 479.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 477.60 | 480.96 | 479.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 480.80 | 480.93 | 479.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 485.70 | 480.54 | 479.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 477.55 | 479.21 | 479.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 477.55 | 479.21 | 479.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 475.25 | 478.40 | 478.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 474.45 | 473.77 | 475.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 474.45 | 473.77 | 475.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 474.45 | 473.77 | 475.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 473.00 | 474.70 | 475.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 472.30 | 470.75 | 471.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 471.40 | 470.75 | 471.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 476.30 | 471.75 | 471.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 476.30 | 471.75 | 471.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 479.30 | 476.20 | 474.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 477.60 | 478.93 | 477.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 477.60 | 478.93 | 477.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 477.60 | 478.93 | 477.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 477.60 | 478.93 | 477.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 478.95 | 478.93 | 477.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 479.40 | 478.93 | 477.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 479.25 | 479.00 | 477.42 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 10:15:00 | 475.20 | 476.73 | 476.93 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 480.05 | 476.99 | 476.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 490.15 | 480.87 | 478.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 483.95 | 484.01 | 481.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:45:00 | 483.50 | 484.01 | 481.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 477.75 | 483.04 | 481.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 477.75 | 483.04 | 481.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 478.55 | 482.14 | 481.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 479.00 | 482.14 | 481.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 480.25 | 481.58 | 481.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 478.40 | 481.17 | 481.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 478.40 | 481.17 | 481.26 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 485.00 | 481.37 | 481.05 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 479.60 | 480.83 | 480.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 476.80 | 479.96 | 480.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 448.70 | 446.97 | 453.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 448.70 | 446.97 | 453.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 444.30 | 445.80 | 450.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 437.85 | 445.08 | 449.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 439.00 | 443.87 | 448.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:15:00 | 439.90 | 443.56 | 446.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 439.75 | 444.10 | 445.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 437.90 | 442.86 | 444.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:00:00 | 436.70 | 441.63 | 443.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 454.75 | 443.13 | 443.31 | SL hit (close>static) qty=1.00 sl=450.60 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 493.05 | 453.11 | 447.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 501.50 | 462.79 | 452.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 475.45 | 477.04 | 465.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:45:00 | 474.10 | 477.04 | 465.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 466.40 | 473.32 | 466.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:00:00 | 466.40 | 473.32 | 466.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 461.15 | 470.89 | 465.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 461.15 | 470.89 | 465.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 457.75 | 468.26 | 465.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 457.40 | 468.26 | 465.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 447.70 | 462.47 | 462.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 444.55 | 448.38 | 452.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 448.35 | 445.18 | 448.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 448.35 | 445.18 | 448.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 448.35 | 445.18 | 448.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 450.25 | 445.18 | 448.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 446.50 | 445.44 | 448.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 446.85 | 445.44 | 448.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 446.45 | 445.78 | 447.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 449.15 | 445.78 | 447.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 453.25 | 447.27 | 448.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 454.35 | 447.27 | 448.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 447.00 | 447.22 | 447.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 446.10 | 447.37 | 447.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 451.25 | 448.83 | 448.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 451.25 | 448.83 | 448.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 453.10 | 449.68 | 448.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 15:15:00 | 447.10 | 449.21 | 448.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 15:15:00 | 447.10 | 449.21 | 448.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 447.10 | 449.21 | 448.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 452.65 | 449.21 | 448.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 10:30:00 | 452.45 | 449.82 | 449.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 446.55 | 450.02 | 450.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 446.55 | 450.02 | 450.36 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 453.95 | 450.83 | 450.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 13:15:00 | 454.90 | 452.93 | 452.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 450.70 | 452.48 | 451.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 450.70 | 452.48 | 451.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 450.70 | 452.48 | 451.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 450.70 | 452.48 | 451.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 449.00 | 451.78 | 451.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 451.00 | 451.78 | 451.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 449.25 | 451.21 | 451.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 449.25 | 451.21 | 451.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 447.90 | 448.85 | 449.34 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 460.60 | 450.90 | 450.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 472.15 | 463.62 | 459.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 476.70 | 478.27 | 473.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:45:00 | 475.95 | 478.27 | 473.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 469.35 | 475.89 | 474.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 469.35 | 475.89 | 474.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 466.80 | 474.07 | 473.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 467.05 | 474.07 | 473.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 464.20 | 472.10 | 472.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 460.50 | 468.93 | 471.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 469.00 | 468.94 | 470.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 469.00 | 468.94 | 470.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 471.30 | 469.42 | 470.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 471.30 | 469.42 | 470.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 470.85 | 469.70 | 470.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 470.75 | 470.10 | 470.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 473.10 | 464.37 | 466.46 | SL hit (close>static) qty=1.00 sl=472.60 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 481.60 | 469.40 | 468.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 492.05 | 479.40 | 473.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 486.00 | 488.08 | 482.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 486.00 | 488.08 | 482.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 482.15 | 486.94 | 483.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 482.15 | 486.94 | 483.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 481.80 | 485.91 | 482.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 481.20 | 485.91 | 482.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 485.10 | 485.75 | 483.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 486.15 | 485.00 | 483.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 481.60 | 485.99 | 485.61 | SL hit (close<static) qty=1.00 sl=481.80 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 481.60 | 485.11 | 485.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 479.00 | 481.92 | 483.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 483.65 | 482.27 | 483.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 483.65 | 482.27 | 483.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 483.65 | 482.27 | 483.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 483.65 | 482.27 | 483.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 481.90 | 482.20 | 483.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:45:00 | 480.45 | 481.74 | 482.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 479.80 | 481.27 | 482.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 456.43 | 464.01 | 469.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 455.81 | 464.01 | 469.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 458.65 | 458.19 | 464.28 | SL hit (close>ema200) qty=0.50 sl=458.19 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 469.90 | 463.44 | 463.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 473.00 | 466.39 | 464.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 469.75 | 469.89 | 467.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:30:00 | 469.85 | 469.89 | 467.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 467.75 | 469.30 | 467.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 467.75 | 469.30 | 467.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 467.05 | 468.85 | 467.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:45:00 | 465.35 | 468.85 | 467.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 467.00 | 468.48 | 467.51 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 464.00 | 466.56 | 466.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 462.00 | 463.64 | 464.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 463.10 | 462.09 | 463.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 13:15:00 | 463.10 | 462.09 | 463.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 463.10 | 462.09 | 463.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 464.75 | 462.09 | 463.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 465.00 | 462.67 | 463.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 465.00 | 462.67 | 463.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 465.00 | 463.14 | 463.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 462.05 | 463.14 | 463.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 438.95 | 445.42 | 449.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 445.50 | 443.47 | 446.38 | SL hit (close>ema200) qty=0.50 sl=443.47 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 445.95 | 442.95 | 442.79 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 440.20 | 442.54 | 442.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 438.80 | 441.79 | 442.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 440.90 | 440.42 | 441.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 440.90 | 440.42 | 441.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 440.90 | 440.42 | 441.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 442.85 | 440.42 | 441.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 438.90 | 440.11 | 441.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 436.40 | 439.37 | 440.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:45:00 | 436.80 | 438.71 | 439.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 437.00 | 438.46 | 439.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 436.75 | 438.25 | 439.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 440.80 | 438.52 | 439.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 446.00 | 440.41 | 439.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 446.00 | 440.41 | 439.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 458.40 | 445.22 | 442.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 449.20 | 450.57 | 446.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:45:00 | 449.05 | 450.57 | 446.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 450.85 | 452.04 | 450.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 453.75 | 452.04 | 450.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 452.65 | 452.14 | 450.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 448.50 | 451.14 | 450.27 | SL hit (close<static) qty=1.00 sl=450.10 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 447.00 | 449.61 | 449.68 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 451.50 | 449.69 | 449.56 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 447.90 | 449.39 | 449.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 444.95 | 448.20 | 448.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 456.75 | 447.73 | 447.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 456.75 | 447.73 | 447.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 456.75 | 447.73 | 447.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 456.75 | 447.73 | 447.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 459.60 | 450.11 | 448.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 466.00 | 457.22 | 455.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 13:15:00 | 479.45 | 480.60 | 475.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:30:00 | 478.95 | 480.60 | 475.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 475.35 | 479.71 | 476.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 475.35 | 479.71 | 476.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 473.35 | 478.44 | 475.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:15:00 | 470.85 | 478.44 | 475.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 468.20 | 476.39 | 475.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 468.20 | 476.39 | 475.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 467.00 | 473.23 | 473.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 463.65 | 471.31 | 472.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 447.00 | 446.12 | 450.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 447.00 | 446.12 | 450.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 450.95 | 446.94 | 449.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 450.95 | 446.94 | 449.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 459.10 | 449.37 | 450.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:15:00 | 467.10 | 449.37 | 450.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 464.05 | 452.31 | 451.94 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 446.85 | 452.59 | 453.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 445.90 | 450.13 | 451.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 430.45 | 425.81 | 432.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 430.45 | 425.81 | 432.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 430.85 | 426.82 | 432.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 430.35 | 426.82 | 432.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 430.10 | 427.47 | 432.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 426.95 | 428.65 | 431.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 443.05 | 418.91 | 417.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 443.05 | 418.91 | 417.89 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 422.35 | 428.70 | 429.02 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 438.00 | 429.24 | 428.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 442.70 | 435.50 | 432.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 436.50 | 437.53 | 434.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:00:00 | 436.50 | 437.53 | 434.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 437.45 | 440.48 | 437.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 437.40 | 440.48 | 437.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 439.60 | 440.31 | 437.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 437.80 | 440.31 | 437.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 440.00 | 440.37 | 438.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 436.70 | 440.37 | 438.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 431.55 | 438.61 | 437.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 431.55 | 438.61 | 437.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 431.35 | 437.16 | 437.25 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 443.80 | 437.55 | 436.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 450.00 | 441.91 | 439.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 463.15 | 464.71 | 457.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:00:00 | 463.15 | 464.71 | 457.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 457.00 | 461.23 | 458.12 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 442.00 | 454.67 | 455.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 440.00 | 451.74 | 454.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 463.50 | 446.39 | 449.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 463.50 | 446.39 | 449.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 463.50 | 446.39 | 449.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 463.50 | 446.39 | 449.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 462.40 | 449.59 | 450.76 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 459.70 | 451.61 | 451.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 465.50 | 457.52 | 454.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 470.00 | 470.23 | 464.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 473.25 | 470.23 | 464.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 470.90 | 476.86 | 472.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 470.90 | 476.86 | 472.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 472.00 | 475.89 | 472.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 470.95 | 475.89 | 472.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 464.05 | 470.88 | 471.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 460.70 | 468.85 | 470.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 442.40 | 437.93 | 442.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 442.40 | 437.93 | 442.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 442.40 | 437.93 | 442.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 442.40 | 437.93 | 442.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 439.50 | 438.24 | 442.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 437.80 | 438.90 | 441.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 415.91 | 424.91 | 431.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 09:15:00 | 394.02 | 402.09 | 412.68 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 379.30 | 372.30 | 372.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 380.70 | 375.60 | 373.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 373.70 | 376.66 | 374.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 373.70 | 376.66 | 374.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 373.70 | 376.66 | 374.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 374.15 | 376.66 | 374.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 371.50 | 375.63 | 374.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 371.50 | 375.63 | 374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 373.00 | 375.10 | 374.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:30:00 | 372.30 | 375.10 | 374.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 371.60 | 374.40 | 374.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 371.60 | 374.40 | 374.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 367.85 | 373.09 | 373.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 366.00 | 371.67 | 372.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 372.85 | 371.16 | 372.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 372.85 | 371.16 | 372.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 372.85 | 371.16 | 372.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 371.20 | 371.16 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 373.75 | 371.68 | 372.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 374.00 | 371.68 | 372.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 372.90 | 371.92 | 372.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 374.05 | 371.92 | 372.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 373.00 | 372.14 | 372.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 372.80 | 372.14 | 372.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 373.30 | 372.37 | 372.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 374.50 | 372.37 | 372.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 356.85 | 348.14 | 352.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:30:00 | 345.00 | 350.63 | 352.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 327.75 | 337.06 | 343.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 340.70 | 332.20 | 338.16 | SL hit (close>ema200) qty=0.50 sl=332.20 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 354.00 | 341.60 | 341.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 362.30 | 345.74 | 343.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 354.65 | 356.38 | 350.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 10:15:00 | 352.20 | 356.38 | 350.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 381.85 | 383.68 | 379.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 380.35 | 383.68 | 379.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 381.70 | 383.28 | 380.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 381.70 | 383.28 | 380.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 378.05 | 382.24 | 379.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 386.45 | 382.24 | 379.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 383.15 | 385.88 | 383.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 421.47 | 410.58 | 406.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 403.65 | 408.55 | 409.02 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 414.45 | 407.74 | 407.61 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 407.25 | 409.96 | 409.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 406.15 | 409.20 | 409.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 407.50 | 404.92 | 406.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 407.50 | 404.92 | 406.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 407.50 | 404.92 | 406.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 403.75 | 406.29 | 406.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 404.00 | 405.83 | 406.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 404.50 | 403.51 | 404.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 402.70 | 403.86 | 404.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 402.35 | 403.56 | 404.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 409.95 | 405.66 | 405.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 409.95 | 405.66 | 405.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 411.00 | 406.72 | 405.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 409.95 | 410.54 | 408.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 11:15:00 | 409.15 | 410.54 | 408.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 409.35 | 410.30 | 408.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 409.20 | 410.30 | 408.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 409.35 | 410.11 | 408.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:30:00 | 407.80 | 410.11 | 408.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 409.20 | 409.93 | 408.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 408.05 | 409.93 | 408.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 463.00 | 2025-05-21 11:15:00 | 460.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-06-24 12:30:00 | 485.00 | 2025-07-02 13:15:00 | 461.89 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-06-24 12:30:00 | 485.00 | 2025-07-03 09:15:00 | 474.95 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-06-25 09:30:00 | 486.20 | 2025-07-03 11:15:00 | 477.50 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2025-07-04 14:30:00 | 478.25 | 2025-07-07 09:15:00 | 475.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-21 09:15:00 | 479.95 | 2025-07-21 09:15:00 | 486.60 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-24 14:30:00 | 509.80 | 2025-07-25 09:15:00 | 501.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-24 15:15:00 | 508.00 | 2025-07-25 09:15:00 | 501.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-28 11:15:00 | 505.75 | 2025-07-31 15:15:00 | 500.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-08-04 10:30:00 | 521.55 | 2025-08-06 10:15:00 | 504.95 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-08-08 12:30:00 | 499.55 | 2025-08-11 09:15:00 | 474.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 12:30:00 | 499.55 | 2025-08-11 13:15:00 | 490.10 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-08-08 15:15:00 | 495.00 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-12 11:00:00 | 499.60 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-08-12 12:00:00 | 496.70 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-12 14:45:00 | 491.05 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-08-18 11:30:00 | 489.00 | 2025-08-19 10:15:00 | 500.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-08-18 13:00:00 | 489.60 | 2025-08-19 10:15:00 | 500.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-18 14:45:00 | 489.00 | 2025-08-19 10:15:00 | 500.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-08-21 09:30:00 | 505.05 | 2025-08-22 09:15:00 | 492.35 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-08-21 13:15:00 | 499.65 | 2025-08-22 09:15:00 | 492.35 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-08-21 15:00:00 | 499.95 | 2025-08-22 09:15:00 | 492.35 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-09-03 09:15:00 | 485.70 | 2025-09-03 15:15:00 | 477.55 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-08 10:30:00 | 473.00 | 2025-09-10 13:15:00 | 476.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-10 10:30:00 | 472.30 | 2025-09-10 13:15:00 | 476.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-10 11:15:00 | 471.40 | 2025-09-10 13:15:00 | 476.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-18 13:15:00 | 479.00 | 2025-09-19 09:15:00 | 478.40 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-09-18 14:30:00 | 480.25 | 2025-09-19 09:15:00 | 478.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-30 09:15:00 | 437.85 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-09-30 10:00:00 | 439.00 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-30 14:15:00 | 439.90 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-10-06 09:15:00 | 439.75 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-10-06 11:00:00 | 436.70 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2025-10-15 10:15:00 | 446.10 | 2025-10-15 12:15:00 | 451.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-16 09:15:00 | 452.65 | 2025-10-17 15:15:00 | 446.55 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-16 10:30:00 | 452.45 | 2025-10-17 15:15:00 | 446.55 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-24 09:15:00 | 451.00 | 2025-10-24 10:15:00 | 449.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-11-07 15:00:00 | 470.75 | 2025-11-11 10:15:00 | 473.10 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-11 11:15:00 | 467.55 | 2025-11-12 11:15:00 | 481.60 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-11-12 11:00:00 | 469.25 | 2025-11-12 11:15:00 | 481.60 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-11-17 09:15:00 | 486.15 | 2025-11-18 11:15:00 | 481.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-19 14:45:00 | 480.45 | 2025-11-24 11:15:00 | 456.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:45:00 | 479.80 | 2025-11-24 11:15:00 | 455.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 14:45:00 | 480.45 | 2025-11-25 09:15:00 | 458.65 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-11-20 11:45:00 | 479.80 | 2025-11-25 09:15:00 | 458.65 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-12-03 09:15:00 | 462.05 | 2025-12-08 13:15:00 | 438.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 462.05 | 2025-12-09 12:15:00 | 445.50 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-17 12:00:00 | 436.40 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-12-18 09:45:00 | 436.80 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-12-18 14:15:00 | 437.00 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-12-18 15:15:00 | 436.75 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-24 09:15:00 | 453.75 | 2025-12-24 11:15:00 | 448.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-24 09:45:00 | 452.65 | 2025-12-24 11:15:00 | 448.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-23 10:45:00 | 426.95 | 2026-01-30 09:15:00 | 443.05 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2026-02-26 15:15:00 | 437.80 | 2026-03-02 12:15:00 | 415.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 437.80 | 2026-03-05 09:15:00 | 394.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 345.00 | 2026-03-30 09:15:00 | 327.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 345.00 | 2026-03-30 14:15:00 | 340.70 | STOP_HIT | 0.50 | 1.25% |
| BUY | retest2 | 2026-04-10 09:15:00 | 386.45 | 2026-04-22 09:15:00 | 421.47 | TARGET_HIT | 1.00 | 9.06% |
| BUY | retest2 | 2026-04-13 09:45:00 | 383.15 | 2026-04-22 10:15:00 | 425.10 | TARGET_HIT | 1.00 | 10.95% |
| SELL | retest2 | 2026-05-05 09:15:00 | 403.75 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-05-05 10:00:00 | 404.00 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-05-06 10:30:00 | 404.50 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-05-06 12:15:00 | 402.70 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.80% |
